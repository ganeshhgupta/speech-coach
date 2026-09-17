"""app.py — FastAPI backend: upload audio, run the pipeline, stream progress, return a report."""
import asyncio
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import audio_io, transcribe, linguistics, prosody, coach, conversation
from pipeline.schema import Report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
TMP_DIR = os.path.join(os.path.dirname(BASE_DIR), "storage", "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}

app = FastAPI(title="Speech Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_pipeline_worker(src_path: str, wav_path: str, progress_q: "queue.Queue", diarize: bool):
    """Runs in a background thread. Puts progress dicts on progress_q as work
    happens, a final {"stage": "done"|"error", ...} dict, then None as an
    end-of-stream sentinel."""
    try:
        progress_q.put({"stage": "decoding"})
        duration = audio_io.decode_to_wav(src_path, wav_path)
        if duration < 1.0:
            progress_q.put({"stage": "error", "message": "Audio too short to analyze (minimum ~1s)."})
            return

        progress_q.put({"stage": "transcribing", "progress": 0.0, "duration_sec": round(duration, 1)})
        t0 = time.monotonic()

        def on_progress(processed_sec: float):
            elapsed = time.monotonic() - t0
            frac = min(processed_sec / duration, 1.0) if duration > 0 else 0.0
            eta = (elapsed / frac - elapsed) if frac > 0.02 else None
            progress_q.put({
                "stage": "transcribing",
                "progress": round(frac, 4),
                "processed_sec": round(min(processed_sec, duration), 1),
                "duration_sec": round(duration, 1),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1) if eta is not None else None,
            })

        full_text, words, segments = transcribe.transcribe_with_progress(wav_path, on_progress=on_progress)
        if not words:
            progress_q.put({"stage": "error", "message": "No speech detected in the audio."})
            return

        progress_q.put({"stage": "analyzing"})
        ling = linguistics.compute_linguistics(full_text, words, duration)
        pauses = linguistics.compute_pauses(words)
        pros = prosody.compute_prosody(wav_path)
        findings = coach.build_findings(ling, pauses, pros)

        conv = None
        if diarize:
            try:
                from pipeline import diarization  # lazy: only pulls in torch/pyannote when actually used

                def on_diarize_progress(step_name: str, completed: int, total: int):
                    progress_q.put({
                        "stage": "diarizing",
                        "step": step_name,
                        "progress": round(completed / total, 3) if total else 0.0,
                    })

                progress_q.put({"stage": "diarizing", "step": "starting", "progress": 0.0})
                turns, exclusive_turns = diarization.diarize(wav_path, num_speakers=2, on_progress=on_diarize_progress)
                conv = conversation.compute_conversation(turns, exclusive_turns, words)
                findings += coach.build_conversation_findings(conv)
            except Exception as e:  # noqa: BLE001 - diarization is optional; fall back to single-speaker report
                progress_q.put({"stage": "diarization_error", "message": str(e)})

        llm_summary = coach.try_llm_summary(full_text, ling, pauses, pros, findings)

        report = Report(
            transcript=full_text,
            linguistics=ling,
            pauses=pauses,
            prosody=pros,
            findings=findings,
            llm_summary=llm_summary,
            conversation=conv,
        )
        progress_q.put({"stage": "done", "report": report.to_dict()})
    except Exception as e:  # noqa: BLE001 - surfaced to the client as an error event
        progress_q.put({"stage": "error", "message": str(e)})
    finally:
        progress_q.put(None)
        for p in (src_path, wav_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


async def _stream_pipeline(src_path: str, wav_path: str, diarize: bool):
    progress_q: "queue.Queue" = queue.Queue()
    thread = threading.Thread(target=_run_pipeline_worker, args=(src_path, wav_path, progress_q, diarize), daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, progress_q.get)
        if item is None:
            break
        yield (json.dumps(item) + "\n").encode("utf-8")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), diarize: bool = Form(False)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex
    src_path = os.path.join(TMP_DIR, f"{job_id}_src{ext}")
    wav_path = os.path.join(TMP_DIR, f"{job_id}.wav")

    with open(src_path, "wb") as f:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(src_path)
                raise HTTPException(400, "File too large (max 200MB).")
            f.write(chunk)

    return StreamingResponse(_stream_pipeline(src_path, wav_path, diarize), media_type="application/x-ndjson")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
