"""transcribe.py — speech-to-text with word-level timestamps.

Two backends, chosen by SC_TRANSCRIBE_BACKEND:
- "local" (default): Faster-Whisper running in-process. True word-level
  timestamps, no network dependency, but needs real CPU/RAM — too much for
  Render's free-tier instance (confirmed OOM-killing it under load).
- "hf": Hugging Face's hosted Inference API. Offloads the heavy compute off
  the container entirely; used for the Render deploy (see render.yaml).
  The shared serverless API only returns chunk-level timestamps (not
  per-word), so word boundaries within a chunk are linearly interpolated —
  an approximation, good enough for WPM/filler-rate/pause heuristics, not
  frame-accurate.
"""
import base64
import os
import time

import requests

from .schema import Word

BACKEND = os.environ.get("SC_TRANSCRIBE_BACKEND", "local")

MODEL_NAME = os.environ.get("SC_WHISPER_MODEL", "small.en")
DEVICE = os.environ.get("SC_WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("SC_WHISPER_COMPUTE", "int8")

HF_ASR_MODEL = os.environ.get("SC_HF_ASR_MODEL", "openai/whisper-large-v3")
# api-inference.huggingface.co (the old classic endpoint) is retired; HF now
# routes serverless inference through router.huggingface.co per provider.
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_ASR_MODEL}"


def transcribe(wav_path: str):
    """Returns (full_text: str, words: list[Word], segments: list[dict])."""
    return transcribe_with_progress(wav_path)


def transcribe_with_progress(wav_path: str, on_progress=None):
    """Same as transcribe(), but calls on_progress(seg.end) as transcription
    reaches further into the audio (local backend: after each segment; hf
    backend: only once, at completion, since it's a single blocking call —
    the caller's own heartbeat covers the gap in between)."""
    if BACKEND == "hf":
        return _transcribe_via_hf(wav_path, on_progress=on_progress)
    return _transcribe_local(wav_path, on_progress=on_progress)


def _get_local_model():
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _cached():
        from faster_whisper import WhisperModel  # lazy: only needed for the local backend
        return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    return _cached()


def _transcribe_local(wav_path: str, on_progress=None):
    model = _get_local_model()
    segments_iter, _info = model.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    words: list[Word] = []
    segments: list[dict] = []
    full_text_parts = []

    for seg in segments_iter:
        seg_text = seg.text.strip()
        full_text_parts.append(seg_text)
        segments.append({"start": seg.start, "end": seg.end, "text": seg_text})
        if seg.words:
            for w in seg.words:
                words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
        if on_progress:
            on_progress(seg.end)

    return " ".join(full_text_parts), words, segments


def _transcribe_via_hf(wav_path: str, on_progress=None):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("SC_TRANSCRIBE_BACKEND=hf but HUGGINGFACE_TOKEN is not set.")

    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    body = {"inputs": audio_b64, "parameters": {"return_timestamps": True}}
    headers = {"Authorization": f"Bearer {token}"}

    # The shared serverless model can be "cold" (unloaded) on first call and
    # returns 503 with an estimated_time while it spins up — retry through that.
    deadline = time.monotonic() + 90
    last_error = None
    while time.monotonic() < deadline:
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=body, timeout=60)
        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(3)
            continue

        if resp.status_code == 503:
            try:
                wait = min(float(resp.json().get("estimated_time", 5)), 20)
            except (ValueError, TypeError):
                wait = 5
            time.sleep(wait)
            continue

        if not resp.ok:
            raise RuntimeError(f"HF Inference API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        full_text = (data.get("text") or "").strip()
        chunks = data.get("chunks") or []
        words = _chunks_to_words(chunks)
        segments = [
            {"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c.get("text", "").strip()}
            for c in chunks
            if c.get("timestamp") and c["timestamp"][0] is not None and c["timestamp"][1] is not None
        ]
        if on_progress and segments:
            on_progress(segments[-1]["end"])
        return full_text, words, segments

    raise RuntimeError(f"HF Inference API did not become ready in time (last error: {last_error}).")


def _chunks_to_words(chunks: list) -> list:
    """HF's shared inference API returns chunk-level timestamps, not
    per-word. Approximate word boundaries by splitting each chunk's text and
    linearly spreading it across the chunk's [start, end] span. Good enough
    for WPM/filler-rate/pause heuristics; not frame-accurate."""
    words: list[Word] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp") or [None, None]
        start, end = ts
        if start is None or end is None or not text:
            continue
        tokens = text.split()
        span = max(end - start, 0.01)
        step = span / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(Word(text=tok, start=start + i * step, end=start + (i + 1) * step))
    return words
