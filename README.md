# Interview Analyzer

A local, offline speech-communication analyzer. Upload a recording (practice
run, presentation, interview, your side of a call) and get back pace,
filler/hedge rate, sentence-structure, pause, and pitch-variation findings,
computed from real audio, not guessed from a transcript alone. For
two-speaker interview recordings, it also detects talk-ratio, turn-taking,
and true overlapping-speech interruptions.

Everything runs on your machine. Audio is decoded, transcribed, and analyzed
locally; nothing is uploaded to a third-party API. The only optional network
calls are downloading model weights on first use (Whisper, and pyannote's
diarization model if you enable two-speaker mode).

## Features

- **Transcript with word-level timestamps** via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- **Linguistic analysis**: words per minute, filler-word rate (`um`, `like`,
  `you know`...), hedge-word rate (`I think`, `maybe`, `sort of`...), average
  sentence length, repeated phrases, trailing-off sentences
- **Pause analysis**: count, frequency, and duration of pauses over 0.4s and 1.2s
- **Prosody** via [Parselmouth](https://github.com/YannickJadoul/Parselmouth)
  (Praat): pitch mean/variation, intensity, voiced fraction
- **Rule-based coaching findings**, severity-tagged (info / watch / flag),
  always computed, no dependency on a model call
- **Optional local-LLM narrative summary**: if an [Ollama](https://ollama.com)
  server is running, a short human-readable summary is layered on top of the
  deterministic findings; if not, the findings stand alone
- **Two-speaker mode** (interviewer / interviewee): real speaker diarization
  via [pyannote.audio](https://github.com/pyannote/pyannote-audio), talk-ratio,
  turn count, turn-taking gaps, and true overlapping-speech interruption
  detection (not just turn-boundary heuristics)
- **Live progress while it works**: a tqdm-style progress bar streamed over
  NDJSON, showing exactly how far into the audio transcription has reached
  and an ETA, not a spinner

## Quickstart

```bash
git clone https://github.com/ganeshhgupta/interview-analyzer.git
cd interview-analyzer/backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8731
```

Open `http://127.0.0.1:8731`, drop in an audio file, done.

First transcription run downloads the Whisper `small.en` model (~150MB) to
`~/.cache/huggingface`; after that it's fully offline. Supported formats:
wav, mp3, m4a, mp4, ogg, flac, webm, up to 200MB.

> **Windows note:** if `pip install -r requirements.txt` fails to resolve
> `torch==...+cpu`, install torch first with
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`,
> then re-run the requirements install.

## Two-speaker mode (interviewer / interviewee)

Check "Two speakers" before uploading to get talk-ratio, turn-taking, and
interruption detection. This needs a Hugging Face token with access to the
diarization model, one-time setup:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Accept the terms on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (read access is enough).
4. Create `backend/.env` (never committed) with:
   ```
   HUGGINGFACE_TOKEN=hf_your_token_here
   ```

Without a token, leave "Two speakers" unchecked for the full single-speaker
report. With it checked but no token set, analysis degrades gracefully: a
"speaker detection failed, continuing without it" note appears and you still
get the normal report, minus the conversation section.

`torch` + `pyannote.audio` (~2-3GB combined) are only pulled in for this
mode; the base install stays light without them.

Role labeling ("Interviewer" vs "Interviewee") is a heuristic: whichever
speaker asks more questions relative to their own talk time is called the
interviewer. It can guess wrong; use the "Swap" button in the Conversation
panel to flip it.

## How it works

```text
interview-analyzer/
├── backend/
│   ├── app.py                 FastAPI: POST /api/analyze (streams NDJSON progress + final report), GET /api/health
│   ├── requirements.txt
│   └── pipeline/
│       ├── audio_io.py        decode any container -> mono 16kHz WAV (PyAV)
│       ├── transcribe.py      Faster-Whisper, word-level timestamps, streams per-segment progress
│       ├── linguistics.py     WPM, fillers, hedges, sentence length, pauses, repetition
│       ├── prosody.py         Parselmouth (Praat): pitch, intensity, voiced fraction
│       ├── diarization.py     pyannote.audio: who's speaking when (two-speaker mode)
│       ├── conversation.py    talk-ratio, turns, interruptions from diarization + transcript words
│       ├── coach.py           rule-based findings + optional Ollama narrative layer
│       └── schema.py          dataclasses shared across the pipeline
├── frontend/
│   └── index.html             drag-drop upload, live progress bar, report view, no build step, no framework
└── storage/tmp/                per-request scratch space (uploads deleted immediately after analysis)
```

A single `POST /api/analyze` request streams newline-delimited JSON: a
`decoding` event, then repeated `transcribing` events carrying `progress`,
`elapsed_sec`, and `eta_sec` as Whisper works through the audio, an
`analyzing` event, `diarizing` events if two-speaker mode is on, and a final
`done` event carrying the complete report. The frontend reads this stream
directly via `fetch()` and a `ReadableStream` reader, no WebSocket or
polling needed.

`pipeline/diarization.py` runs pyannote's pretrained pipeline once and
returns two views of the same speaker turns: one that preserves overlapping
speech (used to detect real interruptions, both speakers active at once)
and one with overlaps resolved to a single speaker per moment (used for
talk-ratio and aligning Whisper's word timestamps to a speaker).
`pipeline/conversation.py` combines both with the transcript to compute
talk-ratio, turn count, turn-taking gaps, and interruption events,
deterministic, no further model calls. Diarization currently assumes exactly
two speakers (`num_speakers=2`); more speakers would need generalizing the
role-labeling heuristic beyond binary Interviewer/Interviewee.

## Configuration

All optional, read from the environment (`backend/.env`, never committed):

| Variable | Default | Purpose |
|---|---|---|
| `HUGGINGFACE_TOKEN` | unset | Enables two-speaker diarization mode |
| `SC_WHISPER_MODEL` | `small.en` | Faster-Whisper model size |
| `SC_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SC_WHISPER_COMPUTE` | `int8` | Faster-Whisper compute type |
| `SC_PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model to load |
| `SC_OLLAMA_URL` | `http://localhost:11434` | Local Ollama server for the optional narrative summary |
| `SC_OLLAMA_MODEL` | `llama3.1` | Ollama model name |

## Thresholds are heuristic, not clinical

Every cutoff in `coach.py` (e.g. filler rate > 6/min = "flag", pitch CV <
0.12 = "monotone") is a reasonable-sounding starting point, not a validated
clinical norm. Treat findings as signal worth investigating, not verdicts.
Adjust the constants at the top of `linguistics.py` and the thresholds in
`coach.py` once you have a feel for your own baseline recordings.

## Roadmap / explicitly not built

| Feature | Status |
|---|---|
| Emotion recognition (Wav2Vec2-style) | Not built, treated as a noisy, low-confidence signal in the original design; add only if you want a "likely frustrated/excited" tag on top of the deterministic findings |
| Holistic audio-LLM reasoning (Qwen2-Audio / SALMONN-style) | Not built, heavy local install, marginal value over the rule-based layer |
| Longitudinal multi-session pattern tracking | Out of scope for now; add a `storage/sessions/` write in `app.py` if you want trend tracking across recordings |
| 3+ speaker diarization | Not built; pipeline is hardcoded to `num_speakers=2` |

## License

No license file yet, all rights reserved by default. Add one (MIT is a
reasonable default for a project like this) if you want others to reuse it.
