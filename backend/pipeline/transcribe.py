"""transcribe.py — Faster-Whisper transcription with word-level timestamps."""
import os
from functools import lru_cache
from faster_whisper import WhisperModel
from .schema import Word

MODEL_NAME = os.environ.get("SC_WHISPER_MODEL", "small.en")
DEVICE = os.environ.get("SC_WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("SC_WHISPER_COMPUTE", "int8")


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe(wav_path: str):
    """Returns (full_text: str, words: list[Word], segments: list[dict])."""
    return transcribe_with_progress(wav_path)


def transcribe_with_progress(wav_path: str, on_progress=None):
    """Same as transcribe(), but calls on_progress(seg.end) after each segment
    Faster-Whisper yields, so a caller can report how far into the audio
    (in seconds) transcription has reached."""
    model = get_model()
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
