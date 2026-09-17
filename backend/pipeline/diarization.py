"""diarization.py: pyannote.audio speaker diarization.

Needs HUGGINGFACE_TOKEN (a free HF account, with the pyannote/speaker-
diarization-3.1 and pyannote/segmentation-3.0 model terms accepted on
huggingface.co) set in the environment. Degrades by raising a clear
RuntimeError if missing; callers decide whether that's fatal or skippable.

pyannote.audio 4.x returns both an overlap-preserving annotation and an
"exclusive" one (one speaker per moment, overlaps resolved). We use the
overlap-preserving one to detect true interruptions (both speakers active
at once) and the exclusive one for talk-ratio / word-to-speaker alignment,
where a single owner per moment is what you actually want.
"""
import os
from functools import lru_cache

from pyannote.audio import Pipeline

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
PYANNOTE_MODEL = os.environ.get("SC_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    if not HF_TOKEN:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN not set. Diarization needs a Hugging Face token with access to "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 (accept their terms "
            "at huggingface.co first, then generate a token at huggingface.co/settings/tokens), "
            "set HUGGINGFACE_TOKEN in backend/.env."
        )
    return Pipeline.from_pretrained(PYANNOTE_MODEL, token=HF_TOKEN)


class _QueueHook:
    """Adapts pyannote's (step_name, artifact, file, total, completed) callback
    signature into a plain on_progress(step_name, completed, total) callback."""

    def __init__(self, on_progress):
        self.on_progress = on_progress

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1
        self.on_progress(step_name, completed, total)


def diarize(wav_path: str, num_speakers: int = 2, on_progress=None):
    """Returns (turns, exclusive_turns): both lists of
    {"speaker": str, "start": float, "end": float}, sorted by start time.

    `turns` preserves overlapping speech (use for interruption detection).
    `exclusive_turns` assigns each moment to exactly one speaker (use for
    talk-ratio and aligning transcript words to a speaker).
    """
    pipeline = get_pipeline()
    hook = _QueueHook(on_progress) if on_progress else None
    output = pipeline(wav_path, num_speakers=num_speakers, hook=hook)

    turns = _annotation_to_turns(output.speaker_diarization)
    exclusive_turns = _annotation_to_turns(output.exclusive_speaker_diarization)
    return turns, exclusive_turns


def _annotation_to_turns(annotation) -> list[dict]:
    turns = [
        {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns
