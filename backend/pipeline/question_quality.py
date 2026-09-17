"""question_quality.py: classifies each question sentence in the recording
as open-ended/probing vs closed/procedural, and measures how long the other
speaker took to respond to it.

Deterministic linguistic heuristic, no training data, no external corpus,
no cloud call. This is deliberately a different design than a trained
classifier: it needs zero setup, ships with the code, and its rules are
auditable in this file rather than baked into opaque model weights.

Response latency is measured per question using real diarized turn
boundaries (via conversation.py's turns), not a linear question-then-answer
assumption over the transcript, so it stays correct even when the "wrong"
speaker answers, interrupts, or asks a follow-up before responding.
"""
from .schema import Word

OPEN_ENDED_STARTERS = (
    "what", "why", "how", "tell me", "describe", "walk me through",
    "explain", "what's", "what would", "what kind", "what do you think",
    "what happens", "what are",
)
CLOSED_STARTERS = (
    "do you", "does", "is it", "is this", "are you", "can you", "have you",
    "did you", "would you like", "so you", "you have", "was it",
)
SUBSTANTIVE_WORD_COUNT = 8       # questions at or above this length lean open-ended
MAX_PLAUSIBLE_RESPONSE_SEC = 30  # longer gaps are treated as a topic change, not a "response," and dropped


def analyze_questions(exclusive_turns: list[dict], words: list[Word], roles: dict[str, str]) -> dict:
    """Returns per-role question-quality breakdown and average response
    latency to that role's questions, keyed by role name."""
    turns = _turns_with_words(exclusive_turns, words)

    sentences = []  # flat, chronological list across the whole recording
    for idx, t in enumerate(turns):
        for s in _split_sentences(t["words"]):
            sentences.append({**s, "speaker": t["speaker"], "turn_index": idx})

    quality_counts: dict[str, dict] = {}
    latencies: dict[str, list[float]] = {}

    for i, s in enumerate(sentences):
        if not s["text"].strip().endswith("?"):
            continue

        role = roles.get(s["speaker"], s["speaker"])
        quality = _classify_question(s["text"])

        bucket = quality_counts.setdefault(role, {"open-ended": 0, "closed": 0, "unclear": 0, "total": 0})
        bucket[quality] += 1
        bucket["total"] += 1

        response_gap = _next_response_latency(sentences, i)
        if response_gap is not None:
            latencies.setdefault(role, []).append(response_gap)

    question_quality_by_role = {
        role: {
            **counts,
            "open_ended_pct": round(100 * counts["open-ended"] / counts["total"], 1) if counts["total"] else 0.0,
        }
        for role, counts in quality_counts.items()
    }

    response_latency_by_role = {
        role: {"avg_response_sec": round(sum(v) / len(v), 2), "n": len(v)}
        for role, v in latencies.items() if v
    }

    return {
        "question_quality_by_role": question_quality_by_role,
        "response_latency_by_role": response_latency_by_role,
    }


def _turns_with_words(exclusive_turns: list[dict], words: list[Word]) -> list[dict]:
    """Assigns each transcript word to the one diarized turn it falls
    inside (by midpoint), preserving chronological turn order."""
    turns = [dict(t, words=[]) for t in exclusive_turns]
    for w in words:
        mid = (w.start + w.end) / 2
        for t in turns:
            if t["start"] <= mid <= t["end"]:
                t["words"].append(w)
                break
    return turns


def _split_sentences(turn_words: list[Word]) -> list[dict]:
    """Groups one turn's words into sentences on terminal punctuation,
    keeping the first/last word's timestamps as the sentence's span."""
    sentences = []
    current: list[Word] = []
    for w in turn_words:
        current.append(w)
        if w.text.strip().endswith((".", "!", "?")):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [
        {"text": " ".join(w.text for w in s), "start": s[0].start, "end": s[-1].end}
        for s in sentences if s
    ]


def _classify_question(text: str) -> str:
    lower = text.strip().lower()
    score = 0
    if lower.startswith(OPEN_ENDED_STARTERS):
        score += 1
    if lower.startswith(CLOSED_STARTERS):
        score -= 1
    if len(text.split()) >= SUBSTANTIVE_WORD_COUNT:
        score += 1

    if score >= 1:
        return "open-ended"
    if score <= -1:
        return "closed"
    return "unclear"


def _next_response_latency(sentences: list[dict], question_idx: int) -> float | None:
    """Finds the next sentence spoken by a DIFFERENT speaker than the
    question and returns the gap in seconds, or None if no plausible
    response follows (e.g. it was the last question, or the gap is long
    enough to be a topic change rather than a reply)."""
    question = sentences[question_idx]
    for later in sentences[question_idx + 1:]:
        if later["speaker"] == question["speaker"]:
            continue
        gap = later["start"] - question["end"]
        if 0 <= gap < MAX_PLAUSIBLE_RESPONSE_SEC:
            return gap
        return None
    return None
