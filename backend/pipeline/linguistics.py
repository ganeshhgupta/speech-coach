"""linguistics.py — deterministic text/timing analysis of the transcript.

All thresholds here are heuristic assumptions (documented in README), not
clinical or scientifically validated cutoffs.
"""
import re
from collections import Counter
from .schema import LinguisticMetrics, PauseMetrics, Word

FILLERS = ["um", "uh", "ah", "erm", "hmm", "like", "basically", "actually",
           "you know", "i mean", "so yeah", "literally", "right"]

HEDGES = ["maybe", "probably", "perhaps", "i think", "i guess", "i feel like",
          "sort of", "kind of", "possibly", "i suppose", "not sure but",
          "could be wrong", "i believe"]

PAUSE_THRESHOLD_SEC = 0.4
LONG_PAUSE_THRESHOLD_SEC = 1.2
LONG_SENTENCE_WORDS = 35


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower())


def _count_phrases(norm_text: str, phrases: list[str]) -> dict:
    counts = {}
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        n = len(re.findall(pattern, norm_text))
        if n:
            counts[phrase] = n
    return counts


def _sentences(full_text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", full_text.strip())
    return [p for p in parts if p]


def _repeated_ngrams(norm_text: str, n: int = 3, min_repeats: int = 2) -> list[str]:
    tokens = [t for t in norm_text.split() if t]
    grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return [g for g, c in counts.items() if c >= min_repeats]


def compute_pauses(words: list[Word]) -> PauseMetrics:
    if len(words) < 2:
        return PauseMetrics(0, 0, 0.0, 0.0, 0.0)

    gaps = []
    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap > PAUSE_THRESHOLD_SEC:
            gaps.append(gap)

    if not gaps:
        return PauseMetrics(0, 0, 0.0, 0.0, 0.0)

    long_pauses = [g for g in gaps if g > LONG_PAUSE_THRESHOLD_SEC]
    return PauseMetrics(
        pause_count=len(gaps),
        long_pause_count=len(long_pauses),
        total_pause_sec=round(sum(gaps), 2),
        avg_pause_sec=round(sum(gaps) / len(gaps), 2),
        longest_pause_sec=round(max(gaps), 2),
    )


def compute_linguistics(full_text: str, words: list[Word], duration_sec: float) -> LinguisticMetrics:
    norm = _normalize(full_text)
    word_count = len(words) if words else len(norm.split())

    speaking_time = (words[-1].end - words[0].start) if len(words) >= 2 else duration_sec
    speaking_time = max(speaking_time, 0.01)

    wpm = round(word_count / (speaking_time / 60.0), 1)

    filler_breakdown = _count_phrases(norm, FILLERS)
    filler_count = sum(filler_breakdown.values())
    filler_rate = round(filler_count / (speaking_time / 60.0), 2)

    hedge_breakdown = _count_phrases(norm, HEDGES)
    hedge_count = sum(hedge_breakdown.values())
    hedge_rate = round(hedge_count / (speaking_time / 60.0), 2)

    sentences = _sentences(full_text)
    sentence_lens = [len(s.split()) for s in sentences] if sentences else [word_count]
    avg_sentence_len = round(sum(sentence_lens) / len(sentence_lens), 1) if sentence_lens else 0.0
    long_sentence_count = sum(1 for l in sentence_lens if l > LONG_SENTENCE_WORDS)

    repeated = _repeated_ngrams(norm)

    trailing_off = sum(
        1 for s in sentences
        if not s.strip().endswith((".", "!", "?")) and len(s.split()) > 2
    )

    return LinguisticMetrics(
        duration_sec=round(duration_sec, 2),
        speaking_time_sec=round(speaking_time, 2),
        word_count=word_count,
        wpm=wpm,
        filler_count=filler_count,
        filler_rate_per_min=filler_rate,
        filler_breakdown=filler_breakdown,
        hedge_count=hedge_count,
        hedge_rate_per_min=hedge_rate,
        hedge_breakdown=hedge_breakdown,
        avg_sentence_len_words=avg_sentence_len,
        long_sentence_count=long_sentence_count,
        repeated_phrases=repeated,
        trailing_off_count=trailing_off,
    )
