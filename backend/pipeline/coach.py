"""coach.py — turns extracted features into concrete, actionable findings.

Deterministic rules run first and always produce a report (no dependency).
If a local Ollama server is reachable, its output is layered on top as a
short narrative summary — never as the source of the metrics themselves.
"""
import os
import requests
from .schema import LinguisticMetrics, PauseMetrics, ProsodyMetrics, Finding
from .conversation import MIN_INTERRUPTION_SEC

OLLAMA_URL = os.environ.get("SC_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("SC_OLLAMA_MODEL", "llama3.1")


def _finding(severity, category, message, evidence):
    return Finding(severity=severity, category=category, message=message, evidence=evidence)


def build_findings(ling: LinguisticMetrics, pauses: PauseMetrics, prosody: ProsodyMetrics) -> list[Finding]:
    findings: list[Finding] = []
    minutes = max(ling.speaking_time_sec / 60.0, 0.01)

    # --- Pace ---
    if ling.wpm > 170:
        findings.append(_finding("flag", "pace",
            "You speak faster than a comfortable conversational pace.",
            f"{ling.wpm} words/min (comfortable range: ~110-170)."))
    elif ling.wpm < 110:
        findings.append(_finding("watch", "pace",
            "Your pace is on the slow side; listeners may lose engagement.",
            f"{ling.wpm} words/min (comfortable range: ~110-170)."))

    # --- Fillers ---
    if ling.filler_rate_per_min > 6:
        top = sorted(ling.filler_breakdown.items(), key=lambda x: -x[1])[:3]
        findings.append(_finding("flag", "fillers",
            "You use filler words heavily, which can undercut perceived confidence.",
            f"{ling.filler_rate_per_min}/min ({ling.filler_count} total). Top: {top}."))
    elif ling.filler_rate_per_min > 2:
        findings.append(_finding("watch", "fillers",
            "Moderate filler-word usage.",
            f"{ling.filler_rate_per_min}/min ({ling.filler_count} total)."))

    # --- Hedging ---
    if ling.hedge_rate_per_min > 5:
        top = sorted(ling.hedge_breakdown.items(), key=lambda x: -x[1])[:3]
        findings.append(_finding("flag", "hedging",
            "You hedge frequently, which can read as uncertainty even when you're sure.",
            f"{ling.hedge_rate_per_min}/min ({ling.hedge_count} total). Top: {top}."))
    elif ling.hedge_rate_per_min > 2:
        findings.append(_finding("watch", "hedging",
            "Some hedging language present.",
            f"{ling.hedge_rate_per_min}/min ({ling.hedge_count} total)."))

    # --- Rambling / sentence length ---
    if ling.avg_sentence_len_words > 28 or ling.long_sentence_count >= 3:
        findings.append(_finding("flag", "structure",
            "You tend toward long, run-on sentences, which can read as rambling.",
            f"avg sentence length {ling.avg_sentence_len_words} words; "
            f"{ling.long_sentence_count} sentences over {35} words."))

    # --- Repetition ---
    if ling.repeated_phrases:
        findings.append(_finding("watch", "repetition",
            "You repeat certain short phrases across the recording.",
            f"Repeated 3-word phrases: {ling.repeated_phrases[:5]}"))

    # --- Trailing off ---
    if ling.trailing_off_count >= 3:
        findings.append(_finding("watch", "structure",
            "Several segments trail off without a clear end to the thought.",
            f"{ling.trailing_off_count} segments ended without terminal punctuation."))

    # --- Pauses ---
    long_pause_rate = pauses.long_pause_count / minutes
    if long_pause_rate > 3:
        findings.append(_finding("flag", "pausing",
            "Frequent long pauses interrupt your flow.",
            f"{pauses.long_pause_count} pauses over {1.2}s "
            f"({round(long_pause_rate, 1)}/min), longest {pauses.longest_pause_sec}s."))
    elif pauses.pause_count / minutes < 2 and ling.wpm > 150:
        findings.append(_finding("watch", "pausing",
            "You rarely pause, especially at a fast pace — listeners get little room to absorb points.",
            f"only {pauses.pause_count} pauses over {0.4}s across the recording."))

    # --- Monotone / pitch variation ---
    if prosody.pitch_cv is not None:
        if prosody.pitch_cv < 0.12:
            findings.append(_finding("flag", "vocal-variety",
                "Your pitch stays quite flat, which can sound monotone.",
                f"pitch coefficient of variation {prosody.pitch_cv} "
                f"(mean {prosody.pitch_mean_hz} Hz, stdev {prosody.pitch_stdev_hz} Hz)."))
        elif prosody.pitch_cv < 0.18:
            findings.append(_finding("watch", "vocal-variety",
                "Somewhat limited pitch variation.",
                f"pitch coefficient of variation {prosody.pitch_cv}."))

    if not findings:
        findings.append(_finding("info", "overall",
            "No strong issues detected against the heuristic thresholds used here.",
            "All measured metrics fell within the comfortable ranges."))

    return findings


def build_conversation_findings(conv: dict) -> list[Finding]:
    """Interview-aware findings from diarization output. Role labels
    ("Interviewer"/"Interviewee") are a heuristic guess, not ground truth."""
    findings: list[Finding] = []
    by_role = {s["role"]: s for s in conv["speakers"]}
    interviewee = by_role.get("Interviewee")
    interviewer = by_role.get("Interviewer")

    if interviewee and interviewee["talk_ratio"] < 0.45:
        findings.append(_finding("watch", "talk-ratio",
            "The interviewee held less than half the talk time; answers may be running short.",
            f"Interviewee talk ratio {round(interviewee['talk_ratio'] * 100)}% "
            f"({interviewee['talk_time_sec']}s of {round(interviewee['talk_time_sec'] + (interviewer['talk_time_sec'] if interviewer else 0))}s)."))
    elif interviewee and interviewee["talk_ratio"] > 0.85:
        findings.append(_finding("info", "talk-ratio",
            "The interviewee held the large majority of talk time.",
            f"Interviewee talk ratio {round(interviewee['talk_ratio'] * 100)}%."))

    total_interruptions = len(conv["interruptions"])
    if total_interruptions >= 3:
        by_interrupter: dict[str, int] = {}
        for ev in conv["interruptions"]:
            by_interrupter[ev["interrupter_role"]] = by_interrupter.get(ev["interrupter_role"], 0) + 1
        top = sorted(by_interrupter.items(), key=lambda x: -x[1])
        findings.append(_finding("flag" if total_interruptions >= 6 else "watch", "interruptions",
            "Multiple instances of overlapping speech (one speaker talking over the other).",
            f"{total_interruptions} overlaps over {MIN_INTERRUPTION_SEC}s. By who interrupted: {top}."))
    elif total_interruptions == 0:
        findings.append(_finding("info", "interruptions",
            "No significant overlapping speech detected.",
            "Turn-taking was clean throughout."))

    if conv["avg_turn_gap_sec"] > 2.0:
        findings.append(_finding("watch", "turn-taking",
            "Noticeable silence between turns; the conversation doesn't flow quickly.",
            f"Average gap between speaker turns: {conv['avg_turn_gap_sec']}s."))

    return findings


def try_llm_summary(transcript: str, ling: LinguisticMetrics, pauses: PauseMetrics,
                     prosody: ProsodyMetrics, findings: list[Finding]) -> str | None:
    """Best-effort: ask a local Ollama model to write a short human-readable
    summary from the ALREADY-COMPUTED metrics. Returns None if unreachable —
    the app must work fully without this."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
    except requests.RequestException:
        return None

    bullet_findings = "\n".join(f"- [{f.severity}] {f.category}: {f.message} ({f.evidence})" for f in findings)
    prompt = f"""You are a communication coach. Based ONLY on the measured data below
(do not invent anything not listed), write a short (4-6 sentence) coaching
summary in plain, direct language. No filler, no hedging, be specific.

Metrics:
- words/min: {ling.wpm}
- filler rate/min: {ling.filler_rate_per_min}
- hedge rate/min: {ling.hedge_rate_per_min}
- avg sentence length: {ling.avg_sentence_len_words} words
- long pauses: {pauses.long_pause_count} (longest {pauses.longest_pause_sec}s)
- pitch variation (CV): {prosody.pitch_cv}

Detected findings:
{bullet_findings}
"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except requests.RequestException:
        return None
