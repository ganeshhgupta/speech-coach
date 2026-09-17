"""conversation.py: turns diarization output + transcript words into
talk-ratio, turn-taking, and interruption metrics for a 2-speaker
(interviewer/interviewee) recording.

Deterministic, no model calls here, diarization.py already did the only
ML step. Role labeling ("Interviewer" vs "Interviewee") is a heuristic:
whichever speaker asks more questions per minute of their own talk time is
called the interviewer, ties broken toward whoever talked less overall.
It can be wrong; the frontend lets you swap the labels.
"""
from .schema import Word
from .question_quality import analyze_questions

MIN_INTERRUPTION_SEC = 0.3  # overlaps shorter than this are treated as backchannel noise, not a real interruption


def compute_conversation(turns: list[dict], exclusive_turns: list[dict], words: list[Word]) -> dict:
    speaker_ids = sorted({t["speaker"] for t in exclusive_turns})

    talk_time = {s: 0.0 for s in speaker_ids}
    for t in exclusive_turns:
        talk_time[t["speaker"]] += t["end"] - t["start"]
    total_talk = sum(talk_time.values()) or 1.0

    words_by_speaker: dict[str, list[Word]] = {s: [] for s in speaker_ids}
    unassigned_words = 0
    for w in words:
        speaker = _speaker_at(exclusive_turns, (w.start + w.end) / 2)
        if speaker:
            words_by_speaker[speaker].append(w)
        else:
            unassigned_words += 1

    turn_count_by_speaker = {s: 0 for s in speaker_ids}
    for t in exclusive_turns:
        turn_count_by_speaker[t["speaker"]] += 1

    question_count = {s: _count_questions(words_by_speaker[s]) for s in speaker_ids}

    interruption_events = _detect_interruptions(turns)
    interruptions_made = {s: 0 for s in speaker_ids}
    interruptions_received = {s: 0 for s in speaker_ids}
    for ev in interruption_events:
        interruptions_made[ev["interrupter"]] = interruptions_made.get(ev["interrupter"], 0) + 1
        interruptions_received[ev["interrupted"]] = interruptions_received.get(ev["interrupted"], 0) + 1

    roles = _assign_roles(speaker_ids, talk_time, question_count)

    speakers = [{
        "speaker_id": s,
        "role": roles[s],
        "talk_time_sec": round(talk_time[s], 1),
        "talk_ratio": round(talk_time[s] / total_talk, 3),
        "turn_count": turn_count_by_speaker[s],
        "word_count": len(words_by_speaker[s]),
        "question_count": question_count[s],
        "interruptions_made": interruptions_made.get(s, 0),
        "interruptions_received": interruptions_received.get(s, 0),
    } for s in speaker_ids]

    gaps = [
        b["start"] - a["end"]
        for a, b in zip(exclusive_turns, exclusive_turns[1:])
        if b["speaker"] != a["speaker"] and b["start"] > a["end"]
    ]

    interruptions = [{
        "time_sec": ev["time_sec"],
        "overlap_sec": ev["overlap_sec"],
        "interrupter_role": roles.get(ev["interrupter"], ev["interrupter"]),
        "interrupted_role": roles.get(ev["interrupted"], ev["interrupted"]),
    } for ev in interruption_events]

    question_analysis = analyze_questions(exclusive_turns, words, roles)

    return {
        "speakers": speakers,
        "interruptions": interruptions,
        "turn_count": len(exclusive_turns),
        "avg_turn_sec": round(sum(t["end"] - t["start"] for t in exclusive_turns) / len(exclusive_turns), 2) if exclusive_turns else 0.0,
        "avg_turn_gap_sec": round(sum(gaps) / len(gaps), 2) if gaps else 0.0,
        "unassigned_words": unassigned_words,
        "question_quality_by_role": question_analysis["question_quality_by_role"],
        "response_latency_by_role": question_analysis["response_latency_by_role"],
    }


def _speaker_at(exclusive_turns: list[dict], t: float) -> str | None:
    for turn in exclusive_turns:
        if turn["start"] <= t <= turn["end"]:
            return turn["speaker"]
    return None


def _count_questions(words: list[Word]) -> int:
    return sum(1 for w in words if w.text.strip().endswith("?"))


def _detect_interruptions(turns: list[dict]) -> list[dict]:
    """turns: overlap-preserving diarization turns, sorted by start. Flags
    every cross-speaker overlap of at least MIN_INTERRUPTION_SEC, attributing
    'interrupter' to whichever turn started later."""
    events = []
    active: list[dict] = []
    for turn in turns:
        active = [a for a in active if a["end"] > turn["start"]]
        for other in active:
            if other["speaker"] == turn["speaker"]:
                continue
            overlap = min(turn["end"], other["end"]) - turn["start"]
            if overlap >= MIN_INTERRUPTION_SEC:
                events.append({
                    "time_sec": round(turn["start"], 2),
                    "interrupter": turn["speaker"],
                    "interrupted": other["speaker"],
                    "overlap_sec": round(overlap, 2),
                })
        active.append(turn)
    return events


def _assign_roles(speaker_ids: list[str], talk_time: dict, question_count: dict) -> dict:
    if len(speaker_ids) != 2:
        return {s: f"Speaker {chr(65 + i)}" for i, s in enumerate(speaker_ids)}

    a, b = speaker_ids
    rate_a = question_count[a] / max(talk_time[a], 1.0)
    rate_b = question_count[b] / max(talk_time[b], 1.0)

    if rate_a == rate_b:
        interviewer = a if talk_time[a] <= talk_time[b] else b
    else:
        interviewer = a if rate_a > rate_b else b
    interviewee = b if interviewer == a else a
    return {interviewer: "Interviewer", interviewee: "Interviewee"}
