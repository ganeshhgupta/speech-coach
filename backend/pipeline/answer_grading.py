"""answer_grading.py — grades interview-answer correctness/completeness via
the Claude API. Only question + transcript TEXT is sent, never audio.

Best-effort like coach.try_llm_summary: any missing key, network error, or
malformed response degrades to AnswerGrading(available=False, ...) rather
than failing the request. The deterministic delivery score never depends on
this.
"""
import json
import os

from .schema import AnswerGrading

CLAUDE_MODEL = os.environ.get("SC_CLAUDE_MODEL", "claude-sonnet-5")

_SYSTEM_PROMPT = """You are grading a spoken interview answer for correctness and completeness, given only the question and a transcript of the answer. Judge the substance: is it factually correct, does it actually answer what was asked, is it complete. Ignore filler words, disfluencies, and phrasing, those are scored separately elsewhere.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": <0-100 integer>, "summary": "<2-3 sentence verdict>", "strengths": ["<short point>", ...], "gaps": ["<short point>", ...]}

"strengths" and "gaps" should each have 0-4 short bullet points. If the transcript doesn't answer the question at all, score it low and say so in the summary."""


def grade_answer(question: str, transcript: str) -> AnswerGrading:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return AnswerGrading(available=False, unavailable_reason="ANTHROPIC_API_KEY not set.")

    try:
        import anthropic
    except ImportError:
        return AnswerGrading(available=False, unavailable_reason="anthropic package not installed.")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Question:\n{question}\n\nAnswer transcript:\n{transcript}",
            }],
        )
        raw = "".join(block.text for block in resp.content if block.type == "text").strip()
        # Strip accidental markdown fences even though the prompt asks against them.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):raw.rfind("}") + 1]
        data = json.loads(raw)

        score = data.get("score")
        score = max(0, min(100, int(score))) if score is not None else None

        return AnswerGrading(
            available=True,
            correctness_score=score,
            summary=data.get("summary"),
            strengths=list(data.get("strengths") or [])[:4],
            gaps=list(data.get("gaps") or [])[:4],
        )
    except Exception as e:  # noqa: BLE001 - grading is optional; degrade, don't 500 the request
        return AnswerGrading(available=False, unavailable_reason=str(e))
