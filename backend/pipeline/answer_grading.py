"""answer_grading.py — grades interview-answer correctness/completeness.
Only question + transcript TEXT is ever sent, never audio.

Two ways to grade, tried in this order:
1. CLAUDE_CODE_OAUTH_TOKEN -> a headless `claude -p` CLI call. Uses the
   Claude Code subscription instead of separate metered API billing.
2. ANTHROPIC_API_KEY -> the `anthropic` SDK directly. Used as a fallback if
   the CLI isn't installed/reachable (e.g. the Render deploy image, which
   doesn't bundle Node/the CLI) or CLAUDE_CODE_OAUTH_TOKEN isn't set.

Neither configured -> AnswerGrading(available=False, ...), same
graceful-degradation pattern as coach.try_llm_summary. The deterministic
delivery score never depends on either path.
"""
import json
import os
import subprocess
import tempfile

from .schema import AnswerGrading

CLAUDE_MODEL = os.environ.get("SC_CLAUDE_MODEL", "claude-sonnet-5")

# Isolated config dir for the nested CLI call so it never touches (or relies
# on) the interactive session's own login state — auth here comes purely
# from CLAUDE_CODE_OAUTH_TOKEN.
_CLI_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "speech-coach-claude-cli")
os.makedirs(_CLI_CONFIG_DIR, exist_ok=True)

_SYSTEM_PROMPT = """You are grading a spoken interview answer for correctness and completeness, given only the question and a transcript of the answer. Judge the substance: is it factually correct, does it actually answer what was asked, is it complete. Ignore filler words, disfluencies, and phrasing, those are scored separately elsewhere.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"score": <0-100 integer>, "summary": "<2-3 sentence verdict>", "strengths": ["<short point>", ...], "gaps": ["<short point>", ...]}

"strengths" and "gaps" should each have 0-4 short bullet points. If the transcript doesn't answer the question at all, score it low and say so in the summary."""


def grade_answer(question: str, transcript: str) -> AnswerGrading:
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if oauth_token:
        result = _grade_via_cli(question, transcript, oauth_token)
        if result.available or not api_key:
            return result
        # CLI path failed (binary missing, timed out, etc.) but an API key
        # is also configured — fall through and try that instead of giving up.

    if api_key:
        return _grade_via_sdk(question, transcript, api_key)

    return AnswerGrading(
        available=False,
        unavailable_reason="Neither CLAUDE_CODE_OAUTH_TOKEN nor ANTHROPIC_API_KEY is set.",
    )


def _grade_via_cli(question: str, transcript: str, oauth_token: str) -> AnswerGrading:
    prompt = f"Question:\n{question}\n\nAnswer transcript:\n{transcript}"
    env = dict(os.environ)
    env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    env["CLAUDE_CONFIG_DIR"] = _CLI_CONFIG_DIR

    try:
        proc = subprocess.run(
            ["claude", "-p", prompt,
             "--system-prompt", _SYSTEM_PROMPT,
             "--model", CLAUDE_MODEL,
             "--output-format", "json"],
            env=env, capture_output=True, text=True, timeout=90,
        )
    except FileNotFoundError:
        return AnswerGrading(available=False, unavailable_reason="claude CLI not installed on this host.")
    except subprocess.TimeoutExpired:
        return AnswerGrading(available=False, unavailable_reason="claude CLI call timed out.")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        return AnswerGrading(available=False, unavailable_reason=f"claude CLI returned non-JSON output: {detail}")

    if envelope.get("is_error"):
        # The real error text lands in "result" on error envelopes, not stderr.
        return AnswerGrading(available=False, unavailable_reason=envelope.get("result") or "claude CLI reported an error.")

    return _parse_grading_json(envelope.get("result", ""))


def _grade_via_sdk(question: str, transcript: str, api_key: str) -> AnswerGrading:
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
        return _parse_grading_json(raw)
    except Exception as e:  # noqa: BLE001 - grading is optional; degrade, don't 500 the request
        return AnswerGrading(available=False, unavailable_reason=str(e))


def _parse_grading_json(raw: str) -> AnswerGrading:
    raw = raw.strip()
    # Strip accidental markdown fences even though the prompt asks against them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):raw.rfind("}") + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return AnswerGrading(available=False, unavailable_reason=f"Could not parse grading response: {e}")

    score = data.get("score")
    score = max(0, min(100, int(score))) if score is not None else None

    return AnswerGrading(
        available=True,
        correctness_score=score,
        summary=data.get("summary"),
        strengths=list(data.get("strengths") or [])[:4],
        gaps=list(data.get("gaps") or [])[:4],
    )
