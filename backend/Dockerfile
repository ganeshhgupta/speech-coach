# Render deploy image for Practice Answer mode (no diarization — see requirements-render.txt).
# Transcription runs via the Hugging Face Inference API (SC_TRANSCRIBE_BACKEND=hf),
# not a local Whisper model — the free-tier instance doesn't have the RAM/CPU
# to run Whisper in-process without OOM-crashing (confirmed live 2026-09-23).
FROM python:3.11-slim

# soundfile's wheel usually bundles libsndfile, but installing it explicitly
# avoids a hard-to-diagnose runtime import failure if it doesn't on a given base image.
# Node is for the `claude` CLI (answer_grading.py's CLAUDE_CODE_OAUTH_TOKEN path) —
# without it, grading silently falls back to ANTHROPIC_API_KEY if set, else unavailable.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get purge -y curl gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements-render.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY frontend/ ../frontend/

EXPOSE 8731
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8731}"]
