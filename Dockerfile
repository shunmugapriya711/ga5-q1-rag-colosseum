FROM python:3.12-slim

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/app/.cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app.py knowledge_base.json ./
# Pre-download the embedding model at build time so the first probe is fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860
# Render/Railway/Koyeb inject $PORT; HF Spaces uses 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
