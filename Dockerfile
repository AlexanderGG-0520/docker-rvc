# syntax=docker/dockerfile:1.7

# ====== CUDA Runtime + Python ======
# GTX1660Sなら CUDA 12系でOK（ホストのNVIDIAドライバは対応版が必要）
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- OS deps ----
# ffmpeg: wav<->pcm変換やデバッグに便利
# libsndfile1: soundfile系が必要になった時に楽
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# torch は CUDA対応 wheel を入れる（デフォルトでcu124を見に行く）
# もし torch のバージョン固定したいなら requirements.txt に torch==... を書く
# その場合は --index-url をここで指定しておくと確実
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch \
 && python3 -m pip install --no-cache-dir -r /app/requirements.txt

# ---- App ----
COPY app /app/app

# ---- Runtime env ----
# /models をPVCでマウントする前提
ENV RVC_MODELS_DIR="/models" \
    RVC_DEFAULT_SR="48000" \
    RVC_DEFAULT_CH="1"

# ---- User (non-root) ----
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]