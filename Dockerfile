# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ===== OS deps =====
# build-essential/python3-dev: pyworld/numba等で必要になりがち
# ffmpeg/sox: 音声I/O・変換
# libsndfile1: soundfile系
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git \
    build-essential cmake \
    ffmpeg sox \
    libsndfile1 \
    libopenblas-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===== PyTorch (CUDA) =====
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchaudio

# ===== Clone official RVC repo =====
# ここは「固定」しろ。latest追従すると突然壊れる。
# 例: RVC_REF をコミットSHAにしておくのが一番安全
ARG RVC_REPO="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
ARG RVC_REF="main"
RUN git clone --depth 1 --branch "${RVC_REF}" "${RVC_REPO}" /opt/rvc

# ===== App deps =====
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# ===== App code =====
COPY app /app/app

# ===== Runtime env =====
# PVCはここにマウントする前提
ENV RVC_MODELS_DIR="/models" \
    PYTHONPATH="/opt/rvc:/app"

RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]