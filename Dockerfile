FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VOXCPM_MODEL=openbmb/VoxCPM2 \
    VOXCPM_OPENAI_MODEL_NAME=voxcpm-tts \
    VOXCPM_VOICE_LIBRARY_DIR=/data/voices \
    VOXCPM_DEVICE=cuda \
    VOXCPM_PRELOAD_MODEL=true \
    VOXCPM_LOAD_DENOISER=false \
    VOXCPM_OPTIMIZE=false \
    PORT=3017

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /data/voices/files

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchaudio==2.10.0 && \
    pip install . && \
    find /usr/local/lib/python3.11 -type d -name __pycache__ -prune -exec rm -rf {} +

EXPOSE 3017

CMD ["python", "-m", "voxcpm.openai_server", "--host", "0.0.0.0", "--port", "3017"]
