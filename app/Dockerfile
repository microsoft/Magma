FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    HF_HUB_CACHE=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    PORT=7860 \
    HOST=0.0.0.0 \
    ENV=production \
    MAGMA_LORA_REPO=Wafair/DOME \
    MAGMA_CHECKPOINT_DIR=/data/huggingface/magma-office-lora \
    OMNIPARSER_WEIGHTS_DIR=/home/user/app/server/weights

# Runtime libs for opencv, easyocr, ultralytics + a free TTF font for SoM annotation
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsm6 libxext6 libglib2.0-0 libgomp1 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires uid 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /data/huggingface && chown -R user:user /data

USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH"

# Install Python deps (pip installs to /home/user/.local when running as non-root)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "fastapi>=0.114" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.20" \
        "pydantic>=2.9" \
        "httpx>=0.27" \
        "python-dotenv>=1.0" \
        "Pillow>=10" \
        "numpy<2" \
        "git+https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2" \
        "peft>=0.10" \
        "accelerate>=0.26" \
        "bitsandbytes>=0.42" \
        "huggingface-hub>=0.20" \
        "ultralytics>=8.0" \
        "opencv-python-headless>=4.8" \
        "easyocr>=1.7" \
        "open-clip-torch>=2.20" \
        "timm>=0.9" \
        "einops>=0.7" \
        "sentencepiece>=0.2"

# Copy server code (no client, no agent — those run on the user's machine)
COPY --chown=user:user server /home/user/app/server

# Symlink the system Liberation TTF to where som.py looks for arial.ttf
RUN mkdir -p /home/user/app/server/src/domains/task/fonts && \
    ln -sf /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
           /home/user/app/server/src/domains/task/fonts/arial.ttf

WORKDIR /home/user/app/server
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
