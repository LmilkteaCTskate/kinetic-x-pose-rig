FROM python:3.12-slim

# Build argument: cpu (default lightweight) or cu126 (NVIDIA GPU server)
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KINETIC_HOST=0.0.0.0 \
    KINETIC_PORT=8000 \
    KINETIC_DEVICE=auto

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--device", "auto"]
