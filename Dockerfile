# Single self-contained image: Ollama + moondream + ffmpeg + the watcher.
# CPU only. No GPU runtime required.
FROM ollama/ollama:latest

# Vision model baked into the image so provisioning needs no network pull.
ARG VLM_MODEL=moondream

ENV DEBIAN_FRONTEND=noninteractive \
    INPUT_DIR=/data/in \
    OUTPUT_DIR=/data/out \
    VLM_MODEL=${VLM_MODEL}

# System deps: python, ffmpeg (brings ffprobe), curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps (container-global; PEP 668 override is fine inside an image).
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Bake the vision model into the image: briefly run the server, pull, stop.
RUN ollama serve & \
    server_pid=$! ; \
    until ollama list >/dev/null 2>&1; do sleep 1; done ; \
    ollama pull "${VLM_MODEL}" ; \
    kill "$server_pid" 2>/dev/null || true ; \
    sleep 2

COPY worker.py /app/worker.py
COPY server.py /app/server.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# HTTP API (MODE=serve|both). Ollama stays internal on 11434.
EXPOSE 8080
ENV MODE=serve PORT=8080
WORKDIR /app

# The base image sets ENTRYPOINT to ollama; override it with ours.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:11434/api/tags >/dev/null || exit 1
