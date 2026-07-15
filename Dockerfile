# Single self-contained image: Ollama + moondream + ffmpeg + the app.
# CPU only. No GPU runtime required.
#
# Layer order is deliberate: the expensive, rarely-changing model bake comes
# FIRST (it depends only on the base image's ollama binary), then system deps,
# then Python deps, then the app source. That way editing worker.py/server.py
# rebuilds only a tiny COPY layer — the ~1.6 GB model layer stays cached and is
# never re-pulled on deploy.
FROM ollama/ollama:latest

# Vision model baked into the image so provisioning needs no network pull.
ARG VLM_MODEL=moondream
ENV VLM_MODEL=${VLM_MODEL}

# Bake the vision model into the image: briefly run the server, pull, stop.
# Placed early so it only invalidates when the base image or model changes.
RUN ollama serve & \
    server_pid=$! ; \
    until ollama list >/dev/null 2>&1; do sleep 1; done ; \
    ollama pull "${VLM_MODEL}" ; \
    kill "$server_pid" 2>/dev/null || true ; \
    sleep 2

# System deps: python, ffmpeg (brings ffprobe), curl for healthchecks.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps (container-global; PEP 668 override is fine inside an image).
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

# App source — the layer that actually changes between releases.
COPY worker.py /app/worker.py
COPY server.py /app/server.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Runtime config + HTTP API (MODE=serve|both). Ollama stays internal on 11434.
ENV INPUT_DIR=/data/in \
    OUTPUT_DIR=/data/out \
    MODE=serve \
    PORT=8080
EXPOSE 8080
WORKDIR /app

# The base image sets ENTRYPOINT to ollama; override it with ours.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:11434/api/tags >/dev/null || exit 1
