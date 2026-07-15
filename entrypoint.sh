#!/usr/bin/env bash
# Start Ollama in the background, make sure the vision model is present,
# then hand off to the watcher. One container, no GPU.
set -euo pipefail

MODEL="${VLM_MODEL:-moondream}"

echo "[entrypoint] starting ollama serve..."
ollama serve &
OLLAMA_PID=$!

# Stop ollama cleanly if the container is signalled.
trap 'echo "[entrypoint] stopping..."; kill "$OLLAMA_PID" 2>/dev/null || true' TERM INT

echo "[entrypoint] waiting for ollama to come up..."
until ollama list >/dev/null 2>&1; do
  # Bail out early if the server process died.
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[entrypoint] ollama serve exited unexpectedly" >&2
    exit 1
  fi
  sleep 1
done

# The model is baked at build time; this is a safety net (e.g. if VLM_MODEL
# was overridden, or an optional SUMMARY_MODEL needs pulling).
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  echo "[entrypoint] pulling $MODEL..."
  ollama pull "$MODEL"
fi
if [ -n "${SUMMARY_MODEL:-}" ] && ! ollama show "$SUMMARY_MODEL" >/dev/null 2>&1; then
  echo "[entrypoint] pulling $SUMMARY_MODEL..."
  ollama pull "$SUMMARY_MODEL"
fi

MODE="${MODE:-serve}"   # serve (HTTP API) | watch (folder) | both
echo "[entrypoint] mode=$MODE"
case "$MODE" in
  serve)
    exec python3 /app/server.py
    ;;
  watch)
    exec python3 /app/worker.py
    ;;
  both)
    python3 /app/worker.py &
    exec python3 /app/server.py
    ;;
  *)
    echo "[entrypoint] unknown MODE '$MODE' (use serve|watch|both)" >&2
    exit 1
    ;;
esac
