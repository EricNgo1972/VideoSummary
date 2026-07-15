#!/usr/bin/env python3
"""
HTTP API for the MapleCam video summarizer.

For when MapleCam runs in a different container / on a different box and can't
share a folder. Callers POST either a whole video or (better) a few
pre-extracted keyframes, and get the summary JSON back.

Endpoints:
  GET  /health                 -> {"status":"ok", ...}
  POST /summarize              -> summary JSON
      multipart/form-data fields:
        frames / images  (repeatable)  pre-extracted keyframe JPGs  [preferred]
        video            (single file) a whole clip; server runs ffmpeg
        name             (optional)    label for the clip in the output
        timestamps       (optional)    comma-separated seconds, one per keyframe

Inference is serialized behind a lock so CPU-only boxes don't thrash when
several requests land at once. Served by waitress (production WSGI).
"""

import os
import tempfile
import threading

from flask import Flask, jsonify, request
from waitress import serve as waitress_serve
from werkzeug.utils import secure_filename

import worker


PORT = int(os.environ.get("PORT", "8080"))
API_KEY = os.environ.get("API_KEY", "")               # optional shared secret
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
THREADS = int(os.environ.get("SERVER_THREADS", "4"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# CPU is the bottleneck: only one summarization runs at a time. Extra requests
# block on this lock rather than fighting for cores.
_PROC_LOCK = threading.Lock()


def log(*a):
    print("[server]", *a, flush=True)


@app.before_request
def _auth():
    if API_KEY and request.endpoint != "health":
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify(error="unauthorized"), 401
    return None


@app.get("/health")
def health():
    return jsonify(status="ok", vlm_model=worker.VLM_MODEL, max_frames=worker.MAX_FRAMES)


def _parse_timestamps(raw):
    if not raw:
        return None
    out = []
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out or None


@app.post("/summarize")
def summarize():
    name = request.form.get("name") or "upload"
    images = request.files.getlist("frames") + request.files.getlist("images")
    video = request.files.get("video")

    if not images and not video:
        return jsonify(error="provide 'frames'/'images' file(s) or a 'video' file"), 400

    timestamps = _parse_timestamps(request.form.get("timestamps"))

    with tempfile.TemporaryDirectory(prefix="upload_") as tmp:
        try:
            if images:
                paths = []
                for i, f in enumerate(images):
                    safe = secure_filename(f.filename or f"kf_{i:04d}.jpg")
                    p = os.path.join(tmp, f"{i:04d}_{safe}")
                    f.save(p)
                    paths.append(p)
                with _PROC_LOCK:
                    result, text = worker.summarize_keyframes(name, paths, timestamps)
            else:
                safe = secure_filename(video.filename or "clip.mp4")
                vp = os.path.join(tmp, safe or "clip.mp4")
                video.save(vp)
                with _PROC_LOCK:
                    result, text = worker.summarize_video_named(vp, name)
        except Exception as e:  # noqa: BLE001 - report cleanly instead of 500 HTML
            log(f"ERROR summarizing {name}: {e}")
            return jsonify(error=str(e), video=name), 500

    result["text"] = text
    return jsonify(result)


def main():
    log(f"listening on 0.0.0.0:{PORT} (auth={'on' if API_KEY else 'off'}, "
        f"max_upload={MAX_UPLOAD_MB}MB, model={worker.VLM_MODEL})")
    waitress_serve(app, host="0.0.0.0", port=PORT, threads=THREADS)


if __name__ == "__main__":
    main()
