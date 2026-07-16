#!/usr/bin/env python3
"""
MapleCam video summarizer worker.

Watches a folder for short surveillance clips, samples a handful of frames with
ffmpeg, captions each frame with a small vision model served by Ollama, and
writes a text + JSON summary next to (or beside) each clip.

Designed for CPU-only boxes: single worker, one clip at a time, bounded frame
count, downscaled frames. No GPU, no audio.

Run modes:
  python3 worker.py            # watch INPUT_DIR forever
  python3 worker.py --once F   # process one file and print the summary (for tests)
"""

import base64
import glob
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


# ----------------------------------------------------------------------------
# Config (all overridable via environment)
# ----------------------------------------------------------------------------
def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


INPUT_DIR = _env("INPUT_DIR", "/data/in")
OUTPUT_DIR = _env("OUTPUT_DIR", "")            # empty => write beside the input clip
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = _env("VLM_MODEL", "moondream")
SUMMARY_MODEL = _env("SUMMARY_MODEL", "")       # optional text model to refine the summary

MAX_FRAMES = int(_env("MAX_FRAMES", "6"))
MIN_FRAMES = int(_env("MIN_FRAMES", "3"))
FRAME_MODE = _env("FRAME_MODE", "scene").lower()  # "scene" (with uniform fallback) or "uniform"
SCENE_THRESHOLD = float(_env("SCENE_THRESHOLD", "0.3"))
FRAME_WIDTH = int(_env("FRAME_WIDTH", "640"))     # downscale width fed to the VLM
NUM_PREDICT = int(_env("NUM_PREDICT", "120"))     # cap caption length (CPU time)

# Keep this a PLAIN, DIRECT question. Moondream (and similar small VLMs) choke on
# over-instructed prompts — a conditional like "if nothing notable, say 'no
# activity'" makes it emit a single stop token and return an empty string. A
# straight question reliably yields an accurate description of people/vehicles.
CAPTION_PROMPT = _env(
    "CAPTION_PROMPT",
    "Are there any people, vehicles, or animals in this image? "
    "Describe what they are doing.",
)

STABLE_SECONDS = int(_env("STABLE_SECONDS", "3"))   # wait for the file to stop growing
PROCESS_EXISTING = _env("PROCESS_EXISTING", "true").lower() == "true"
FORCE = _env("FORCE", "false").lower() == "true"    # reprocess even if a summary exists
VIDEO_EXTS = tuple(
    e if e.startswith(".") else "." + e
    for e in _env("VIDEO_EXTS", ".mp4,.mkv,.avi,.mov,.m4v,.ts").lower().split(",")
)
REQUEST_TIMEOUT = int(_env("REQUEST_TIMEOUT", "300"))


def log(*a):
    print("[worker]", *a, flush=True)


# ----------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ----------------------------------------------------------------------------
def get_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=False,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return 0.0


def _evenly_pick(items, n):
    """Pick at most n items evenly spaced across the list."""
    if len(items) <= n:
        return items
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def extract_scene(path, tmpdir):
    """Return [(pts_seconds_or_None, jpg_path), ...] at scene changes, downscaled."""
    pattern = os.path.join(tmpdir, "s_%04d.jpg")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo,scale={FRAME_WIDTH}:-1",
         "-vsync", "vfr", "-frames:v", str(MAX_FRAMES * 3), pattern],
        capture_output=True, text=True, check=False,
    )
    times = re.findall(r"pts_time:([0-9.]+)", proc.stderr)
    files = sorted(glob.glob(os.path.join(tmpdir, "s_*.jpg")))
    out = []
    for i, f in enumerate(files):
        t = float(times[i]) if i < len(times) else None
        out.append((t, f))
    return out


def extract_uniform(path, tmpdir, n, duration):
    """Grab n frames evenly spaced across the clip, downscaled."""
    if duration and duration > 0:
        stamps = [duration * (i + 1) / (n + 1) for i in range(n)]
    else:
        stamps = [0.0]
    frames = []
    for i, t in enumerate(stamps):
        outp = os.path.join(tmpdir, f"u_{i:04d}.jpg")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
             "-vf", f"scale={FRAME_WIDTH}:-1", outp],
            check=False,
        )
        if os.path.exists(outp):
            frames.append((t, outp))
    return frames


def select_frames(path, tmpdir):
    duration = get_duration(path)
    frames = []
    if FRAME_MODE == "scene":
        scene = extract_scene(path, tmpdir)
        if len(scene) >= MIN_FRAMES:
            frames = _evenly_pick(scene, MAX_FRAMES)
    if not frames:
        frames = extract_uniform(path, tmpdir, MAX_FRAMES, duration)
    return duration, frames


# ----------------------------------------------------------------------------
# Ollama calls
# ----------------------------------------------------------------------------
def caption_frame(image_path):
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    payload = {
        "model": VLM_MODEL,
        "prompt": CAPTION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0.1},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return " ".join(r.json().get("response", "").split()).strip()


def refine_summary(timeline_text):
    prompt = (
        "The following are timestamped observations from a short security "
        "camera clip. Write a factual 1-3 sentence summary of what happened. "
        "Do not invent details.\n\n"
        f"{timeline_text}\n\nSummary:"
    )
    payload = {
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 160, "temperature": 0.2},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return " ".join(r.json().get("response", "").split()).strip()


# ----------------------------------------------------------------------------
# Summarization
# ----------------------------------------------------------------------------
def fmt_ts(t):
    if t is None:
        return "--:--"
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _is_noise(text):
    t = text.lower()
    return ("no activity" in t) or ("nothing" in t and "notable" in t)


def build_overview(captions):
    """Template overview with no extra model: dedupe informative captions."""
    seen, uniq = set(), []
    for _, txt in captions:
        if _is_noise(txt):
            continue
        key = re.sub(r"[^a-z ]", "", txt.lower())[:60]
        if key and key not in seen:
            seen.add(key)
            uniq.append(txt)
    if not uniq:
        return "No notable activity detected in the sampled frames."
    if len(uniq) == 1:
        return uniq[0]
    return " ".join(uniq[:3])


def caption_frames(frames):
    """frames: [(t_or_None, jpg_path), ...] -> [(t_or_None, caption), ...]"""
    captions = []
    for t, fp in frames:
        try:
            txt = caption_frame(fp)
        except Exception as e:  # noqa: BLE001 - keep going on a bad frame
            txt = f"(caption failed: {e})"
        captions.append((t, txt))
    return captions


def assemble_result(name, duration, captions):
    """Turn per-frame captions into the summary result + text form."""
    timeline_lines = [f"[{fmt_ts(t)}] {txt}" for t, txt in captions]
    timeline_text = "\n".join(timeline_lines)

    if SUMMARY_MODEL:
        try:
            overview = refine_summary(timeline_text)
        except Exception as e:  # noqa: BLE001 - fall back to template
            log(f"{name}: summary model failed ({e}); using template")
            overview = build_overview(captions)
    else:
        overview = build_overview(captions)

    result = {
        "video": name,
        "duration_seconds": round(duration, 2),
        "frames_analyzed": len(captions),
        "vlm_model": VLM_MODEL,
        "summary": overview,
        "timeline": [{"t": t, "ts": fmt_ts(t), "caption": txt} for t, txt in captions],
    }
    text_out = f"{name}\nDuration: {duration:.1f}s\n\nSummary:\n{overview}\n\nTimeline:\n{timeline_text}\n"
    return result, text_out


def summarize_video(path):
    return summarize_video_named(path, os.path.basename(path))


def summarize_video_named(path, name):
    """Extract frames from a video file, caption them, assemble the summary."""
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        duration, frames = select_frames(path, tmp)
        if not frames:
            raise RuntimeError("no frames extracted (unreadable clip?)")
        log(f"{name}: {len(frames)} frame(s), {duration:.1f}s, captioning...")
        captions = caption_frames(frames)
    return assemble_result(name, duration, captions)


def summarize_keyframes(name, image_paths, timestamps=None):
    """Caption already-extracted keyframes (no ffmpeg). timestamps optional (seconds)."""
    if not image_paths:
        raise RuntimeError("no keyframes provided")
    # Safety cap: if the caller sends more than MAX_FRAMES, subsample evenly.
    if len(image_paths) > MAX_FRAMES:
        keep = set(_evenly_pick(list(range(len(image_paths))), MAX_FRAMES))
        log(f"{name}: {len(image_paths)} keyframes -> capped to {MAX_FRAMES}")
        image_paths = [p for i, p in enumerate(image_paths) if i in keep]
        if timestamps:
            timestamps = [t for i, t in enumerate(timestamps) if i in keep]
    frames = []
    for i, p in enumerate(image_paths):
        t = timestamps[i] if timestamps and i < len(timestamps) else None
        frames.append((t, p))
    duration = max([t for t, _ in frames if t is not None], default=0.0)
    log(f"{name}: {len(frames)} keyframe(s), captioning...")
    captions = caption_frames(frames)
    return assemble_result(name, duration, captions)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def output_paths(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    dest_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.dirname(video_path)
    os.makedirs(dest_dir, exist_ok=True)
    return (os.path.join(dest_dir, base + ".summary.txt"),
            os.path.join(dest_dir, base + ".summary.json"))


def already_done(video_path):
    _, jpath = output_paths(video_path)
    return os.path.exists(jpath)


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
    os.replace(tmp, path)


def process(video_path):
    txt_path, json_path = output_paths(video_path)
    result, text_out = summarize_video(video_path)
    write_atomic(txt_path, text_out)
    write_atomic(json_path, json.dumps(result, indent=2, ensure_ascii=False))
    log(f"{os.path.basename(video_path)}: wrote {os.path.basename(txt_path)}")
    return result


# ----------------------------------------------------------------------------
# Watcher
# ----------------------------------------------------------------------------
def is_video(path):
    return path.lower().endswith(VIDEO_EXTS)


def wait_stable(path):
    """Block until the file stops growing (MapleCam may still be writing it)."""
    last, stable = -1, 0
    while stable < STABLE_SECONDS:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size > 0 and size == last:
            stable += 1
        else:
            stable, last = 0, size
        time.sleep(1)
    return True


def worker_loop(q):
    while True:
        path = q.get()
        try:
            if not os.path.exists(path) or not is_video(path):
                continue
            if not FORCE and already_done(path):
                continue
            if not wait_stable(path):
                log(f"{os.path.basename(path)}: vanished before it stabilized")
                continue
            if not FORCE and already_done(path):
                continue
            process(path)
        except Exception as e:  # noqa: BLE001 - never let the worker die
            log(f"ERROR processing {path}: {e}")
        finally:
            q.task_done()


class Handler(FileSystemEventHandler):
    def __init__(self, q):
        self.q = q

    def on_created(self, event):
        if not event.is_directory and is_video(event.src_path):
            self.q.put(event.src_path)

    def on_moved(self, event):
        if not event.is_directory and is_video(event.dest_path):
            self.q.put(event.dest_path)


def run_watch():
    os.makedirs(INPUT_DIR, exist_ok=True)
    q = queue.Queue()
    threading.Thread(target=worker_loop, args=(q,), daemon=True).start()

    if PROCESS_EXISTING:
        backlog = [p for p in sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*"), recursive=True))
                   if os.path.isfile(p) and is_video(p) and (FORCE or not already_done(p))]
        if backlog:
            log(f"queuing {len(backlog)} existing clip(s)")
            for p in backlog:
                q.put(p)

    obs = Observer()
    obs.schedule(Handler(q), INPUT_DIR, recursive=True)
    obs.start()
    log(f"watching {INPUT_DIR} (model={VLM_MODEL}, mode={FRAME_MODE}, max_frames={MAX_FRAMES})")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--once":
        result = process(sys.argv[2])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    run_watch()


if __name__ == "__main__":
    main()
