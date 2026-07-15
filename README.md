# mk-video-summarizer

CPU-only container that turns short MapleCam event clips into text summaries.

**Flow:** watch a folder → `ffmpeg` samples a few frames per clip → each frame is
captioned by a small vision model (`moondream`) served by Ollama → captions are
aggregated into a summary → a `.summary.txt` + `.summary.json` is written.

No GPU. No audio. One clip at a time (so it can't thrash the CPU).

## Build

```bash
docker build -t mk-video-summarizer:latest .
```

The vision model (~1.6 GB) is **baked into the image** at build time, so a
provisioned host needs no model download on first start. Image is ~3–3.5 GB.

## Run

```bash
docker compose up -d --build
```

or plain docker:

```bash
docker run -d --name mk-video-summarizer --restart unless-stopped \
  --cpus="4" --memory="6g" --memory-swap="6g" \
  -v /path/to/maplecam/events:/data/in \
  -v /path/to/summaries:/data/out \
  mk-video-summarizer:latest
```

Point MapleCam's event-clip output at the `/data/in` mount. For each `clip.mp4`
that lands, the container writes `clip.summary.txt` and `clip.summary.json`
(to `/data/out`, or beside the clip if `OUTPUT_DIR` is blank).

## Test one clip (no watching)

```bash
docker exec mk-video-summarizer python3 /app/worker.py --once /data/in/some_clip.mp4
```

## Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `INPUT_DIR` | `/data/in` | Folder to watch (recursive). |
| `OUTPUT_DIR` | `/data/out` | Where summaries go; blank = beside each clip. |
| `VLM_MODEL` | `moondream` | Ollama vision model for captioning. |
| `SUMMARY_MODEL` | *(unset)* | Optional tiny text LLM to refine the summary (e.g. `qwen2.5:1.5b`). Pulled on first start. |
| `FRAME_MODE` | `scene` | `scene` = scene-change sampling (uniform fallback); `uniform` = evenly spaced. |
| `MAX_FRAMES` | `6` | Max frames captioned per clip (caps CPU time). |
| `MIN_FRAMES` | `3` | Below this from scene detection, fall back to uniform. |
| `SCENE_THRESHOLD` | `0.3` | Scene-change sensitivity (lower = more frames). |
| `FRAME_WIDTH` | `640` | Frames downscaled to this width before captioning. |
| `NUM_PREDICT` | `80` | Max caption tokens. |
| `CAPTION_PROMPT` | *(security-camera prompt)* | Override to change what the VLM reports. |
| `STABLE_SECONDS` | `3` | Wait for a clip to stop growing before processing. |
| `PROCESS_EXISTING` | `true` | Also process clips already present at startup. |
| `FORCE` | `false` | Reprocess even if a summary already exists. |
| `VIDEO_EXTS` | `.mp4,.mkv,.avi,.mov,.m4v,.ts` | Extensions treated as clips. |

## Cost / timing note

Per 3-min clip on CPU: scene detect (~1–2s) + `MAX_FRAMES` captions (~3–8s each)
+ aggregation ≈ **20–50s**. Clips are queued and processed one at a time; if many
arrive at once they back up rather than running in parallel (parallel VLM
inference on CPU just thrashes). Raise `--cpus` for faster single-clip latency,
lower `MAX_FRAMES` for cheaper summaries.

## Output shape (`*.summary.json`)

```json
{
  "video": "clip.mp4",
  "duration_seconds": 172.4,
  "frames_analyzed": 6,
  "vlm_model": "moondream",
  "summary": "A person in dark clothing approaches the front door and leaves a package.",
  "timeline": [
    {"t": 12.0, "ts": "00:12", "caption": "A person walks toward the door."},
    {"t": 47.5, "ts": "00:47", "caption": "The person sets a box down."}
  ]
}
```
