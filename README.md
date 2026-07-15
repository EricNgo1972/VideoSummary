# mk-video-summarizer

CPU-only container that turns short MapleCam event clips into text summaries.

Two ways to feed it, set by the `MODE` env var:

- **`serve`** (default) — an HTTP API. MapleCam (in another container / on
  another box) POSTs a clip **or a few keyframes** and gets the summary back.
- **`watch`** — watches a mounted folder and writes a summary next to each clip.
- **`both`** — runs the watcher and the HTTP API together.

**Flow:** frames (uploaded, or sampled from the video with `ffmpeg`) → each frame
is captioned by a small vision model (`moondream`) served by Ollama → captions
are aggregated into a summary.

No GPU. No audio. One summarization at a time (so it can't thrash the CPU).

## HTTP API (MODE=serve)

`POST /summarize` (`multipart/form-data`):

| Field | Repeatable | Meaning |
|-------|-----------|---------|
| `frames` / `images` | yes | **Preferred.** Pre-extracted keyframe JPGs. No ffmpeg runs server-side. |
| `video` | no | A whole clip; the server runs ffmpeg to sample frames. |
| `name` | no | Label for the clip in the output. |
| `timestamps` | no | Comma-separated seconds, one per keyframe, for the timeline. |

`GET /health` → `{"status":"ok", ...}`.

If `API_KEY` is set, every request (except `/health`) must send header
`X-API-Key: <key>`.

### Upload keyframes — preferred (light on bandwidth + server CPU)

Extract keyframes on the **MapleCam side**, send only the JPGs:

```bash
# on the box that has the clip
ffmpeg -i clip.mp4 -vf "select='gt(scene,0.3)',scale=640:-1" -vsync vfr kf_%03d.jpg

curl -sf -X POST http://SUMMARIZER_HOST:8080/summarize \
  -F "name=front_door_1830" \
  -F "timestamps=1.2,4.7,9.3" \
  -F "frames=@kf_001.jpg" -F "frames=@kf_002.jpg" -F "frames=@kf_003.jpg"
```

### Upload the whole video (server does the frame extraction)

```bash
curl -sf -X POST http://SUMMARIZER_HOST:8080/summarize \
  -F "name=front_door_1830" \
  -F "video=@clip.mp4"
```

Both return the same JSON (see **Output shape** below), plus a `text` field.

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
# HTTP API (default): MapleCam POSTs clips/keyframes from anywhere
docker run -d --name mk-video-summarizer --restart unless-stopped \
  --cpus="4" --memory="6g" --memory-swap="6g" \
  -p 8080:8080 \
  mk-video-summarizer:latest
```

For the folder workflow instead, add `-e MODE=watch` and mount `/data/in`
(+ `/data/out`); each `clip.mp4` that lands produces `clip.summary.txt` and
`clip.summary.json`. Use `-e MODE=both` to run the API and the watcher together.

## Test one clip (no watching)

```bash
docker exec mk-video-summarizer python3 /app/worker.py --once /data/in/some_clip.mp4
```

## Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODE` | `serve` | `serve` (HTTP API), `watch` (folder), or `both`. |
| `PORT` | `8080` | HTTP API port (serve/both). |
| `API_KEY` | *(unset)* | If set, requests must send header `X-API-Key`. |
| `MAX_UPLOAD_MB` | `200` | Max request body size for uploads. |
| `INPUT_DIR` | `/data/in` | Folder to watch (recursive), MODE=watch/both. |
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
