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


# Landing page: a working upload UI + API docs, so an operator who just installed
# the container knows it's alive, can try it in the browser, and can wire other
# apps to the API. Fully self-contained (no external assets).
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Summarizer</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.55 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0f1115; color: #e6e8ee; }
  @media (prefers-color-scheme: light) { body { background: #f6f7f9; color: #1c1f26; } }
  .wrap { max-width: 860px; margin: 0 auto; padding: 28px 20px 64px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 28px 0 10px; }
  .sub { opacity: .7; margin: 0 0 18px; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 12px; padding: 18px; margin: 14px 0; }
  @media (prefers-color-scheme: light) { .card { background: #fff; border-color: #e3e6ec; } }
  label { display: block; font-weight: 600; margin: 12px 0 5px; font-size: 13px; }
  input[type=text], input[type=file] { width: 100%; padding: 9px 11px; border-radius: 8px;
         border: 1px solid #333a48; background: #0f1218; color: inherit; font: inherit; }
  @media (prefers-color-scheme: light) { input { background: #fafbfc; border-color: #d3d8e0; } }
  .hint { font-size: 12px; opacity: .6; margin-top: 4px; }
  button { margin-top: 16px; padding: 11px 20px; border: 0; border-radius: 8px; cursor: pointer;
           background: #3b82f6; color: #fff; font: inherit; font-weight: 600; }
  button:disabled { opacity: .55; cursor: default; }
  .pill { display: inline-block; font-size: 12px; padding: 3px 10px; border-radius: 20px;
          background: #22272f; opacity: .85; }
  .pill.ok { color: #34d399; } .pill.bad { color: #f87171; }
  #status { margin-top: 14px; font-size: 14px; }
  .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid #3b82f6;
          border-top-color: transparent; border-radius: 50%; animation: r .8s linear infinite; vertical-align: -2px; }
  @keyframes r { to { transform: rotate(360deg); } }
  .summary { font-size: 16px; line-height: 1.6; margin: 6px 0 0; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #262b36; vertical-align: top; }
  td.ts { white-space: nowrap; opacity: .7; font-variant-numeric: tabular-nums; width: 60px; }
  pre { background: #0b0d12; border: 1px solid #262b36; border-radius: 8px; padding: 12px;
        overflow-x: auto; font-size: 13px; margin: 8px 0; }
  @media (prefers-color-scheme: light) { pre { background: #f0f2f5; border-color: #dde1e8; } }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .muted { opacity: .6; }
  .hidden { display: none; }
  .copy { float: right; margin: 0; padding: 3px 10px; font-size: 12px; background: #2a3240; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Video Summarizer <span id="health" class="pill">checking…</span></h1>
  <p class="sub">Upload a short clip (or a few keyframe images) and get a text summary. Runs on CPU — no GPU.</p>

  <div class="card">
    <form id="f">
      <label for="file">Video or keyframe images</label>
      <input id="file" type="file" accept="video/*,image/*" multiple>
      <div class="hint">One video (mp4/mkv/mov…) <b>or</b> several JPG/PNG keyframes. Keyframes are faster — no server-side frame extraction.</div>

      <label for="name">Label <span class="muted">(optional)</span></label>
      <input id="name" type="text" placeholder="front_door_1830">

      <label for="apikey">API key <span class="muted">(only if this server requires one)</span></label>
      <input id="apikey" type="text" placeholder="leave blank if auth is off" autocomplete="off">

      <button id="go" type="submit">Summarize</button>
    </form>
    <div id="status"></div>
  </div>

  <div id="result" class="card hidden">
    <h2>Summary</h2>
    <p id="summary" class="summary"></p>
    <h2>Timeline</h2>
    <table id="timeline"><tbody></tbody></table>
    <h2>Raw JSON</h2>
    <pre><code id="raw"></code></pre>
  </div>

  <h2>Using the API from another app</h2>
  <div class="card">
    <p>MapleCam (or any service) sends a clip or keyframes to <code>POST /summarize</code> and gets JSON back. <code>multipart/form-data</code> fields:</p>
    <table>
      <tr><th>Field</th><th>Meaning</th></tr>
      <tr><td><code>frames</code></td><td>Repeatable. Pre-extracted keyframe images — <b>preferred</b> (light on bandwidth, no server-side ffmpeg).</td></tr>
      <tr><td><code>video</code></td><td>A whole clip; the server samples frames with ffmpeg.</td></tr>
      <tr><td><code>name</code></td><td>Optional label for the clip.</td></tr>
      <tr><td><code>timestamps</code></td><td>Optional, comma-separated seconds — one per keyframe, for the timeline.</td></tr>
    </table>

    <h2>Upload keyframes (preferred)</h2>
    <pre><button class="copy">copy</button><code>ffmpeg -i clip.mp4 -vf "select='gt(scene,0.3)',scale=640:-1" -vsync vfr kf_%03d.jpg

curl -sf -X POST __BASE__/summarize \
  -F "name=front_door_1830" \
  -F "timestamps=1.2,4.7,9.3" \
  -F "frames=@kf_001.jpg" -F "frames=@kf_002.jpg" -F "frames=@kf_003.jpg"</code></pre>

    <h2>Upload a whole video</h2>
    <pre><button class="copy">copy</button><code>curl -sf -X POST __BASE__/summarize \
  -F "name=front_door_1830" \
  -F "video=@clip.mp4"</code></pre>

    <h2>Response shape</h2>
    <pre><code>{
  "video": "front_door_1830",
  "duration_seconds": 172.4,
  "frames_analyzed": 6,
  "vlm_model": "moondream",
  "summary": "A person approaches the door and leaves a package.",
  "timeline": [ { "t": 12.0, "ts": "00:12", "caption": "A person walks toward the door." } ],
  "text": "…human-readable version of the above…"
}</code></pre>
    <p class="hint">If this server has an API key set, add header <code>-H "X-API-Key: YOUR_KEY"</code> to every request (not needed for <code>GET /health</code>).</p>
  </div>
</div>

<script>
  var base = location.origin;
  // Fill API examples with this server's own URL.
  document.querySelectorAll('pre code').forEach(function(el){
    if (el.textContent.indexOf('__BASE__') >= 0)
      el.textContent = el.textContent.split('__BASE__').join(base);
  });
  document.querySelectorAll('.copy').forEach(function(b){
    b.addEventListener('click', function(){
      var code = b.parentNode.querySelector('code');
      navigator.clipboard.writeText(code.textContent).then(function(){
        b.textContent = 'copied'; setTimeout(function(){ b.textContent = 'copy'; }, 1200);
      });
    });
  });

  // Health pill.
  fetch('/health').then(function(r){return r.json();}).then(function(d){
    var el = document.getElementById('health');
    el.textContent = '● online · ' + d.vlm_model; el.className = 'pill ok';
  }).catch(function(){
    var el = document.getElementById('health'); el.textContent = '● offline'; el.className = 'pill bad';
  });

  var t0 = 0, timer = null;
  function setStatus(html){ document.getElementById('status').innerHTML = html; }

  document.getElementById('f').addEventListener('submit', function(e){
    e.preventDefault();
    var files = document.getElementById('file').files;
    if (!files.length) { setStatus('<span class="pill bad">Choose a video or some keyframe images first.</span>'); return; }

    var fd = new FormData();
    var name = document.getElementById('name').value.trim();
    if (name) fd.append('name', name);
    var isVideo = (files[0].type || '').indexOf('video') === 0;
    if (isVideo) { fd.append('video', files[0]); }
    else { for (var i = 0; i < files.length; i++) fd.append('frames', files[i]); }

    var headers = {};
    var key = document.getElementById('apikey').value.trim();
    if (key) headers['X-API-Key'] = key;

    var go = document.getElementById('go');
    go.disabled = true;
    document.getElementById('result').classList.add('hidden');
    t0 = Date.now();
    if (timer) clearInterval(timer);
    timer = setInterval(function(){
      var s = ((Date.now()-t0)/1000).toFixed(0);
      setStatus('<span class="spin"></span> Processing… ' + s + 's <span class="muted">(CPU: usually 20–50s)</span>');
    }, 500);

    fetch('/summarize', { method:'POST', body: fd, headers: headers })
      .then(function(res){ return res.json().then(function(d){ return {ok:res.ok, status:res.status, d:d}; }); })
      .then(function(r){
        clearInterval(timer); go.disabled = false;
        if (!r.ok) { setStatus('<span class="pill bad">Error: ' + (r.d.error || ('HTTP '+r.status)) + '</span>'); return; }
        var secs = ((Date.now()-t0)/1000).toFixed(1);
        setStatus('<span class="pill ok">Done in ' + secs + 's · ' + (r.d.frames_analyzed||0) + ' frames</span>');
        render(r.d);
      })
      .catch(function(err){ clearInterval(timer); go.disabled = false; setStatus('<span class="pill bad">Error: ' + err.message + '</span>'); });
  });

  function render(d){
    document.getElementById('summary').textContent = d.summary || '(no summary)';
    var tb = document.querySelector('#timeline tbody'); tb.innerHTML = '';
    (d.timeline || []).forEach(function(row){
      var tr = document.createElement('tr');
      tr.innerHTML = '<td class="ts">' + (row.ts||'--:--') + '</td><td>' + escapeHtml(row.caption||'') + '</td>';
      tb.appendChild(tr);
    });
    document.getElementById('raw').textContent = JSON.stringify(d, null, 2);
    document.getElementById('result').classList.remove('hidden');
  }
  function escapeHtml(s){ return String(s).replace(/[&<>"]/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
</script>
</body>
</html>
"""


@app.before_request
def _auth():
    # The landing page and health check are always viewable; the API needs the key (if configured).
    if API_KEY and request.endpoint not in ("health", "index"):
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify(error="unauthorized"), 401
    return None


@app.get("/")
def index():
    return INDEX_HTML


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
