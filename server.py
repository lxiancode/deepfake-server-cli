"""
Rope Face-Swap API Server (Deep-Live-Cam backend)
==================================================
Receives full JPEG frames from a client over WebSocket, runs insightface
face detection + inswapper on GPU, and streams swapped frames back.

Usage:
    python server.py --source /path/to/source_face.jpg [--host 0.0.0.0] [--port 8765]

WebSocket endpoint: ws://<host>:<port>/ws

Protocol (msgpack-serialised dicts):

  Client -> Server:
    { "type": "set_source", "images": [<bytes>, ...] }
    { "type": "frame", "frame_id": int, "frame": <bytes>, "params": { ... } }

    Supported params keys:
      enhance    : null | "gpen256" | "gpen512"   — face restoration model
      opacity    : float 0–1                       — blend strength (default 1.0)
      mouth_mask : bool                            — preserve original mouth (default false)
      sharpness  : float 0+                        — post-swap sharpening (default 0.0)

  Server -> Client:
    { "type": "source_set",   "success": bool, "faces_used": int, "images_sent": int }
    { "type": "source_ready" }
    { "type": "frame_result", "frame_id": int, "frame": <bytes> }
    { "type": "error",        "message": str }
"""

import argparse
import asyncio
import concurrent.futures
import glob
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# Preload libcudnn.so.9 so ORT's CUDA provider can dlopen it.
# nvidia-cudnn-cu12 installs it under site-packages/nvidia/cudnn/lib/ which the
# dynamic linker won't search without an explicit preload.
import ctypes as _ctypes, site as _site
for _sp in _site.getsitepackages() + [_site.getusersitepackages()]:
    _cudnn = os.path.join(_sp, "nvidia", "cudnn", "lib", "libcudnn.so.9")
    if os.path.exists(_cudnn):
        try:
            _ctypes.CDLL(_cudnn, _ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        break
del _ctypes, _site

import cv2
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ------------------------------------------------------------------ #
# Bootstrap DLC modules                                                #
# ------------------------------------------------------------------ #
sys.path.insert(0, os.path.dirname(__file__))

import modules.globals
modules.globals.headless = True
modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
modules.globals.many_faces = True
modules.globals.mouth_mask = False
modules.globals.opacity = 1.0
modules.globals.sharpness = 0.0
modules.globals.enable_interpolation = False
modules.globals.poisson_blend = False

from modules.face_analyser import get_one_face, get_many_faces
from modules.processors.frame.face_swapper import swap_face, get_face_swapper
from modules.processors.frame import face_enhancer_gpen256, face_enhancer_gpen512

# ------------------------------------------------------------------ #
# Globals                                                              #
# ------------------------------------------------------------------ #
_source_face = None      # insightface Face object
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_verbose = False         # set by --verbose flag
_prev_result_bgr: np.ndarray | None = None  # for temporal blending

# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #
app = FastAPI()


@app.on_event("startup")
async def _startup():
    global _source_face
    print("Loading models …")
    get_face_swapper()
    print("Models ready.")

    if _args.source:
        raw_images = _load_source_images(_args.source)
        if not raw_images:
            print(f"[WARN] No readable images found at: {_args.source}")
        else:
            loop = asyncio.get_running_loop()
            face = await loop.run_in_executor(_executor, _detect_source_face, raw_images)
            if face is None:
                print(f"[WARN] No face detected in source image(s): {_args.source}")
            else:
                _source_face = face
                print(f"Source face loaded from {_args.source}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": get_face_swapper() is not None,
        "source_face_loaded": _source_face is not None,
    }


@app.get("/")
async def index():
    html = """<!DOCTYPE html>
<html>
<head><title>Rope Server</title>
<style>body{font-family:monospace;padding:2em;background:#111;color:#eee}
  #log{margin-top:1em;white-space:pre;color:#0f0}</style>
</head>
<body>
<h2>Rope Face-Swap Server</h2>
<div id="status">Checking health…</div>
<button onclick="pingWS()">Ping WebSocket</button>
<div id="log"></div>
<script>
const log = s => document.getElementById('log').textContent += s + '\\n';
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent = JSON.stringify(d, null, 2);
});
function pingWS() {
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => log('WS connected');
  ws.onmessage = e => log('WS message received (' + e.data.byteLength + ' bytes)');
  ws.onerror = e => log('WS error: ' + e);
  ws.onclose = () => log('WS closed');
}
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    print(f"[WS] client connected: {client.host}:{client.port}")

    source_face = _source_face
    if source_face is not None:
        await _send(websocket, {"type": "source_ready"})

    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    loop = asyncio.get_running_loop()

    async def _recv_loop():
        nonlocal source_face
        try:
            while True:
                raw = await websocket.receive_bytes()
                msg = msgpack.unpackb(raw, raw=False)
                mtype = msg.get("type")

                if mtype == "set_source":
                    raw_list = msg.get("images") or (
                        [msg["image"]] if "image" in msg else [])
                    print(f"[set_source] received {len(raw_list)} image(s) …")
                    t0 = time.perf_counter()
                    face = await loop.run_in_executor(
                        _executor, _detect_source_face, raw_list)
                    elapsed = (time.perf_counter() - t0) * 1000
                    if face is None:
                        print(f"[set_source] FAILED — no face detected ({elapsed:.0f} ms)")
                        await _send(websocket, {
                            "type": "source_set", "success": False,
                            "message": "No face detected in any source image.",
                        })
                    else:
                        source_face = face
                        print(f"[set_source] OK ({elapsed:.0f} ms)")
                        await _send(websocket, {
                            "type": "source_set", "success": True,
                            "faces_used": 1, "images_sent": len(raw_list),
                        })

                elif mtype == "frame":
                    if source_face is None:
                        await _send(websocket, {
                            "type": "error",
                            "message": "No source face set. Send set_source first.",
                        })
                        continue

                    item = (
                        msg.get("frame_id", 0),
                        msg.get("frame"),
                        source_face,
                        msg.get("params") or {},
                    )
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        frame_queue.put_nowait(item)
                    except asyncio.QueueFull:
                        pass

                else:
                    await _send(websocket, {
                        "type": "error",
                        "message": f"Unknown message type: {mtype}",
                    })

        except WebSocketDisconnect:
            print(f"[WS] client disconnected: {client.host}:{client.port}")
        except Exception as exc:
            print(f"[RECV] unexpected error: {exc}")
        finally:
            try:
                frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                frame_queue.get_nowait()
                frame_queue.put_nowait(None)

    async def _proc_loop():
        while True:
            item = await frame_queue.get()
            if item is None:
                break
            frame_id, frame_bytes, s_face, params = item
            t0 = time.perf_counter()
            try:
                result_bytes, metrics = await loop.run_in_executor(
                    _executor, _process_frame_sync, frame_bytes, s_face, params)
                elapsed = (time.perf_counter() - t0) * 1000
                if result_bytes is not None:
                    if _verbose:
                        print(f"[proc] frame {frame_id} done in {elapsed:.1f} ms "
                              f"({len(result_bytes)//1024}KB)"
                              + (f"  sharp={metrics.get('sharpness', 0):.0f}"
                                 f"  det={metrics.get('det_score', 0):.2f}"
                                 f"  flicker={metrics.get('temporal_diff', 0):.1f}"
                                 if metrics else ""))
                    await _send(websocket, {
                        "type": "frame_result",
                        "frame_id": frame_id,
                        "frame": result_bytes,
                        "metrics": metrics,
                    })
                else:
                    await _send(websocket, {
                        "type": "error", "message": "Failed to decode frame.",
                    })
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"[proc] frame {frame_id} ERROR after {elapsed:.1f} ms: {exc}")
                try:
                    await _send(websocket, {"type": "error", "message": str(exc)})
                except Exception:
                    pass

    recv_task = asyncio.create_task(_recv_loop())
    proc_task = asyncio.create_task(_proc_loop())
    done, pending = await asyncio.wait(
        [recv_task, proc_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


# ------------------------------------------------------------------ #
# Synchronous GPU processing                                           #
# ------------------------------------------------------------------ #

def _compute_quality_metrics(
    result_bgr: np.ndarray,
    pre_blend_bgr: np.ndarray,
    prev_bgr: np.ndarray | None,
    target_faces,
) -> dict:
    """Compute lightweight per-frame quality metrics (all NumPy/OpenCV, no extra deps)."""
    metrics: dict = {}

    # Sharpness: Laplacian variance of the largest face crop in the final output.
    if target_faces:
        best = max(target_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        metrics["det_score"] = float(best.det_score)
        x1, y1, x2, y2 = [int(v) for v in best.bbox]
        h, w = result_bgr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = result_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            metrics["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Temporal flicker: mean absolute difference between consecutive pre-blend
    # frames.  Measured before blending so we capture raw GAN variance.
    if prev_bgr is not None and prev_bgr.shape == pre_blend_bgr.shape:
        diff = np.abs(pre_blend_bgr.astype(np.float32) - prev_bgr.astype(np.float32))
        metrics["temporal_diff"] = float(diff.mean())

    return metrics


def _process_frame_sync(
    frame_bytes: bytes, source_face, params: dict
) -> tuple[bytes | None, dict]:
    """Decode JPEG → detect → swap → optional enhance → re-encode JPEG.

    Returns (jpeg_bytes_or_None, quality_metrics_dict).
    """
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None, {}

    # Apply per-frame params to globals (safe: single-threaded executor)
    modules.globals.opacity = float(params.get("opacity", 1.0))
    modules.globals.mouth_mask = bool(params.get("mouth_mask", False))
    modules.globals.sharpness = float(params.get("sharpness", 0.0))
    modules.globals.poisson_blend = bool(params.get("poisson_blend", False))
    modules.globals.interpolation_weight = float(params.get("interpolation_weight", 0.0))
    enhance = params.get("enhance")  # None | "gpen256" | "gpen512"

    t_det = time.perf_counter()
    target_faces = get_many_faces(frame_bgr)
    if _verbose:
        n = len(target_faces) if target_faces else 0
        h, w = frame_bgr.shape[:2]
        print(f"[proc] {w}x{h}  det={n} faces  {(time.perf_counter()-t_det)*1000:.1f}ms")

    if not target_faces:
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return (buf.tobytes() if ok else None), {}

    t_swap = time.perf_counter()
    result = frame_bgr
    for target_face in target_faces:
        result = swap_face(source_face, target_face, result)
    if _verbose:
        print(f"[proc] swap {(time.perf_counter()-t_swap)*1000:.1f}ms")

    if enhance:
        t_enh = time.perf_counter()
        for face in target_faces:
            if enhance == "gpen256":
                result = face_enhancer_gpen256.enhance_face(result, face)
            elif enhance == "gpen512":
                result = face_enhancer_gpen512.enhance_face(result, face)
        if _verbose:
            print(f"[proc] enhance({enhance}) {(time.perf_counter()-t_enh)*1000:.1f}ms")

    # Capture pre-blend result for flicker measurement before temporal smoothing.
    pre_blend = result.copy()

    # Temporal blend with previous frame to suppress GAN flicker.
    # 25% weight on the previous output keeps ghosting negligible while
    # significantly damping per-pixel noise between consecutive frames.
    global _prev_result_bgr
    if _prev_result_bgr is not None and _prev_result_bgr.shape == result.shape:
        result = cv2.addWeighted(result, 0.75, _prev_result_bgr, 0.25, 0)

    metrics = _compute_quality_metrics(result, pre_blend, _prev_result_bgr, target_faces)
    if metrics and not getattr(_process_frame_sync, "_metrics_logged", False):
        print(f"[metrics] first frame: {metrics}")
        _process_frame_sync._metrics_logged = True
    _prev_result_bgr = pre_blend  # track pre-blend so temporal_diff reflects raw GAN variance

    ok, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return (buf.tobytes() if ok else None), metrics


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_source_images(path: str) -> list[bytes]:
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            return []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [buf.tobytes()] if ok else []

    if os.path.isdir(path):
        results = []
        for ext in _IMAGE_EXTS:
            for p in glob.glob(os.path.join(path, f"*{ext}")) + \
                     glob.glob(os.path.join(path, f"*{ext.upper()}")):
                img = cv2.imread(p)
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    results.append(buf.tobytes())
        return results

    return []


def _detect_source_face(raw_images: list):
    """Return the highest-confidence insightface Face from any source image."""
    best_face = None
    best_score = -1.0
    for img_bytes in raw_images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        face = get_one_face(img)
        if face is not None and face.det_score > best_score:
            best_score = face.det_score
            best_face = face
    return best_face


async def _send(ws: WebSocket, payload: dict):
    await ws.send_bytes(msgpack.packb(payload, use_bin_type=True))


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
_args: argparse.Namespace | None = None


def main():
    global _args, _verbose
    parser = argparse.ArgumentParser(
        description="Rope face-swap WebSocket server (Deep-Live-Cam backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                        help="Bind address (env: HOST)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")),
                        help="Port to listen on (env: PORT)")
    parser.add_argument("--source", default=os.getenv("SOURCE_PATH", ""),
                        help="Source face image or folder (env: SOURCE_PATH)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame timing logs")
    _args = parser.parse_args()
    _verbose = _args.verbose

    uvicorn.run(app, host=_args.host, port=_args.port, log_level="warning")


if __name__ == "__main__":
    main()
