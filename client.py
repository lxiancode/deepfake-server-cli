"""
Rope Face-Swap Client
=====================
Captures webcam frames, sends them as JPEG to the server, and displays
the fully-composited swapped frame that comes back.

All face detection and compositing runs on the server GPU — the client
does nothing heavier than JPEG encode/decode.

Usage:
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source.jpg
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source_folder/

    Or set values in .env and run: python client.py

Keys:
    q    - quit
    s    - resend source to server
    r    - cycle enhance mode: off → GPEN-256 → GPEN-512 → off
    m    - toggle mouth mask
    +/-  - opacity up / down (0.05 steps)
"""

import argparse
import asyncio
import glob
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
load_dotenv()

import cv2
import msgpack
import numpy as np
import websockets

try:
    import pyvirtualcam
    _VCAM_AVAILABLE = True
except ImportError:
    _VCAM_AVAILABLE = False
    print("[vcam] pyvirtualcam not installed — virtual camera disabled")

_SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ------------------------------------------------------------------ #
# Source helpers                                                       #
# ------------------------------------------------------------------ #

def _collect_source_images(path: str) -> list[bytes]:
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read: {path}")
            return []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [buf.tobytes()] if ok else []

    if os.path.isdir(path):
        results = []
        for ext in _SOURCE_EXTS:
            for p in sorted(glob.glob(os.path.join(path, f"*{ext}")) +
                            glob.glob(os.path.join(path, f"*{ext.upper()}"))):
                img = cv2.imread(p)
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    results.append(buf.tobytes())
        return results

    print(f"[WARN] Source path not found: {path}")
    return []


# ------------------------------------------------------------------ #
# Shared display state                                                 #
# ------------------------------------------------------------------ #

# Cross-fade duration: how long to blend old→new swap result on arrival.
_FADE_SECS = 0.12

# Capture thread writes raw frames here at full camera FPS.
_raw_lock = threading.Lock()
_raw_frame: np.ndarray | None = None  # RGB

# Receiver writes server-processed frames here.
_swap_lock = threading.Lock()
_swap_frame: np.ndarray | None = None       # latest swap result (RGB)
_prev_swap_frame: np.ndarray | None = None  # swap result before the latest
_swap_arrive_time: float | None = None      # monotonic time latest result arrived


def _set_raw(frame_rgb: np.ndarray) -> None:
    global _raw_frame
    with _raw_lock:
        _raw_frame = frame_rgb


def _set_swap(frame_rgb: np.ndarray) -> None:
    global _swap_frame, _prev_swap_frame, _swap_arrive_time
    with _swap_lock:
        _prev_swap_frame = _swap_frame
        _swap_frame = frame_rgb
        _swap_arrive_time = time.monotonic()


def _get_display_frame() -> np.ndarray | None:
    """Return the frame to display, cross-fading between the last two swap
    results while a new one is settling in.  Falls back to raw if no swap
    result has arrived yet."""
    now = time.monotonic()
    with _swap_lock:
        cur   = _swap_frame
        prev  = _prev_swap_frame
        arr_t = _swap_arrive_time

    if cur is None:
        with _raw_lock:
            return _raw_frame

    if prev is None or arr_t is None:
        return cur

    elapsed = now - arr_t
    if elapsed >= _FADE_SECS:
        return cur

    # Linearly fade from prev → cur over _FADE_SECS.
    alpha = elapsed / _FADE_SECS          # 0 at arrival → 1 when fade done
    return cv2.addWeighted(cur, alpha, prev, 1.0 - alpha, 0)


# ------------------------------------------------------------------ #
# Stats (terminal debug)                                               #
# ------------------------------------------------------------------ #

_stats_lock = threading.Lock()
_stats: dict = {"captured": 0, "sent": 0, "recv": 0, "roundtrips": []}
_send_times: dict[int, float] = {}

# Live stats exposed to the display overlay (written by stats thread).
_live_stats: dict = {"swap_fps": 0.0, "rt_ms": 0.0}


def _stats_printer():
    while True:
        time.sleep(2.0)
        with _stats_lock:
            captured  = _stats["captured"]
            sent      = _stats["sent"]
            recv      = _stats["recv"]
            rts       = _stats["roundtrips"][:]
            _stats["captured"]   = 0
            _stats["sent"]       = 0
            _stats["recv"]       = 0
            _stats["roundtrips"] = []

        swap_fps = recv / 2.0
        rt_ms = sum(rts) / len(rts) if rts else 0.0
        _live_stats["swap_fps"] = swap_fps
        _live_stats["rt_ms"] = rt_ms
        rt_str = f"{rt_ms:.0f}ms" if rt_ms else "—"
        print(
            f"[client] capture={captured/2:.1f}fps  "
            f"sent={sent/2:.1f}fps  "
            f"swap={swap_fps:.1f}fps  "
            f"roundtrip={rt_str}"
        )


# ------------------------------------------------------------------ #
# Frame queue (raw BGR) for the sender                                 #
# ------------------------------------------------------------------ #
_frame_q: queue.Queue = queue.Queue(maxsize=2)


# ------------------------------------------------------------------ #
# Capture thread — raw BGR frames, no processing                      #
# ------------------------------------------------------------------ #

def _capture_thread(camera_id: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        return

    print(f"[capture] Camera {camera_id} opened at {width}x{height}")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Always update the raw display so the preview is smooth at camera FPS.
        _set_raw(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        with _stats_lock:
            _stats["captured"] += 1

        # Drop oldest frame if sender hasn't caught up.
        if _frame_q.full():
            try:
                _frame_q.get_nowait()
            except queue.Empty:
                pass
        _frame_q.put(frame_bgr)

    cap.release()


# ------------------------------------------------------------------ #
# Async WebSocket client                                               #
# ------------------------------------------------------------------ #

# One frame in-flight at a time: ensures we always send the freshest
# available frame and never build a stale backlog.
_in_flight: asyncio.Semaphore | None = None


async def _ws_client(uri: str, source_path: str, params_ref: dict):
    global _in_flight
    _in_flight = asyncio.Semaphore(1)

    print(f"[ws] Connecting to {uri} …")
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        print("[ws] Connected.")
        _ws_ref.append(ws)

        if source_path:
            await _send_source(ws, source_path)

        send_task = asyncio.create_task(_sender(ws, params_ref))
        recv_task = asyncio.create_task(_receiver(ws))
        await asyncio.gather(send_task, recv_task)


async def _send_source(ws, path: str):
    if not path:
        return
    images = _collect_source_images(path)
    if not images:
        print(f"[WARN] No readable images found at: {path}")
        return
    await ws.send(msgpack.packb({"type": "set_source", "images": images}, use_bin_type=True))
    label = f"{len(images)} image(s)" if len(images) > 1 else "1 image"
    print(f"[source] Sent {label} from {path}")


async def _sender(ws, params_ref: dict):
    loop = asyncio.get_event_loop()
    frame_id = 0
    while True:
        # Wait for the server to finish the previous frame before grabbing
        # the next one — this keeps _frame_q flushed to the freshest frame.
        await _in_flight.acquire()

        try:
            frame_bgr = await loop.run_in_executor(
                None, lambda: _frame_q.get(timeout=1.0))
        except queue.Empty:
            _in_flight.release()
            continue

        # Downscale + encode in a thread so the event loop stays free.
        def _encode(bgr):
            if _send_scale < 1.0:
                h, w = bgr.shape[:2]
                bgr = cv2.resize(bgr,
                                 (max(1, int(w * _send_scale)),
                                  max(1, int(h * _send_scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _send_quality])
            return (ok, buf)

        ok, buf = await loop.run_in_executor(None, lambda: _encode(frame_bgr))
        if not ok:
            _in_flight.release()
            continue

        _send_times[frame_id] = time.time()

        payload = msgpack.packb({
            "type": "frame",
            "frame_id": frame_id,
            "params": params_ref.copy(),
            "frame": buf.tobytes(),
        }, use_bin_type=True)
        await ws.send(payload)

        with _stats_lock:
            _stats["sent"] += 1

        frame_id += 1


async def _receiver(ws):
    async for raw in ws:
        msg = msgpack.unpackb(raw, raw=False)
        mtype = msg.get("type")

        if mtype == "frame_result":
            fid = msg.get("frame_id", -1)
            frame_bytes = msg.get("frame")

            if frame_bytes:
                # Decode + upscale in a thread so the event loop stays free.
                nparr = np.frombuffer(frame_bytes, np.uint8)

                def _decode(arr):
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return None
                    if _send_scale < 1.0 and _capture_size:
                        bgr = cv2.resize(bgr, _capture_size,
                                         interpolation=cv2.INTER_LINEAR)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                loop = asyncio.get_event_loop()
                result_rgb = await loop.run_in_executor(None, lambda: _decode(nparr))
                if result_rgb is not None:
                    _set_swap(result_rgb)

            sent_t = _send_times.pop(fid, None)
            rt_ms = (time.time() - sent_t) * 1000 if sent_t else 0.0

            with _stats_lock:
                _stats["recv"] += 1
                if sent_t:
                    _stats["roundtrips"].append(rt_ms)

            if _in_flight:
                _in_flight.release()

        elif mtype == "source_set":
            if msg.get("success"):
                n_ok  = msg.get("faces_used", "?")
                n_sent = msg.get("images_sent", "?")
                print(f"[source_set] OK — {n_ok}/{n_sent} image(s) had a detectable face")
            else:
                print(f"[source_set] FAILED: {msg.get('message', 'no face found')}")

        elif mtype == "source_ready":
            print("[server] Source face ready (loaded at server startup).")

        elif mtype == "error":
            print(f"[server error] {msg.get('message')}")
            if _in_flight:
                _in_flight.release()


# ------------------------------------------------------------------ #
# Display loop (main thread)                                           #
# ------------------------------------------------------------------ #

_ENHANCE_MODES = [None, "gpen256", "gpen512"]
_ENHANCE_LABELS = {None: "off", "gpen256": "GPEN-256", "gpen512": "GPEN-512"}

# waitKey delay sets the display update interval (~33ms ≈ 30fps).
# The cross-fade animation needs the loop to run continuously even when
# no new swap result has arrived, so we cannot gate imshow on object identity.
_DISPLAY_INTERVAL_MS = 33


def _display_loop(params_ref: dict, source_path_ref: list, width: int = 1280, height: int = 720):
    print("[display] q=quit  s=resend source  r=cycle enhance  m=mouth mask  +/-=opacity")

    vcam = None
    if _VCAM_AVAILABLE:
        try:
            vcam = pyvirtualcam.Camera(width=width, height=height, fps=30,
                                       fmt=pyvirtualcam.PixelFormat.RGB)
            print(f"[vcam] Virtual camera active: {vcam.device}")
        except Exception as e:
            print(f"[vcam] Could not open virtual camera: {e}")

    try:
        while True:
            frame = _get_display_frame()

            if frame is not None:
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                enhance_label = _ENHANCE_LABELS[params_ref.get("enhance")]
                mouth_on  = params_ref.get("mouth_mask", False)
                opacity   = params_ref.get("opacity", 1.0)
                swap_fps  = _live_stats["swap_fps"]
                rt_ms     = _live_stats["rt_ms"]
                rt_str    = f"{rt_ms:.0f}ms" if rt_ms else "—"

                cv2.putText(display, f"Swap {swap_fps:.1f}fps  rt {rt_str}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display,
                            f"Enhance: {enhance_label}  Mouth: {'on' if mouth_on else 'off'}  Opacity: {opacity:.2f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.imshow("Rope — Face Swap", display)

                if vcam is not None:
                    out = frame  # already RGB
                    if out.shape[1] != width or out.shape[0] != height:
                        out = cv2.resize(out, (width, height))
                    vcam.send(np.ascontiguousarray(out))
            elif vcam is not None:
                vcam.send(np.zeros((height, width, 3), dtype=np.uint8))

            key = cv2.waitKey(_DISPLAY_INTERVAL_MS) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if source_path_ref[0] and _ws_loop and _ws_ref:
                    asyncio.run_coroutine_threadsafe(
                        _send_source(_ws_ref[0], source_path_ref[0]),
                        _ws_loop)
            elif key == ord('r'):
                cur = params_ref.get("enhance")
                nxt = _ENHANCE_MODES[(_ENHANCE_MODES.index(cur) + 1) % len(_ENHANCE_MODES)]
                params_ref["enhance"] = nxt
                print(f"[params] enhance: {_ENHANCE_LABELS[nxt]}")
            elif key == ord('m'):
                params_ref["mouth_mask"] = not params_ref.get("mouth_mask", False)
                print(f"[params] mouth_mask: {'on' if params_ref['mouth_mask'] else 'off'}")
            elif key == ord('+'):
                params_ref["opacity"] = round(min(params_ref.get("opacity", 1.0) + 0.05, 1.0), 2)
                print(f"[params] opacity: {params_ref['opacity']:.2f}")
            elif key == ord('-'):
                params_ref["opacity"] = round(max(params_ref.get("opacity", 1.0) - 0.05, 0.0), 2)
                print(f"[params] opacity: {params_ref['opacity']:.2f}")
    finally:
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ #
# Source-face picker                                                   #
# ------------------------------------------------------------------ #

def _pick_source_face() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select source face image",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        print("No source face selected — exiting.")
        sys.exit(0)
    return path


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_ref: list = []

# Set by main() before threads start — used by sender/receiver.
_send_scale: float = 0.5
_send_quality: int = 60
_capture_size: tuple[int, int] | None = None  # (width, height) for upscaling results


def main():
    global _ws_loop, _ws_ref, _send_scale, _send_quality, _capture_size

    parser = argparse.ArgumentParser(
        description="Rope face-swap client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", default=os.getenv("SERVER_URL", "ws://127.0.0.1:8765/ws"),
                        help="WebSocket server URL (env: SERVER_URL)")
    parser.add_argument("--source", default=os.getenv("SOURCE_PATH", ""),
                        help="Source face image or folder (env: SOURCE_PATH)")
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")),
                        help="Camera device index (env: CAMERA_INDEX)")
    parser.add_argument("--width", type=int, default=int(os.getenv("CAPTURE_WIDTH", "1280")),
                        help="Capture width (env: CAPTURE_WIDTH)")
    parser.add_argument("--height", type=int, default=int(os.getenv("CAPTURE_HEIGHT", "720")),
                        help="Capture height (env: CAPTURE_HEIGHT)")
    parser.add_argument("--send-scale", type=float, default=float(os.getenv("SEND_SCALE", "0.5")),
                        help="Downscale factor for frames sent to server (env: SEND_SCALE)")
    parser.add_argument("--send-quality", type=int, default=int(os.getenv("SEND_QUALITY", "60")),
                        help="JPEG quality for frames sent to server (env: SEND_QUALITY)")
    args = parser.parse_args()

    _send_scale   = max(0.1, min(1.0, args.send_scale))
    _send_quality = max(1,   min(100, args.send_quality))
    _capture_size = (args.width, args.height)

    print(f"[config] send_scale={_send_scale}  send_quality={_send_quality}  "
          f"capture={args.width}x{args.height}")

    if not args.source:
        args.source = _pick_source_face()

    params_ref: dict = {"enhance": None, "mouth_mask": False, "opacity": 1.0}
    source_path_ref: list = [args.source]

    cap_thread = threading.Thread(target=_capture_thread,
                                  args=(args.camera, args.width, args.height),
                                  daemon=True)
    cap_thread.start()

    stats_thread = threading.Thread(target=_stats_printer, daemon=True)
    stats_thread.start()

    def _run_ws():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        try:
            _ws_loop.run_until_complete(_ws_client(args.server, args.source, params_ref))
        except Exception as e:
            print(f"[ws] Disconnected: {e}")

    ws_thread = threading.Thread(target=_run_ws, daemon=True)
    ws_thread.start()

    _display_loop(params_ref, source_path_ref, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
