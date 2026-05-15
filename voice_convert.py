"""
voice_convert.py — Real-time RVC voice conversion
Runs alongside server.py as an independent audio-only pipeline.

Setup
-----
1.  Install audio deps:
        uv pip install sounddevice soundfile scipy rvc-python

2.  rvc-python will auto-download hubert_base.pt and rmvpe.pt on first run.
    You only need to provide a trained RVC voice model:
        models/<voice>.pth      — trained RVC v2 model
        models/<voice>.index    — (optional) feature index for timbre retrieval

    Train or download .pth models from: https://huggingface.co/lj1995/VoiceConversionWebUI

3.  List your audio devices:
        python voice_convert.py --list-devices

4.  Run:
        python voice_convert.py --model models/<voice>.pth \\
                                 --input-device 1 --output-device 3

    Point your virtual mic (BlackHole / VB-Cable) at output-device.
    Point Zoom / Discord at that same virtual mic.

Runtime hotkeys (press Enter after each):
    q    quit
    +    pitch shift up 1 semitone
    -    pitch shift down 1 semitone
    p    toggle pass-through (bypass RVC, raw mic audio)
"""

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[ERROR] Missing deps. Run:  uv pip install sounddevice soundfile scipy rvc-python")
    sys.exit(1)

try:
    from rvc_python.infer import RVCInference
    _RVC_OK = True
except ImportError:
    _RVC_OK = False
    print("[rvc] rvc-python not installed — running in pass-through mode")
    print("      Install with:  uv pip install rvc-python")


# ── SOLA blender ──────────────────────────────────────────────────────────────

class SOLABlender:
    """Synchronized Overlap-Add crossfader — smooths boundaries between chunks."""

    def __init__(self, overlap_samples: int):
        self.overlap = overlap_samples
        self._tail = np.zeros(overlap_samples, dtype=np.float32)

    def blend(self, chunk: np.ndarray) -> np.ndarray:
        if self.overlap == 0 or len(chunk) <= self.overlap:
            if len(chunk) >= self.overlap:
                self._tail = chunk[-self.overlap:].copy()
            return chunk
        fade_in  = np.linspace(0.0, 1.0, self.overlap, dtype=np.float32)
        fade_out = 1.0 - fade_in
        out = chunk.copy()
        out[:self.overlap] = chunk[:self.overlap] * fade_in + self._tail * fade_out
        self._tail = chunk[-self.overlap:].copy()
        return out


# ── RVC wrapper ───────────────────────────────────────────────────────────────

class RVCWrapper:
    def __init__(self, model_path: str, index_path: str,
                 pitch: int, f0_method: str, index_rate: float, device: str):
        self.pitch       = pitch
        self.f0_method   = f0_method
        self.index_rate  = index_rate
        self.passthrough = False
        self._rvc        = None

        if not _RVC_OK:
            return

        print(f"[rvc] Loading {Path(model_path).name} …")
        self._rvc = RVCInference(device=device)
        self._rvc.load_model(model_path, index_path or "")
        print("[rvc] Model ready.")

    def toggle_passthrough(self):
        self.passthrough = not self.passthrough
        state = "ON (raw mic)" if self.passthrough else "OFF (converting)"
        print(f"[rvc] pass-through: {state}")

    def convert(self, audio_f32: np.ndarray, sr: int) -> np.ndarray:
        """
        audio_f32 : float32 mono, `sr` Hz
        returns   : float32 mono, `sr` Hz
        """
        if self._rvc is None or self.passthrough:
            return audio_f32

        # Write to a temp WAV, run RVC, read result back.
        # /tmp is tmpfs on Linux so this adds ~1–2 ms overhead for 400 ms chunks.
        in_fd,  in_path  = tempfile.mkstemp(suffix=".wav")
        out_path = in_path.replace(".wav", "_out.wav")
        os.close(in_fd)

        try:
            sf.write(in_path, audio_f32, sr, subtype="FLOAT")

            self._rvc.infer_file(
                in_path, out_path,
                f0_up_key=self.pitch,
                f0_method=self.f0_method,
                index_rate=self.index_rate,
                protect=0.33,
            )

            result, out_sr = sf.read(out_path, dtype="float32")
            if result.ndim > 1:
                result = result.mean(axis=1)

            # Resample back to input sr if RVC changed it (e.g. 40000 → sr)
            if out_sr != sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, out_sr)
                result = resample_poly(result, sr // g, out_sr // g).astype(np.float32)

            return result

        except Exception as exc:
            print(f"[rvc] inference error: {exc}")
            return audio_f32

        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ── Audio pipeline ────────────────────────────────────────────────────────────

class AudioPipeline:
    def __init__(self, rvc: RVCWrapper, sr: int, chunk_ms: int,
                 in_device, out_device):
        self.rvc     = rvc
        self.sr      = sr
        self.chunk   = int(sr * chunk_ms / 1000)
        self.overlap = int(self.chunk * 0.15)
        self.blender = SOLABlender(self.overlap)

        self._in_q   = queue.Queue(maxsize=4)
        self._out_q  = queue.Queue(maxsize=8)
        self._running = False
        self._n_proc  = 0

        # Pre-fill with silence so playback doesn't underrun while the first
        # chunk is being processed (RVC takes 90–170 ms on first call).
        silence = np.zeros(self.chunk, dtype=np.float32)
        for _ in range(3):
            self._out_q.put(silence.copy())

        self._in_stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            device=in_device, blocksize=self.chunk,
            callback=self._in_cb,
        )
        self._out_stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            device=out_device, blocksize=self.chunk,
            callback=self._out_cb,
        )

    # sounddevice callbacks run on a real-time audio thread — keep them minimal.

    def _in_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[audio in]  {status}")
        try:
            self._in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop: capture thread will get the next chunk immediately

    def _out_cb(self, outdata, frames, time_info, status):
        if status:
            print(f"[audio out] {status}")
        try:
            chunk = self._out_q.get_nowait()
        except queue.Empty:
            chunk = np.zeros(frames, dtype=np.float32)
        # Guarantee exact `frames` length regardless of RVC output length variance.
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
        elif len(chunk) > frames:
            chunk = chunk[:frames]
        outdata[:, 0] = chunk

    def _proc_loop(self):
        chunk_ms = self.chunk / self.sr * 1000
        log_every = max(1, round(2000 / chunk_ms))  # log ~every 2 s

        while self._running:
            try:
                raw = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            converted = self.rvc.convert(raw, self.sr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            blended = self.blender.blend(converted)
            # Trim/pad to exactly chunk samples (RVC output length can drift slightly)
            if len(blended) > self.chunk:
                blended = blended[:self.chunk]
            elif len(blended) < self.chunk:
                blended = np.pad(blended, (0, self.chunk - len(blended)))

            try:
                self._out_q.put_nowait(blended)
            except queue.Full:
                self._out_q.get_nowait()  # discard oldest to keep latency tight
                self._out_q.put_nowait(blended)

            self._n_proc += 1
            if self._n_proc % log_every == 0:
                mode = "pass-through" if self.rvc.passthrough else f"pitch={self.rvc.pitch:+d}"
                print(f"[rvc] {elapsed_ms:.0f}ms inference  "
                      f"in_q={self._in_q.qsize()}  out_q={self._out_q.qsize()}  {mode}")

    def start(self):
        self._running = True
        self._proc_thread = threading.Thread(target=self._proc_loop, daemon=True)
        self._proc_thread.start()
        self._in_stream.start()
        self._out_stream.start()
        print(f"[audio] Running — sr={self.sr}  chunk={self.chunk/self.sr*1000:.0f}ms  "
              f"overlap={self.overlap/self.sr*1000:.0f}ms")

    def stop(self):
        self._running = False
        self._in_stream.stop()
        self._out_stream.stop()
        self._proc_thread.join(timeout=2.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time RVC voice conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="Print audio device list and exit")
    parser.add_argument("--model",  default=os.getenv("RVC_MODEL", ""),
                        help="Path to .pth RVC model  (env: RVC_MODEL)")
    parser.add_argument("--index",  default=os.getenv("RVC_INDEX", ""),
                        help="Path to .index file     (env: RVC_INDEX)")
    parser.add_argument("--pitch",  type=int, default=int(os.getenv("RVC_PITCH", "0")),
                        help="Pitch shift semitones   (env: RVC_PITCH)")
    parser.add_argument("--f0-method", default="rmvpe",
                        choices=["rmvpe", "harvest", "crepe", "fcpe"],
                        help="Pitch extraction method")
    parser.add_argument("--index-rate", type=float, default=0.75,
                        help="Feature retrieval blend (0=off, 1=full index)")
    parser.add_argument("--input-device",  type=int, default=None,
                        help="Input device index  (default: system default)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Output device index (default: system default)")
    parser.add_argument("--sr", type=int, default=16_000,
                        help="Sample rate (Hz) — match your virtual mic's SR")
    parser.add_argument("--chunk-ms", type=int, default=400,
                        help="Chunk size ms — smaller=lower latency, larger=better quality")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device for inference")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.model:
        print("[ERROR] --model is required.\n"
              "  Example: python voice_convert.py --model models/my_voice.pth\n"
              "  Run --list-devices to see audio device indices.")
        sys.exit(1)

    rvc = RVCWrapper(
        model_path=args.model,
        index_path=args.index,
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        device=args.device,
    )

    pipeline = AudioPipeline(
        rvc=rvc,
        sr=args.sr,
        chunk_ms=args.chunk_ms,
        in_device=args.input_device,
        out_device=args.output_device,
    )
    pipeline.start()

    print("\nHotkeys (press Enter after each):  q=quit  +=pitch up  -=pitch down  p=pass-through\n")
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == "q":
                break
            elif cmd == "+":
                rvc.pitch += 1
                print(f"[rvc] pitch: {rvc.pitch:+d} semitones")
            elif cmd == "-":
                rvc.pitch -= 1
                print(f"[rvc] pitch: {rvc.pitch:+d} semitones")
            elif cmd == "p":
                rvc.toggle_passthrough()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\n[audio] Stopping …")
        pipeline.stop()


if __name__ == "__main__":
    main()
