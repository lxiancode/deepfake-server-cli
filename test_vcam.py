import numpy as np
import pyvirtualcam
import time

width, height, fps = 1280, 720, 30

print("Opening virtual camera...")
with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB) as cam:
    print(f"Virtual camera active: {cam.device}")
    print("Sending red frames for 15 seconds — check Zoom now...")
    start = time.time()
    while time.time() - start < 15:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # red
        cam.send(frame)
        cam.sleep_until_next_frame()
    print("Done.")
