# SOP — How to Run the Real-Time Face Swap Pipeline
_Last updated: 2026-04-25_

This is the exact sequence of commands to go from zero to face-swapped video in Zoom.

---

## Every time you want to run it

### Step 1 — Start the face swap client (Mac terminal)

```bash
cd /Users/xianl/Documents/Research/2026-deepfake/deepfake-server-cli
source .venv/bin/activate
python client.py
```

You should see:
```
[capture] Camera 1 opened at 1280x720
[ws] Connected.
[source] Sent 1 image from ./images/face.jpg
[source_set] OK — 1/1 image(s) had a detectable face
[vcam] Virtual camera active: OBS Virtual Camera
[client] capture=30.0fps  sent=22.0fps  swap=22.0fps  roundtrip=44ms
```

A face swap preview window will open on your screen.

### Step 2 — Start OBS virtual camera

1. Open OBS (`/Applications/OBS.app`)
2. Under **Sources**, confirm you have a **macOS Screen Capture** source set to Display
3. Click **Start Virtual Camera** (bottom right Controls panel)

### Step 3 — Select camera in Zoom

1. Open Zoom → Settings → Video
2. Under Camera, select **OBS Virtual Camera**
3. You should see your face-swapped video in the Zoom preview

---

## Controls while running

| Key | Action |
|-----|--------|
| `r` | Cycle enhancement: off → GPEN-256 → GPEN-512 (use GPEN-512 for best quality) |
| `m` | Toggle mouth mask on/off (turn ON for better teeth realism) |
| `+` / `-` | Increase / decrease blend opacity |
| `s` | Resend source face to server |
| `q` | Quit |

**Recommended settings for best quality:** press `r` twice (GPEN-512), press `m` once (mouth mask on).

---

## If something is broken

### "Cannot open camera" or black screen
```bash
# Check camera index — try 0 if 1 doesn't work
# Edit .env and change CAMERA_INDEX=1 to CAMERA_INDEX=0
nano .env
```

### Client can't connect to server
```bash
# Verify server is running
curl http://141.212.114.81:8765/health
# Should return: {"status":"ok","models_loaded":true,...}
```

### OBS not showing screen / virtual camera not working
- Go to **System Settings → Privacy & Security → Screen & System Audio Recording** → make sure OBS is enabled
- Go to **System Settings → Privacy & Security → Camera** → make sure OBS and Terminal are enabled
- Restart OBS after changing permissions

### pyvirtualcam error on startup
- Make sure OBS is NOT running with Virtual Camera already started before launching `client.py`
- Or ignore the error — OBS screen capture approach works fine without pyvirtualcam

---

## First time setup (one-time only)

Only needed if setting up on a new machine or fresh clone.

### Mac (client)
```bash
# Clone the repo
git clone https://github.com/byron123t/deepfake-server-cli.git
cd deepfake-server-cli

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements_client.txt

# Copy and configure env file
cp .env.example .env
# Edit .env — set:
#   SERVER_URL=ws://141.212.114.81:8765/ws
#   SOURCE_PATH=./images/face.jpg
#   CAMERA_INDEX=1
open -e .env

# Add your source face image
mkdir -p images
# copy your face.jpg into images/

# Install OBS (one time)
brew install --cask obs
# Then open OBS, go to System Settings and enable Camera Extension
```

### Server (already running — only if server needs to be restarted)
```bash
# SSH into server
ssh your_username@141.212.114.81

# Navigate to repo and activate environment
cd deepfake-server-cli
conda activate rope

# Start server on GPU 1
CUDA_VISIBLE_DEVICES=1 python server.py

# Verify
curl http://localhost:8765/health
```

---

## Current .env values (Mac)

```ini
SERVER_URL=ws://141.212.114.81:8765/ws
SOURCE_PATH=./images/face.jpg
CAMERA_INDEX=1
CAPTURE_WIDTH=1280
CAPTURE_HEIGHT=720
SEND_SCALE=0.5
SEND_QUALITY=60
```
