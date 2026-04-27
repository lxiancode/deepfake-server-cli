# Deepfake Research Plan
_Last updated: 2026-04-25_

---

## Setup Summary

- **Local:** MacBook (M-series, no GPU) — runs `client.py` only
- **Remote server:** `141.212.114.81` — Ubuntu 24.04, 8× NVIDIA RTX 6000 Ada (49GB VRAM each), GPUs 0–3 free, GPUs 4–7 occupied by VLLM
- **Current stack:** inswapper_128 (face swap) + GPEN-256/512 (enhancement) + OBS Virtual Camera → Zoom
- **Reference paper:** DeepSpeak dataset (CVPR 2026) — arxiv.org/abs/2408.05366

---

## Goals

### G1 — Improve face swap quality
Baseline is inswapper_128 + GPEN. Current issues: face appears too large, lacks texture, poor mouth/teeth.

### G2 — Real-time voice cloning
Add voice conversion alongside face swap so both audio and video are synthetic in Zoom calls. Pipeline: mic → voice model (server) → virtual audio device → Zoom.

### G3 — Human perceptual evaluation + behavioral impact
Two-part goal:
- **Detection:** Can people tell it's a deepfake in a real-time video call context?
- **Behavioral impact:** How do people respond when they *don't* detect it — trust, compliance, susceptibility to deception/scams?

DeepSpeak did not do either of these. The behavioral harm angle (real-time deepfakes used in scam/social engineering contexts) is a clear literature gap and a strong novel contribution.

### G4 — Adversarial robustness
Evaluate and improve robustness of generation and detection models against disruptions (e.g., hand occlusion, lighting changes, compression artifacts). Two sub-directions:
- Making detectors more robust to adversarial examples
- Making generators more resilient to physical disruptions

---

## Overall vision

Build a fully immersive real-time deepfake pipeline — face swap + voice cloning simultaneously in a live video call — realistic enough that the person on the other end does not notice. Detection and adversarial robustness are a **second phase** once the generation pipeline is solid.

```
Phase 1 (now):   Webcam + Mic → [face swap + voice clone] → Zoom call
Phase 2 (later): Evaluate pipeline against detectors, study human responses
```

## Recommended Order

| Priority | Goal | Why |
|----------|------|-----|
| 1 | G1 — Face swap quality | Foundation of the pipeline |
| 2 | G2 — Voice cloning | Completes the real-time immersive experience |
| 3 | G3 — Human perceptual study | Needs G1+G2 working well first |
| 4 | G4 — Detection + adversarial | Phase 2, builds on complete pipeline |

---

## Next Steps

### G1 — Face swap quality improvements

**Immediate (no code changes needed):**
- [ ] Press `r` while running to enable GPEN-512 enhancement
- [ ] Press `m` to enable mouth mask (preserves real teeth/mouth)
- [ ] Press `-` to reduce opacity slightly for more natural blending

**Config changes (`.env`):**
- [ ] Set `SEND_SCALE=0.75` (currently 0.5) — sends higher res frames to server
- [ ] Set `SEND_QUALITY=80` (currently 60) — less JPEG compression

**Model upgrades (server-side, medium effort):**
- [ ] Replace GPEN with **CodeFormer** — better texture restoration, fidelity weight tunable (github.com/sczhou/CodeFormer)
- [ ] Try **SimSwap** as alternative to inswapper — better preserves facial attributes and face shape
- [ ] Try **FaceFusion** — supports multiple swap models + built-in CodeFormer enhancement

### G2 — Real-time voice cloning

**Recommended model: Seed-VC** (state of the art, 2024) or **RVC** (most community support)

Pipeline to build:
```
Mic → voice conversion model (GPU 1 on server) → virtual audio output → Zoom
```

- [ ] Evaluate Seed-VC vs RVC latency on the server
- [ ] Set up virtual audio device on Mac (BlackHole — free, open source)
- [ ] Build a voice client similar to `client.py` that streams audio to server and back

### G3 — Human perceptual evaluation

Study design (to finalize):
- [ ] Decide stimulus conditions: face-only / voice-only / face+voice / real
- [ ] Decide platform: Prolific (recommended) or in-person
- [ ] Build stimulus set from your existing data (Celeb-DF-v2, FakeAVCeleb)
- [ ] Define metrics: detection accuracy, confidence rating (1–7 scale), response time

### G4 — Adversarial robustness

- [ ] Clarify direction: attack on detectors vs. robustness of generator
- [ ] Evaluate current system against detectors from DeepSpeak paper (all failed to generalize)
- [ ] Design occlusion/disruption test protocol (hand waving, partial face coverage)

---

## Open Questions / Clarifications Needed

1. **Adversarial direction:** Are you trying to make deepfakes *harder to detect* (attack detectors), or make detectors *more robust* to adversarial examples? Or both?

2. **Hand occlusion specifically:** Is the goal to keep generating a good face swap *despite* occlusion, or to test whether detectors can be fooled by occluding the face?

3. **Voice cloning target:** Do you want to clone a *specific person's voice* (identity transfer, like the face swap) or just disguise your own voice?

4. **Human study scope:** Is this for a paper, or internal validation? This affects how rigorous the study design needs to be (IRB, sample size, counterbalancing, etc.).

5. **Server access:** Is the server shared with others? GPUs 4–7 are currently running a VLLM job. Can you use GPU 0 (partially occupied, 3.7GB used) or only GPUs 1–3?

6. **CodeFormer integration:** The server runs the existing `deepfake-server-cli` repo. Do you want to modify `server.py` directly, or run a separate enhanced pipeline?

---

## Key References

- DeepSpeak dataset (CVPR 2026): arxiv.org/abs/2408.05366
  - Used 14 video engines (including inswapper, SimSwap, FaceFusion) + 3 voice cloning APIs
  - Found all detectors fail to generalize to new deepfake methods
  - No human perceptual study — gap to fill
- CodeFormer (NeurIPS 2022): github.com/sczhou/CodeFormer
- SimSwap: arxiv.org/abs/2106.06340
- Your datasets: Celeb-DF-v2, FakeAVCeleb v1.2, FaceForensics++
