# Literature Review Brief — Deepfake Generation & Detection Research

## Who I am
I am a researcher building a real-time deepfake system (face swap + voice cloning) for research purposes. My system currently uses inswapper_128 for face swapping and GPEN for enhancement, running on a server with 8× NVIDIA RTX 6000 Ada GPUs. I am working toward a paper.

## Key reference paper
**DeepSpeak (CVPR 2026)** — arxiv.org/abs/2408.05366
- Built a multimodal dataset with 500 participants, 14 video synthesis engines, 3 voice cloning engines
- Found that all evaluated detectors fail to generalize to new deepfake methods
- Did NOT include a human perceptual study — this is a gap I want to fill
- Video engines used: FaceFusion, INSwapper, SimSwap, Wav2Lip, LatentSync, LivePortrait, HelloMeme, Memo
- Voice engines used: ElevenLabs, PlayAI, Speechify (all commercial APIs)

## My datasets
Currently have locally (not limited to these):
- Celeb-DF-v2
- FakeAVCeleb v1.2
- FaceForensics++ (downloading)

Also have pending access requests to more recent datasets, and I am interested in using:
- Any recent benchmark datasets (2023–2026)
- In-the-wild deepfake video datasets (real-world, not lab-generated)
- Multimodal (audio+video) deepfake datasets

The literature review should recommend the most relevant and comprehensive datasets available, regardless of whether I currently have access. Include where/how to request access for gated datasets.

## Research framing

The ultimate goal is a **fully immersive real-time deepfake pipeline**: face swap + voice cloning running simultaneously in a live video call, realistic enough that the person on the other end does not notice.

```
Phase 1 (current focus): Webcam + Mic → [face swap + voice clone] → Zoom/video call
Phase 2 (future):        Evaluate against detectors, study human behavioral responses
```

**Phase 1 is the priority for this literature review.** Detection and adversarial robustness are out of scope for now — focus on what makes the generation pipeline maximally realistic and real-time capable.

The broader research questions (human vulnerability, behavioral responses in scam/deception contexts) are noted as future directions and known literature gaps, but do not need to be the focus of the literature search right now.

## What I need from this literature review

### 1. Perceptual realism in face swap generation
The goal is not just visual quality metrics (FID, SSIM) but **perceptual realism** — would a human, in a real-time video call context, believe this is a real person?

What are the best current methods for:
- High-quality real-time face swapping (better than inswapper_128)?
- Face enhancement/restoration post-processing (better than GPEN — e.g., CodeFormer, GFPGAN, RestoreFormer)?
- Handling mouth/teeth realism in face swaps?
- Temporal consistency for video (reducing flicker, maintaining identity across frames)?
- Skin texture, lighting adaptation, and hair boundary blending?

Focus on methods with available open-source code, ideally with ONNX support or PyTorch inference. Flag which have been evaluated on perceptual realism specifically (human ratings, MOS scores) vs. only automated metrics.

### 2. Real-time voice cloning
What are the best current methods for:
- Real-time voice conversion/cloning (low latency, <200ms)?
- Open-source alternatives to commercial APIs (ElevenLabs, PlayAI, Speechify)?
- Audio-visual synchronization (lip sync + voice matching)?
- Emotional/prosodic transfer in voice cloning?
Key candidates to evaluate: Seed-VC, RVC, FreeVC, XTTS v2, OpenVoice v2.

### 3. Human perceptual studies on deepfakes
What work exists on:
- Human ability to detect deepfakes in real-time or video call contexts (not just static images)?
- Study designs: forced choice, confidence ratings, naturalistic exposure?
- Multimodal deepfakes (face+voice) vs. unimodal — how does combined audio-visual affect detection?
- Human vulnerability in adversarial/scam contexts — does anyone study behavioral outcomes (compliance, deception) rather than just detection accuracy?
- What factors make deepfakes more or less detectable to humans (quality, familiarity with target, context)?

**Specifically flag:** any gap where real-time, multimodal deepfake realism has NOT been studied in terms of human behavioral responses (trust, susceptibility to deception).

### 4. Adversarial robustness
What are the state-of-the-art methods for:
- Adversarial attacks on deepfake detectors?
- Physical adversarial perturbations (occlusion, lighting changes) affecting generation/detection?
- Making detectors robust to domain shift?
- Anti-forensic techniques in generation (making outputs harder to detect as fake)?

## Scope
Please search across:
- CVPR, ICCV, ECCV 2024–2026
- NeurIPS, ICML, ICLR 2024–2026
- arXiv preprints (2024–2026)
- OpenReview submissions

## Output format I want
For each area above:
1. Top 5–10 most relevant papers with title, venue, year, one-sentence summary
2. Key techniques/methods ranked by: quality, open-source availability, real-time feasibility
3. Specific recommendations for what I should implement given my setup (server GPU, real-time requirement)
4. Identified gaps that could be novel contributions
