# MediSphere — Post-Stroke Hand Rehabilitation Platform

**Problem** → Post-stroke hand therapy needs objective, repeatable movement metrics — most clinics rely on visual assessment alone.
**Solution** → Real-time webcam hand tracking (MediaPipe) feeding a full biomechanics pipeline, with live audio-visual biofeedback.
**Validation** → Every metric is unit-tested against known synthetic ground truth (7 modules + full pipeline dry-run — see table below).
**Result** → A working, gracefully-degrading real-time system: joint angles, tremor (FFT), smoothness (jerk-based), finger independence, session logging.

## What's Tested

| Component | Status |
|---|---|
| Joint angles, ROM, openness | Unit-tested vs. synthetic geometry |
| Velocity / acceleration / smoothness | Unit-tested vs. synthetic motion |
| Finger independence (correlation-based) | Unit-tested, incl. near-immobile-finger fix |
| Tremor detection (filtered FFT) | Unit-tested vs. known injected frequency |
| Rep counting, reaction time, fatigue | Unit-tested vs. synthetic sequences |
| Session logging (CSV + SQLite) | Round-trip tested |
| Full pipeline (`main.py`) | Dry-run tested end-to-end, camera/MediaPipe mocked |
| 3D hologram (OpenGL) | Verified in Linux sandbox (software GPU) |
| Audio feedback | Waveform verified via FFT; real hardware playback untested |

**Not a clinical instrument.** The "Motor Score" is a custom index for this project — not Fugl-Meyer, ARAT, or any standardized scale. Clinical-flag thresholds are heuristic (informed by general stroke-rehab literature direction), not derived from patient data. Treat this as a movement-quantification and biofeedback demo, not a diagnostic device.

## How It Works

Continuous hand openness drives a rotating 3D hologram (or procedural fallback) — size, rotation speed, and audio pitch all track hand state in real time.

| Input | Effect |
|---|---|
| Open hand | Sphere grows, pitch drops |
| Close hand | Sphere shrinks to a point, pitch rises |
| Full open→close cycle | Counts as one repetition |
| `c` | Reaction-time cue |
| `r` | Reset session |
| `q` | Quit, write session report |

## Setup

```bash
git clone https://github.com/Amyari007/medisphere.git
cd medisphere
pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python main.py
```

Optionally add `brain_hologram.glb` to `models/` for the full 3D hologram — otherwise a procedural fallback (NeuroSphere) runs automatically.

## Architecture

```
medisphere/
├── main.py                 # real-time loop, orchestration
├── vision/                 # MediaPipe wrapper, gesture recognition
├── biomechanics/            # joint angles, ROM, kinematics
├── signal_processing/        # Butterworth filter, FFT tremor detection, stability
├── session/                   # reps, reaction time, fatigue, motor score, logging
├── audio/                      # pitch-mapped continuous tone feedback
├── visualization/                # NeuroSphere, OpenGL hologram, skeleton overlay, HUD
├── tests/                          # one file per module + integration + main dry-run
└── requirements.txt
```

## Fallback Behavior

- No `.glb` file or OpenGL context fails → falls back to procedural NeuroSphere
- No audio device → audio silently disabled, HUD notes it; everything else unaffected
- No hand in frame → metrics show "measuring...", never a stale/fabricated value

## Key Engineering Notes

- Rep counting is a real state machine — completes on an actual open→close transition, not a timer
- Fatigue trend uses linear regression (`numpy.polyfit`) on real per-rep peak values
- Smoothness = rolling-window Log Dimensionless Jerk (Hogan & Sternad, 2009)
- Tremor detection scoped to 3–12 Hz, Butterworth-filtered before FFT, to avoid misreading ordinary movement as tremor
- Finger independence excludes near-zero-variance fingers from correlation (real bug caught during integration testing, not hypothetical)

## Testing

```bash
python tests/test_gestures.py
python tests/test_biomechanics.py
python tests/test_signal_processing.py
python tests/test_audio.py
python tests/test_session.py
python tests/test_integration.py
python tests/test_main_dryrun.py
```

## Roadmap

Depth camera fusion (RealSense/OAK-D) · EMG sensor fusion · clinician multi-patient dashboard · structured multi-exercise protocols

## License

MIT
