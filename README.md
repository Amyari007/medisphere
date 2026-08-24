# MediSphere — Post-Stroke Hand Rehabilitation Platform

**Problem** → Post-stroke hand therapy needs objective, repeatable movement metrics — most clinics rely on visual assessment alone.
**Solution** → Real-time webcam hand tracking (MediaPipe) feeding a full biomechanics pipeline, with live audio-visual biofeedback.
**Validation** → Every metric is unit-tested against known synthetic ground truth (7 modules + full pipeline dry-run — see table below).
**Result** → A working, gracefully-degrading real-time system: joint angles, tremor (FFT), smoothness (jerk-based), finger independence, session logging.

## Demo

!\[Open hand — high openness, motor score trending up](demo/demo\_open\_hand.jpeg)
!\[Closed fist — low openness, tracked repetition](demo/demo\_closed\_fist.jpeg)
!\[Partial close — mid-range extension](demo/demo\_partial\_close.jpeg)

## What's Tested

|Component|Status|
|-|-|
|Joint angles, ROM, openness|Unit-tested vs. synthetic geometry|
|Velocity / acceleration / smoothness|Unit-tested vs. synthetic motion|
|Finger independence (correlation-based)|Unit-tested, incl. near-immobile-finger fix|
|Tremor detection (filtered FFT)|Unit-tested vs. known injected frequency|
|Rep counting, reaction time, fatigue|Unit-tested vs. synthetic sequences|
|Session logging (CSV + SQLite)|Round-trip tested|
|Full pipeline (`main.py`)|Dry-run tested end-to-end, camera/MediaPipe mocked|
|3D hologram (OpenGL)|Verified in Linux sandbox (software GPU)|
|Audio feedback|Waveform verified via FFT; real hardware playback untested|

**Not a clinical instrument.** The "Motor Score" is a custom index for this project — not Fugl-Meyer, ARAT, or any standardized scale. Clinical-flag thresholds are heuristic (informed by general stroke-rehab literature direction), not derived from patient data. Treat this as a movement-quantification and biofeedback demo, not a diagnostic device.

## How It Works

Continuous hand openness drives a rotating 3D hologram (or procedural fallback) — size, rotation speed, and audio pitch all track hand state in real time.

|Input|Effect|
|-|-|
|Open hand|Sphere grows, pitch drops|
|Close hand|Sphere shrinks to a point, pitch rises|
|Full open→close cycle|Counts as one repetition|
|`c`|Reaction-time cue|
|`r`|Reset session|
|`q`|Quit, write session report|

## Setup

```bash
git clone https://github.com/Amyari007/medisphere.git
cd medisphere
pip install -r requirements.txt
curl -L -o models/hand\_landmarker.task https://storage.googleapis.com/mediapipe-models/hand\_landmarker/hand\_landmarker/float16/latest/hand\_landmarker.task
python main.py
```

Optionally add `brain\_hologram.glb` to `models/` for the full 3D hologram — otherwise a procedural fallback (NeuroSphere) runs automatically.

## Architecture

```
medisphere/
├── main.py                 # real-time loop, orchestration
├── vision/                 # MediaPipe wrapper, gesture recognition
├── biomechanics/            # joint angles, ROM, kinematics
├── signal\_processing/        # Butterworth filter, FFT tremor detection, stability
├── session/                   # reps, reaction time, fatigue, motor score, logging
├── audio/                      # pitch-mapped continuous tone feedback
├── visualization/                # NeuroSphere, OpenGL hologram, skeleton overlay, HUD
├── tests/                          # one file per module + integration + main dry-run
└── requirements.txt
```

## Fallback Behavior

* No `.glb` file or OpenGL context fails → falls back to procedural NeuroSphere
* No audio device → audio silently disabled, HUD notes it; everything else unaffected
* No hand in frame → metrics show "measuring...", never a stale/fabricated value

## Key Engineering Notes

* Rep counting is a real state machine — completes on an actual open→close transition, not a timer
* Fatigue trend uses linear regression (`numpy.polyfit`) on real per-rep peak values
* Smoothness = rolling-window Log Dimensionless Jerk (Hogan \& Sternad, 2009)
* Tremor detection scoped to 3–12 Hz, Butterworth-filtered before FFT, to avoid misreading ordinary movement as tremor
* Finger independence excludes near-zero-variance fingers from correlation (real bug caught during integration testing, not hypothetical)

## Testing

```bash
python tests/test\_gestures.py
python tests/test\_biomechanics.py
python tests/test\_signal\_processing.py
python tests/test\_audio.py
python tests/test\_session.py
python tests/test\_integration.py
python tests/test\_main\_dryrun.py
```

## Roadmap

Depth camera fusion (RealSense/OAK-D) · EMG sensor fusion · clinician multi-patient dashboard · structured multi-exercise protocols

## License

MIT



\## Testing \& Validation



\*\*57 / 57 automated tests passing.\*\* Full methodology, per-module benchmarks, and known limitations are documented in \[`TESTING.md`](TESTING.md) — this section is a summary.



Every computational module is validated against \*\*synthetic inputs with a known, independently-calculated correct answer\*\*, not just "does it run."



| Module | Tests | Benchmark used |

|---|---|---|

| Gestures | 8 | Hand-built landmark coordinates with known geometry |

| Biomechanics (joints, ROM, independence) | 9 | Hand-calculated angles (e.g. colinear points → ≈180°, right-angle bend → ≈90°) |

| Kinematics (velocity, smoothness) | included above | Exact arithmetic speed (100 px/s) checked against measured output |

| Signal processing (tremor, filtering) | 5 | A known, deliberately injected frequency (e.g. 6 Hz) recovered by the detector |

| Audio waveform | 3 | Generated tone re-analyzed with an independent FFT to confirm it contains the requested frequency |

| Session logic (reps, reaction, fatigue, motor score, logging) | 12 | Known timestamp/state sequences; SQLite+CSV round-trip verified by reading back written data |

| Clinical flags | 17 | Values placed exactly at, above, and below every documented threshold |

| Full pipeline integration | 2 | A complete simulated multi-repetition session run through every module together |

| `main.py` itself | 1 | The real application code, camera/MediaPipe swapped for deterministic input — not a reimplementation |



Reproduce it yourself:

```bash

pip install pytest

python -m pytest tests/ -v

```



\*\*Two real bugs were caught this way during live testing\*\* (visibility against a real background, and repetition-count inflation from landmark jitter) — both are detailed in `TESTING.md`, along with dedicated regression tests added to prevent them recurring silently.



\*\*What isn't validated:\*\* this is not a clinical instrument. The Motor Score and clinical flag thresholds are heuristics defined for this project, not derived from patient data or compared against a validated clinical scale. See `TESTING.md` for the full list of what remains unverified.

