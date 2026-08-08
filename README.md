# MediSphere — Post-Stroke Hand Rehabilitation Platform

A real-time computer vision rehabilitation platform: MediaPipe hand tracking
feeds a full biomechanics/kinematics/signal-processing pipeline computing
clinically meaningful movement metrics, presented through a futuristic HUD
with a continuous audio-visual biofeedback loop (a rotating 3D hologram —
or a procedural fallback — whose size and pitch track your hand's real
openness).

## What's real vs. what's a demo — please read this

**Every metric in the HUD is computed from real tracked hand landmarks.**
Nothing is randomly walked or hardcoded. See the module list below and
`tests/` for validation against known synthetic ground truth (e.g. the
tremor detector is fed a *known* 5–6 Hz signal and checked against it; the
smoothness metric is checked against genuinely smooth vs. jerky synthetic
motion).

**This is not a validated clinical instrument.** The "Motor Score" is a
custom composite index defined for this project — it is *not* the
Fugl-Meyer Assessment, ARAT, or any standardized clinical scale. Treat all
of this as a real-time movement-quantification and biofeedback demo, not a
diagnostic device.

**What's genuinely tested vs. what needs your hardware to confirm:**

| Component | Status |
|---|---|
| Joint angles, ROM, openness, extension % | Unit-tested against known synthetic geometry |
| Velocity / acceleration / smoothness (jerk) | Unit-tested against known synthetic motion |
| Finger independence (correlation-based) | Unit-tested, including a robustness fix for near-immobile fingers |
| Tremor detection (filtered FFT) | Unit-tested against a known injected frequency |
| Rep counting, reaction time, fatigue trend | Unit-tested against known synthetic sequences |
| Session logging (CSV + SQLite) | Round-trip tested (write, reopen, verify) |
| Full pipeline wiring in `main.py` | Dry-run tested end-to-end with the camera and MediaPipe mocked out |
| **3D hologram rendering (OpenGL)** | Rendered and verified end-to-end in a Linux sandbox using a *software* GPU (Mesa llvmpipe) — real driver behavior on your machine is the next test |
| **Audio feedback (pitch mapping)** | Waveform math verified via FFT on the generated signal itself — actual speaker playback untested (no audio hardware in the build sandbox) |

If the hologram or audio don't work on your machine, the app is designed to
degrade gracefully rather than crash — see "Fallback behavior" below.

## Gestures & interaction

The sphere/hologram isn't a binary "summoned" prop — its size, rotation
speed, and audio pitch all track your **continuous hand openness** in
real time.

| Input | Effect |
|---|---|
| Open hand | Sphere/hologram grows, particles emit, pitch drops |
| Close hand | Sphere/hologram smoothly shrinks to a glowing point, pitch rises |
| Full open→close cycle | Counts as one repetition |
| `c` key | Issues a reaction-time cue ("OPEN NOW") |
| `r` key | Resets the session (fresh rep count, fresh log) |
| `q` key | Quits and writes a session report |

## Setup

```bash
git clone <your-repo-url>
cd medisphere
pip install -r requirements.txt
```

Then get the two model files (see `models/README.md` for details):

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Optionally, copy your `brain_hologram.glb` into `models/` for the real 3D
hologram (otherwise the procedural NeuroSphere is used automatically).

Run it:

```bash
python main.py
```

## Fallback behavior (won't crash on a bad environment)

- **No `brain_hologram.glb`, or OpenGL/GLFW context creation fails** → falls
  back to the procedural NeuroSphere. A message prints explaining which
  path was taken; the app keeps running either way.
- **No audio output device, or PortAudio isn't installed** → audio feedback
  is silently disabled; a small "(audio feedback off)" note shows in the
  HUD. Vision, metrics, and logging are unaffected.
- **No hand in frame** → all metrics show "measuring..." rather than a
  stale or fabricated number.

## Project structure

```
medisphere/
├── main.py                      # orchestrates everything, the real-time loop
├── vision/
│   ├── hand_tracker.py          # MediaPipe Tasks API wrapper
│   └── gestures.py              # geometric gesture recognition, palm center
├── biomechanics/
│   ├── joints.py                # MCP/PIP/DIP angles, ROM, openness, finger independence
│   └── kinematics.py            # velocity, acceleration, jerk-based smoothness
├── signal_processing/
│   ├── filters.py                # Butterworth low-pass filter (scipy)
│   ├── tremor.py                 # filtered FFT tremor detection
│   └── stability.py              # positional-jitter stability score
├── session/
│   ├── reps.py                   # repetition counting from real state transitions
│   ├── reaction.py               # cue-to-response reaction time
│   ├── fatigue.py                # linear-regression fatigue trend across reps
│   ├── motor_score.py            # composite score (explicitly not a clinical scale)
│   └── logger.py                 # CSV + SQLite session storage, report generation
├── audio/
│   └── feedback.py               # continuous tone, pitch mapped to hand openness
├── visualization/
│   ├── neurosphere.py            # procedural rotating/particle/bloom sphere
│   ├── hologram.py               # real OpenGL 3D GLB renderer + compositor
│   ├── skeleton.py                # finger heat-map skeleton overlay
│   └── hud.py                     # clinical HUD, trend graphs, cue banner
├── tests/                         # one test file per module above, plus:
│   ├── test_integration.py        # full pipeline through synthetic session data
│   └── test_main_dryrun.py        # main.py itself, camera/MediaPipe mocked out
└── requirements.txt
```

## Clinical flags — heuristic, not diagnostic

The HUD shows live warnings (e.g. "Reduced finger extension", "Tremor
detected - significant", "Hand opening below target") when a metric crosses
a threshold defined in `session/clinical_flags.py`. Read this carefully:

**These thresholds are not validated clinical cutoffs.** They're reasonable
heuristic values translated from the general direction of findings in
movement-science / stroke-rehab literature (e.g. "post-stroke smoothness is
often markedly reduced") into a concrete number, so the app can flag
*something* actionable. They were not derived from patient data or a
clinical study. Do not use these flags to make clinical decisions — treat
them as a prompt to look more closely, not a finding.

## Design notes

- **Rep counting is a real state machine**, not a timer — a rep completes on
  an actual open→close transition.
- **Fatigue detection uses linear regression** (`numpy.polyfit`) on real
  per-rep peak values, not a scripted decline.
- **Smoothness is a rolling-window approximation of Log Dimensionless Jerk**
  (Hogan & Sternad, 2009), a real measure from movement-science literature —
  approximate because true LDLJ is defined per discrete movement, and this
  applies the same formula over a continuous rolling window instead.
- **Tremor detection is scoped to 3–12 Hz** and the signal is Butterworth
  low-pass filtered before the FFT, so ordinary hand movement and
  camera/tracking jitter aren't misread as tremor.
- **Finger independence** uses real correlation between fingers' extension
  time series; fingers with near-zero movement variance (e.g. limited
  post-stroke mobility) are excluded from the correlation rather than
  corrupting the whole score with NaN — this was a real bug caught during
  integration testing, not a hypothetical edge case.

## Roadmap (explicitly out of scope for this build)

- Depth camera input (Intel RealSense / OAK-D) for 3D-accurate landmarks
- EMG sensor fusion alongside vision
- Clinician-facing dashboard (multi-patient, multi-session comparison view)
- Structured multi-exercise protocols beyond the single open/close cycle
  and reaction-time cue currently implemented

## Testing without a camera

```bash
python tests/test_gestures.py
python tests/test_biomechanics.py
python tests/test_signal_processing.py
python tests/test_audio.py
python tests/test_session.py
python tests/test_integration.py
python tests/test_main_dryrun.py     # exercises main.py's actual code, not just modules
```

## License

MIT — use freely, attribution appreciated.
