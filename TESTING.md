# Testing & Validation

This document explains what has actually been verified in MediSphere, how, and against what benchmark — as opposed to what has *not* been verified. The goal is to be precise about the difference between "the code runs" and "the code is correct."

## Philosophy

Every computational module (biomechanics, kinematics, signal processing, session logic) is validated against **synthetic inputs with a known, independently-calculated correct answer** — not just "does it run without crashing." A test that only checks a function doesn't throw an exception proves far less than a test that feeds in a signal with a known 6 Hz oscillation and checks the output actually says 6 Hz.

## Current status

**57 / 57 automated tests passing**, reproducible with:

```bash
pip install pytest
python -m pytest tests/ -v
```

or, without pytest, by running any test file directly (each is self-contained):

```bash
python tests/test_biomechanics.py
```

## Test inventory and benchmark used per module

### `tests/test_gestures.py` — 8 tests
**Benchmark:** hand-constructed synthetic landmark coordinates with an explicit, known geometry (e.g., fingertips deliberately placed closer to the wrist than the mid-joint, to represent "curled").
Validates: open palm / closed fist / pinch / point / swipe detection.

### `tests/test_biomechanics.py` — 9 tests
**Benchmark:** synthetic landmarks with hand-calculated correct angles.
- Colinear points → joint angle must read ≈180°
- A deliberate right-angle bend → must read ≈90°
- Identical values fed to all 5 fingers → correlation-based independence score must be low
- Independent random values per finger → independence score must be high
- A point moved at an exact, arithmetic 100 px/s → measured velocity must match ≈100 px/s
- A constant-velocity path vs. a randomly-perturbed path of the same route → smoothness score must rank the constant path higher

### `tests/test_signal_processing.py` — 5 tests
**Benchmark:** signals with a known, deliberately injected frequency.
- A synthetic 6 Hz oscillation injected into fingertip position → detector must recover ≈6 Hz
- A 25 Hz "noise" component injected on top of a 2 Hz "real" component → low-pass filter must suppress the 25 Hz energy
- A perfectly static signal → must report no tremor
- 4 seconds of pure random jitter (no periodicity at all) → debounce logic must prevent a false "detected"

### `tests/test_audio.py` — 3 tests
**Benchmark:** the generated waveform checked against itself via independent FFT analysis, not just trusting the generating formula.
- Requested a 440 Hz tone, ran FFT on the actual output samples, confirmed the spectral peak is ≈440 Hz
- Checked no phase discontinuity ("click") between consecutive audio buffer chunks

### `tests/test_session.py` — 12 tests
**Benchmark:** hand-constructed timestamp/state sequences with a known correct outcome.
- 200 frames of synthetic rapid flicker across the open/close threshold → must count **zero** repetitions
- One deliberate, sustained open→close cycle → must count exactly **one**
- Data written to SQLite + CSV, then read back → must match exactly what was written (round-trip test)
- A declining synthetic sequence vs. a flat synthetic sequence → fatigue trend must fire only on the declining case

### `tests/test_clinical_flags.py` — 17 tests
**Benchmark:** values placed exactly at, just above, and just below every documented threshold, to confirm boundaries behave correctly (not just "roughly right").

### `tests/test_integration.py` — 2 tests
**Benchmark:** a complete simulated session (multiple repetitions, injected tremor) run through every module together, checking the whole pipeline agrees, not just each piece in isolation.

### `tests/test_main_dryrun.py` — 1 test
**Benchmark:** `main.py`'s own actual code, executed end-to-end, with only the camera and MediaPipe calls substituted for deterministic synthetic input. This is the closest proof available, without a physical camera, that the *real* application — not a reimplementation of its logic — runs correctly.

## Real bugs this process actually caught

Two defects were found during live webcam testing (not by the automated suite, since neither was a *logic* bug — the code worked exactly as written, the design was wrong):

1. **Visibility** — the biofeedback sphere/hologram was tuned against a flat test background and was nearly invisible against a real, cluttered room. Fixed by adding a darkened backdrop and increasing opacity; verified by re-rendering against a synthetic high-contrast background.
2. **Repetition count inflation** — an early version registered 79 repetitions in under 4 minutes of live use, versus a much smaller number of deliberate hand-openings. Traced to landmark-tracking jitter flickering across the open/close threshold. Fixed by requiring a sustained minimum hold duration before a transition counts, and a dedicated regression test (`test_repetition_tracker_ignores_brief_threshold_jitter`) was added specifically to prevent this from recurring silently.

## What is explicitly NOT validated

- **No clinical validation.** The composite Motor Score and every clinical flag threshold are heuristics defined for this project, motivated by the general direction of findings in rehabilitation literature — not derived from patient data, not compared against a validated instrument like the Fugl-Meyer Assessment.
- **Tremor/stability/smoothness thresholds are heuristic**, tuned by reasoning and spot-checking, not calibrated against a labeled dataset of real tremor or real patient movement.
- **3D hologram rendering** was verified end-to-end using a software OpenGL renderer (Mesa llvmpipe) in a Linux test environment; behavior against a real GPU driver is separately confirmed only by direct use, not by the automated suite.
- **Audio playback** — the waveform math is verified by FFT; actual sound through real speakers can only be confirmed by listening.

## Reproducing this yourself

```bash
git clone https://github.com/Amyari007/medisphere.git
cd medisphere
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v
```

You should see `57 passed`. If any test fails, that's a genuine regression worth investigating, not a flaky test — none of these tests are timing-sensitive or dependent on external state.
