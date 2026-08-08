import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from signal_processing.filters import lowpass_filter
from signal_processing.tremor import TremorAnalyzer
from signal_processing.stability import StabilityTracker


def test_lowpass_filter_removes_high_frequency_noise():
    fs = 60.0
    t = np.arange(0, 3.0, 1.0 / fs)
    low = np.sin(2 * np.pi * 2.0 * t)          # 2 Hz signal we want to keep
    high = 0.8 * np.sin(2 * np.pi * 25.0 * t)  # 25 Hz noise we want removed
    combined = low + high

    filtered = lowpass_filter(combined, cutoff_hz=15.0, fs=fs)

    # high-frequency energy should drop substantially after filtering
    residual_high_energy = np.sum((filtered - low) ** 2)
    raw_high_energy = np.sum((combined - low) ** 2)
    assert residual_high_energy < raw_high_energy * 0.3, (
        f"filter should suppress 25Hz noise; residual={residual_high_energy:.2f} "
        f"vs raw={raw_high_energy:.2f}"
    )


def test_tremor_detects_known_frequency_after_filtering():
    analyzer = TremorAnalyzer(buffer_seconds=3.0)
    fs = 30.0
    target_hz = 6.0
    n_samples = int(fs * 4.0)  # longer run so the debounce window has time to confirm
    result = None
    for i in range(n_samples):
        t = i / fs
        x = 300.0 + 10.0 * math.sin(2 * math.pi * target_hz * t)
        analyzer.update((x, 300.0), t)
        result = analyzer.analyze()
    assert result is not None
    assert result["detected"], f"expected sustained tremor detected, got {result}"
    assert abs(result["frequency_hz"] - target_hz) < 1.0, f"expected ~{target_hz}Hz, got {result}"


def test_tremor_still_ignores_still_hand_after_filtering():
    analyzer = TremorAnalyzer(buffer_seconds=3.0)
    fs = 30.0
    result = None
    for i in range(int(fs * 4.0)):
        t = i / fs
        analyzer.update((300.0, 300.0), t)
        result = analyzer.analyze()
    assert result is not None and not result["detected"]


def test_tremor_ignores_a_single_noisy_frame_spike():
    """A single lucky FFT spike from ordinary tracking jitter shouldn't
    flip 'detected' to True - only a SUSTAINED pattern should."""
    import random
    random.seed(7)
    analyzer = TremorAnalyzer(buffer_seconds=3.0)
    fs = 30.0
    results = []
    for i in range(int(fs * 4.0)):
        t = i / fs
        # small random per-frame jitter, no real periodic tremor content
        x = 300.0 + random.uniform(-1.5, 1.5)
        analyzer.update((x, 300.0), t)
        r = analyzer.analyze()
        if r is not None:
            results.append(r["detected"])
    # even if a handful of individual FFT windows spike above threshold by
    # chance, the debounced 'detected' flag should never latch True across
    # this much random, non-periodic jitter
    assert not any(results), f"random jitter should never trigger sustained detection, got {sum(results)} True frames"


def test_stability_score_sane_range():
    tracker = StabilityTracker()
    t = 0.0
    for _ in range(30):
        tracker.update((300.0, 300.0), t)
        t += 0.033
    score, jitter = tracker.score()
    assert 0 <= score <= 100
    assert jitter < 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
