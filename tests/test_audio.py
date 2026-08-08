import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from audio.feedback import openness_to_frequency, generate_sine_block, MIN_FREQ_HZ, MAX_FREQ_HZ


def test_pitch_decreases_as_hand_opens():
    freq_closed = openness_to_frequency(0.0)
    freq_open = openness_to_frequency(1.0)
    assert freq_closed == MAX_FREQ_HZ, f"closed hand should map to the higher pitch, got {freq_closed}"
    assert freq_open == MIN_FREQ_HZ, f"open hand should map to the lower pitch, got {freq_open}"
    freq_mid = openness_to_frequency(0.5)
    assert freq_closed > freq_mid > freq_open, (
        f"pitch should decrease as hand opens: closed={freq_closed}, mid={freq_mid}, open={freq_open}"
    )


def test_generated_waveform_actually_contains_target_frequency():
    """Don't just trust the formula - actually FFT the generated samples
    and confirm the dominant frequency matches what we asked for."""
    sample_rate = 44100
    target_freq = 440.0
    n_samples = sample_rate  # 1 second, gives clean 1Hz FFT resolution
    samples, _ = generate_sine_block(target_freq, n_samples, phase=0.0, sample_rate=sample_rate)

    fft_vals = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    peak_freq = freqs[np.argmax(np.abs(fft_vals))]

    assert abs(peak_freq - target_freq) < 2.0, f"expected ~{target_freq}Hz in the generated audio, got {peak_freq}Hz"


def test_consecutive_blocks_are_phase_continuous():
    """Verify no discontinuity (click) between two consecutive generated blocks."""
    sample_rate = 44100
    freq = 440.0
    block1, phase_after = generate_sine_block(freq, 512, phase=0.0, sample_rate=sample_rate)
    block2, _ = generate_sine_block(freq, 512, phase=phase_after, sample_rate=sample_rate)

    # the last sample of block1 and first sample of block2 should be close
    # (continuous waveform), not an arbitrary jump
    expected_next = block1[-1] + (block1[-1] - block1[-2])  # rough local slope continuation
    assert abs(block2[0] - block1[-1]) < 0.05, (
        f"discontinuity between blocks: {block1[-1]} -> {block2[0]}"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
