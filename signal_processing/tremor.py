"""
Real tremor detection via FFT on a filtered, real (timestamp, x, y)
trajectory buffer. No simulated values — reports None until enough real
time has been buffered.

Detection requires a SUSTAINED pattern across several consecutive
analyze() calls, not a single lucky FFT spike. A single noisy frame
crossing the amplitude threshold is common even for a genuinely still
hand (ordinary camera/tracking jitter) — requiring confirm_frames
consecutive raw detections before reporting "detected" filters that out
while still catching real, sustained tremor (analyze() is called every
frame on a sliding, heavily-overlapping buffer in normal use, so a real
tremor keeps re-triggering the raw check frame after frame).
"""

from collections import deque

import numpy as np

from signal_processing.filters import lowpass_filter

# Typical physiological tremor range (Parkinsonian rest tremor ~4-6 Hz,
# essential tremor ~4-12 Hz). Restricting the FFT search to this band
# keeps normal voluntary movement (usually <2 Hz) from being misread.
TREMOR_BAND_HZ = (3.0, 12.0)
FILTER_CUTOFF_HZ = 15.0  # removes high-frequency tracking jitter, preserves tremor band


class TremorAnalyzer:
    def __init__(self, buffer_seconds=3.0, max_buffer=180, amplitude_threshold=1.0, confirm_frames=5):
        self._buffer = deque(maxlen=max_buffer)
        self._buffer_seconds = buffer_seconds
        self._amplitude_threshold = amplitude_threshold
        self._confirm_frames = confirm_frames
        self._consecutive_raw = 0

    def update(self, fingertip_xy, timestamp):
        self._buffer.append((timestamp, fingertip_xy[0], fingertip_xy[1]))
        while self._buffer and timestamp - self._buffer[0][0] > self._buffer_seconds:
            self._buffer.popleft()

    def analyze(self):
        if len(self._buffer) < 20:
            return None

        times = np.array([b[0] for b in self._buffer], dtype=float)
        xs = np.array([b[1] for b in self._buffer], dtype=float)
        duration = times[-1] - times[0]
        if duration < 1.0:
            return None

        fs = len(times) / duration
        uniform_t = np.linspace(times[0], times[-1], len(times))
        xs_uniform = np.interp(uniform_t, times, xs)
        xs_uniform = lowpass_filter(xs_uniform, FILTER_CUTOFF_HZ, fs)
        xs_detrended = xs_uniform - np.mean(xs_uniform)

        fft_vals = np.fft.rfft(xs_detrended)
        freqs = np.fft.rfftfreq(len(xs_detrended), d=1.0 / fs)
        magnitude = np.abs(fft_vals)

        band_mask = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
        if not np.any(band_mask):
            self._consecutive_raw = 0
            return {"frequency_hz": 0.0, "amplitude": 0.0, "detected": False}

        band_freqs = freqs[band_mask]
        band_mag = magnitude[band_mask]
        peak_idx = int(np.argmax(band_mag))
        peak_freq = float(band_freqs[peak_idx])
        peak_amp = float(band_mag[peak_idx]) / len(xs_detrended)

        raw_detected = peak_amp > self._amplitude_threshold
        self._consecutive_raw = self._consecutive_raw + 1 if raw_detected else 0
        confirmed = self._consecutive_raw >= self._confirm_frames

        return {
            "frequency_hz": round(peak_freq, 1),
            "amplitude": round(peak_amp, 2),
            "detected": confirmed,
        }
