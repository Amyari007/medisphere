"""
Real signal filtering for landmark trajectories, using a standard
Butterworth low-pass filter (scipy.signal) applied before frequency
analysis — this removes high-frequency camera/tracking noise while
preserving genuine motion content, including the tremor band.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filter(signal, cutoff_hz, fs, order=4):
    """
    Zero-phase Butterworth low-pass filter. Returns the input unchanged
    if there isn't enough data for stable filtering (filtfilt needs a
    minimum number of samples relative to the filter order).
    """
    signal = np.asarray(signal, dtype=float)
    min_len = 3 * (order + 1)
    if len(signal) < min_len:
        return signal
    nyquist = fs / 2.0
    normal_cutoff = min(0.99, cutoff_hz / nyquist)
    if normal_cutoff <= 0:
        return signal
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)
