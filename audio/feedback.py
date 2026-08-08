"""
Continuous tone feedback: pitch decreases as the hand opens (per spec),
via a live sine-wave audio stream whose frequency updates in real time
from a shared value the main loop writes to every frame.

This module could not be tested against real speaker output in the
development sandbox (no audio hardware there) — the waveform-generation
math is unit-tested against its own FFT in tests/test_audio.py, but the
actual sound needs to be verified on your machine. If no audio device is
available, this fails silently rather than crashing the app — vision and
metrics keep working either way.
"""

import threading

import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    _SOUNDDEVICE_AVAILABLE = False

SAMPLE_RATE = 44100
MIN_FREQ_HZ = 220.0   # hand fully open -> lower pitch (per spec: pitch decreases as hand opens)
MAX_FREQ_HZ = 880.0   # hand fully closed -> higher pitch


def openness_to_frequency(openness):
    """openness in [0,1] -> frequency in Hz. Inverted: more open = lower pitch."""
    openness = max(0.0, min(1.0, openness))
    return MAX_FREQ_HZ - openness * (MAX_FREQ_HZ - MIN_FREQ_HZ)


def generate_sine_block(frequency_hz, n_samples, phase, sample_rate=SAMPLE_RATE, amplitude=0.2):
    """
    Generates n_samples of a sine wave continuing from `phase`, returning
    (samples, new_phase) so consecutive blocks stay phase-continuous
    (avoids audible clicks between callback blocks).
    """
    t = np.arange(n_samples) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * frequency_hz * t + phase)
    new_phase = (phase + 2 * np.pi * frequency_hz * n_samples / sample_rate) % (2 * np.pi)
    return samples.astype(np.float32), new_phase


class ToneFeedback:
    """
    Background audio stream producing a continuous tone whose pitch is
    updated live via set_openness(). Degrades to a harmless no-op if no
    audio device is available or the stream fails to start.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._openness = 0.0
        self._phase = 0.0
        self._stream = None
        self._enabled = False

    def start(self):
        if not _SOUNDDEVICE_AVAILABLE:
            print("[audio] sounddevice not available - continuing without audio feedback.")
            return False
        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, callback=self._callback, blocksize=1024,
            )
            self._stream.start()
            self._enabled = True
            return True
        except Exception as e:
            print(f"[audio] could not start audio feedback ({e}) - continuing without sound.")
            self._enabled = False
            return False

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            freq = openness_to_frequency(self._openness)
            phase = self._phase
        samples, new_phase = generate_sine_block(freq, frames, phase)
        with self._lock:
            self._phase = new_phase
        outdata[:, 0] = samples

    def set_openness(self, openness):
        with self._lock:
            self._openness = openness

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._enabled = False

    @property
    def enabled(self):
        return self._enabled
