"""
Movement kinematics from real (timestamp, x, y) position samples —
velocity, acceleration, and jerk-based smoothness. No simulated values.

The smoothness metric is a rolling-window approximation of Log
Dimensionless Jerk (LDLJ), a standard measure in movement science and
stroke rehabilitation literature (Hogan & Sternad, 2009) for quantifying
how smooth vs. jerky a movement is — higher (less negative) = smoother.
This is an approximation because true LDLJ is defined over a single
discrete movement; here we apply the same formula over a rolling window
of continuous motion, which is a reasonable proxy but not the exact
per-movement definition from the literature.
"""

from collections import deque

import numpy as np


class KinematicsTracker:
    def __init__(self, window_seconds=1.0, max_buffer=120):
        self._buffer = deque(maxlen=max_buffer)  # (t, x, y)
        self._window_seconds = window_seconds

    def update(self, xy, timestamp):
        self._buffer.append((timestamp, xy[0], xy[1]))
        while self._buffer and timestamp - self._buffer[0][0] > self._window_seconds:
            self._buffer.popleft()

    def _series(self):
        times = np.array([b[0] for b in self._buffer], dtype=float)
        xs = np.array([b[1] for b in self._buffer], dtype=float)
        ys = np.array([b[2] for b in self._buffer], dtype=float)
        return times, xs, ys

    def velocity_px_s(self):
        """Instantaneous speed (px/s) from the two most recent samples, or None."""
        if len(self._buffer) < 2:
            return None
        t1, x1, y1 = self._buffer[-2]
        t2, x2, y2 = self._buffer[-1]
        dt = t2 - t1
        if dt <= 0:
            return None
        return float(np.hypot(x2 - x1, y2 - y1) / dt)

    def snapshot(self):
        """
        Returns dict(velocity_px_s, acceleration_px_s2, smoothness) or
        None fields where there isn't yet enough data for that derivative
        (velocity needs 2 points, acceleration needs 3, smoothness/jerk
        needs a real window of samples).
        """
        if len(self._buffer) < 2:
            return {"velocity_px_s": None, "acceleration_px_s2": None, "smoothness": None}

        times, xs, ys = self._series()
        if len(times) < 3:
            return {"velocity_px_s": self.velocity_px_s(), "acceleration_px_s2": None, "smoothness": None}

        dt = np.diff(times)
        dt[dt <= 0] = 1e-6
        vx = np.diff(xs) / dt
        vy = np.diff(ys) / dt
        speed = np.hypot(vx, vy)

        if len(speed) < 2:
            return {"velocity_px_s": float(speed[-1]), "acceleration_px_s2": None, "smoothness": None}

        dt_v = dt[1:]
        accel = np.diff(speed) / dt_v

        smoothness = None
        if len(accel) >= 2:
            dt_a = dt_v[1:]
            jerk = np.diff(accel) / dt_a
            duration = times[-1] - times[0]
            peak_speed = float(np.max(speed))
            if duration > 0 and peak_speed > 1e-3:
                trapz_fn = getattr(np, "trapezoid", None) or np.trapz
                jerk_sq_integral = float(trapz_fn(jerk ** 2, dx=np.mean(dt_a)))
                inner = (duration ** 5 / (peak_speed ** 2)) * jerk_sq_integral
                # Near-zero jerk (very smooth, e.g. constant velocity) drives
                # `inner` toward 0, which is the *good* end of this metric
                # (-log(small number) is a large positive score) — clamp
                # instead of skipping, so genuinely smooth motion doesn't
                # get reported as "no data" and cap it to a sane range.
                inner = max(inner, 1e-9)
                smoothness = max(-50.0, min(50.0, float(-np.log(inner))))

        return {
            "velocity_px_s": float(speed[-1]),
            "acceleration_px_s2": float(accel[-1]) if len(accel) else None,
            "smoothness": smoothness,
        }
