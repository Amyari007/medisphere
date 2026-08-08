"""Real hand-stability score from positional jitter over a rolling time window."""

from collections import deque

import numpy as np


class StabilityTracker:
    def __init__(self, window_seconds=1.5, max_buffer=90):
        self._positions = deque(maxlen=max_buffer)
        self._window_seconds = window_seconds

    def update(self, palm_xy, timestamp):
        self._positions.append((timestamp, palm_xy[0], palm_xy[1]))
        while self._positions and timestamp - self._positions[0][0] > self._window_seconds:
            self._positions.popleft()

    def score(self):
        """Returns (stability_0_100, raw_jitter_px), or (None, None) if not enough data yet."""
        if len(self._positions) < 5:
            return None, None
        xs = np.array([p[1] for p in self._positions], dtype=float)
        ys = np.array([p[2] for p in self._positions], dtype=float)
        jitter_px = float(np.sqrt(xs.std() ** 2 + ys.std() ** 2))
        score = max(0.0, min(100.0, 100.0 * (1.0 - jitter_px / 25.0)))
        return score, jitter_px
