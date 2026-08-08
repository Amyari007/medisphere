"""
Detects a real declining trend across repetitions (a common fatigue
signature: peak openness or velocity dropping rep-over-rep) using linear
regression on the actual per-rep values collected during the session —
not a scripted or simulated trend.
"""

import numpy as np


class FatigueTracker:
    def __init__(self, min_reps=4):
        self._min_reps = min_reps

    def analyze(self, completed_reps, key="peak_openness"):
        """
        completed_reps: list of per-rep dicts from RepetitionTracker.
        Returns dict(slope, declining, values) or None if too few reps yet.
        """
        if len(completed_reps) < self._min_reps:
            return None
        values = np.array([r[key] for r in completed_reps], dtype=float)
        rep_indices = np.arange(len(values))
        slope, intercept = np.polyfit(rep_indices, values, 1)
        # normalize slope relative to the mean value so the "declining"
        # threshold isn't sensitive to the metric's absolute scale
        mean_val = float(np.mean(values)) or 1e-6
        relative_slope = slope / mean_val
        declining = relative_slope < -0.03  # >3% relative drop per rep, heuristic threshold
        return {
            "slope": float(slope),
            "relative_slope": float(relative_slope),
            "declining": bool(declining),
            "values": values.tolist(),
        }
