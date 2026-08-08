"""
Counts real open-close repetitions from actual sphere state transitions —
with debounce. Real hand-tracking data is noisy: if openness hovers near
the open/close threshold, raw landmark jitter can flicker across it
several times a second. Without debounce, each flicker would count as a
full repetition (this was a real bug, caught from an actual run showing
79 "reps" in under 4 minutes). A candidate state now has to be held for a
minimum real duration before it's accepted as a genuine transition.
"""


class RepetitionTracker:
    def __init__(self, open_hold_s=0.15, close_hold_s=0.20):
        self.count = 0
        self._state = "closed"
        self._candidate = None
        self._candidate_since = None
        self._rep_start_time = None
        self._rep_peak_openness = 0.0
        self._rep_peak_velocity = 0.0
        self.completed_reps = []  # list of dicts: {duration, peak_openness, peak_velocity}
        self._open_hold_s = open_hold_s
        self._close_hold_s = close_hold_s

    def update(self, is_open, is_closed, timestamp, openness=0.0, velocity_px_s=0.0):
        """
        Call every frame with is_open/is_closed (from openness crossing
        the two separate thresholds — leaving a dead zone between them
        that never counts as a candidate transition on its own) plus the
        current openness/velocity, so completed reps record real peaks.
        """
        target = "open" if is_open else ("closed" if is_closed else None)

        if target is None or target == self._state:
            self._candidate = None
        else:
            if target != self._candidate:
                self._candidate = target
                self._candidate_since = timestamp
            hold_needed = self._open_hold_s if target == "open" else self._close_hold_s
            if timestamp - self._candidate_since >= hold_needed:
                self._transition(target, timestamp)
                self._candidate = None

        if self._state == "open":
            self._rep_peak_openness = max(self._rep_peak_openness, openness)
            self._rep_peak_velocity = max(self._rep_peak_velocity, velocity_px_s or 0.0)

    def _transition(self, new_state, timestamp):
        if new_state == "open" and self._state == "closed":
            self._state = "open"
            self._rep_start_time = timestamp
            self._rep_peak_openness = 0.0
            self._rep_peak_velocity = 0.0
        elif new_state == "closed" and self._state == "open":
            duration = (timestamp - self._rep_start_time) if self._rep_start_time is not None else 0.0
            self.completed_reps.append({
                "duration_s": round(duration, 2),
                "peak_openness": round(self._rep_peak_openness, 2),
                "peak_velocity_px_s": round(self._rep_peak_velocity, 1),
            })
            self.count += 1
            self._state = "closed"
