"""
Real reaction-time measurement: a cue is issued at a real timestamp, and
the elapsed time is measured against the real timestamp the response
condition (e.g. open palm detected) is first met. No fabricated values —
returns None until a cue has actually been issued and answered.
"""


class ReactionTimer:
    def __init__(self):
        self._cue_time = None
        self.last_reaction_time_s = None
        self.history = []

    def issue_cue(self, timestamp):
        self._cue_time = timestamp
        self.last_reaction_time_s = None

    def register_response(self, timestamp):
        """Call when the target gesture/condition is first detected. Only
        registers if a cue is currently pending (ignores spontaneous gestures)."""
        if self._cue_time is None:
            return None
        elapsed = timestamp - self._cue_time
        self.last_reaction_time_s = round(elapsed, 3)
        self.history.append(self.last_reaction_time_s)
        self._cue_time = None
        return self.last_reaction_time_s

    def is_waiting_for_response(self):
        return self._cue_time is not None
