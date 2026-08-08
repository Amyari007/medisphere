"""
Real joint-level biomechanics from tracked hand landmarks — no placeholder
or simulated values anywhere in this module.

Per-finger joint chains use MediaPipe's 21-point layout. Each finger is a
5-point chain (wrist -> base joint -> mid joint -> distal joint -> tip);
we compute the geometric angle at each of the three interior joints.
Naming follows standard hand anatomy for the four fingers (MCP/PIP/DIP);
the thumb's equivalent joints are CMC/MCP/IP, but we expose them under
the same joint1/joint2/joint3 keys for a uniform API — see FINGER_CHAINS.
"""

import math

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# (wrist, joint1_landmark, joint2_landmark, joint3_landmark, tip_landmark)
# joint1=CMC/MCP, joint2=MCP/PIP, joint3=IP/DIP depending on thumb vs finger.
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

# Anatomical labels per finger for display purposes only (thumb differs).
JOINT_LABELS = {
    "thumb": ("CMC", "MCP", "IP"),
    "index": ("MCP", "PIP", "DIP"),
    "middle": ("MCP", "PIP", "DIP"),
    "ring": ("MCP", "PIP", "DIP"),
    "pinky": ("MCP", "PIP", "DIP"),
}


def _angle_at(p_prev, p_joint, p_next):
    """Geometric angle at p_joint between rays to p_prev and p_next, in degrees.
    ~180 = straight through the joint, smaller = more flexed."""
    v1 = (p_prev[0] - p_joint[0], p_prev[1] - p_joint[1])
    v2 = (p_next[0] - p_joint[0], p_next[1] - p_joint[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def finger_joint_angles(landmarks, finger_name):
    """Returns {'joint1': deg, 'joint2': deg, 'joint3': deg} for one finger."""
    chain = FINGER_CHAINS[finger_name]
    p = [landmarks[i][:2] for i in chain]
    return {
        "joint1": _angle_at(p[0], p[1], p[2]),
        "joint2": _angle_at(p[1], p[2], p[3]),
        "joint3": _angle_at(p[2], p[3], p[4]),
    }


def all_finger_joint_angles(landmarks):
    """Returns {finger_name: {'joint1':..,'joint2':..,'joint3':..}} for all 5 fingers."""
    return {name: finger_joint_angles(landmarks, name) for name in FINGER_NAMES}


def hand_openness(landmarks):
    """
    Overall hand openness in [0, 1], averaged across all 5 fingers' joint2
    (PIP-equivalent) angles, normalized against an empirical curled/straight
    range. Used to drive the sphere size and audio pitch continuously,
    rather than the binary open/closed gesture check.
    """
    angles = [finger_joint_angles(landmarks, name)["joint2"] for name in FINGER_NAMES]
    avg = sum(angles) / len(angles)
    # empirical range: ~60 deg curled, ~180 deg straight
    ratio = (avg - 60.0) / (180.0 - 60.0)
    return max(0.0, min(1.0, ratio))


class ROMTracker:
    """
    Tracks observed range of motion (max - min joint2 angle) per finger
    across a session, from real per-frame joint-angle measurements.
    """

    def __init__(self):
        self._min = {name: 180.0 for name in FINGER_NAMES}
        self._max = {name: 0.0 for name in FINGER_NAMES}
        self._seen = False

    def update(self, landmarks):
        self._seen = True
        for name in FINGER_NAMES:
            angle = finger_joint_angles(landmarks, name)["joint2"]
            self._min[name] = min(self._min[name], angle)
            self._max[name] = max(self._max[name], angle)

    def rom_degrees(self):
        """{finger_name: rom_degrees}, or None if no data yet."""
        if not self._seen:
            return None
        return {name: round(self._max[name] - self._min[name], 1) for name in FINGER_NAMES}


class FingerIndependenceTracker:
    """
    Measures how independently fingers move from each other, using real
    correlation between each finger's extension time series over a
    rolling window. High correlation (fingers moving in lockstep, a
    common post-stroke pattern) -> low independence score. Low
    correlation (fingers moving separately) -> high independence score.
    """

    def __init__(self, window=60):
        self._window = window
        self._buffers = {name: [] for name in FINGER_NAMES}

    def update(self, extension_ratios):
        """extension_ratios: 5 floats [thumb, index, middle, ring, pinky] in [0,1]."""
        for name, value in zip(FINGER_NAMES, extension_ratios):
            buf = self._buffers[name]
            buf.append(value)
            if len(buf) > self._window:
                buf.pop(0)

    def score(self):
        """Returns independence score 0-100, or None if not enough data yet."""
        import numpy as np

        if any(len(buf) < 10 for buf in self._buffers.values()):
            return None

        data = np.array([self._buffers[name] for name in FINGER_NAMES])
        stds = data.std(axis=1)

        # Fingers with ~zero variance (e.g. a digit with very limited
        # mobility) can't meaningfully correlate with anything and would
        # otherwise poison the whole matrix with NaN — exclude them and
        # compute independence only across fingers that actually moved.
        varying_mask = stds > 1e-4
        n_varying = int(np.sum(varying_mask))
        if n_varying == 0:
            return 100.0  # nothing moved at all -> trivially "independent"
        if n_varying == 1:
            return None  # can't correlate a single signal against itself

        varying_data = data[varying_mask]
        corr = np.corrcoef(varying_data)
        n = corr.shape[0]
        off_diag = corr[~np.eye(n, dtype=bool)]
        off_diag = off_diag[~np.isnan(off_diag)]
        if len(off_diag) == 0:
            return None
        avg_corr = float(np.mean(np.abs(off_diag)))
        independence = (1.0 - avg_corr) * 100.0
        return max(0.0, min(100.0, independence))
