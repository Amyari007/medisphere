"""
Gesture recognition from 21-point MediaPipe hand landmarks.

Deliberately avoids a trained gesture classifier — everything here is
geometric (distances/ratios between landmarks). That keeps it fast,
dependency-free, and easy to tune for a demo, at the cost of being less
robust than a learned model. Swap in a trained classifier under
`Level 3+` if you want to extend this later.
"""

import math
from collections import deque

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_extended(landmarks):
    """
    Returns a 5-item bool list [thumb, index, middle, ring, pinky].
    A finger counts as extended if its tip sits farther from the wrist
    than its PIP joint does — this works regardless of hand rotation,
    unlike a plain y-coordinate comparison.
    """
    wrist = landmarks[WRIST]
    extended = []
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip_dist = _dist(landmarks[tip_idx], wrist)
        pip_dist = _dist(landmarks[pip_idx], wrist)
        extended.append(tip_dist > pip_dist * 1.15)
    return extended


def is_open_palm(landmarks):
    return sum(fingers_extended(landmarks)) >= 4


def is_closed_fist(landmarks):
    return sum(fingers_extended(landmarks)) == 0


def is_point_finger(landmarks):
    ext = fingers_extended(landmarks)
    # index extended, the rest curled
    return ext[1] and not any([ext[0], ext[2], ext[3], ext[4]])


def pinch_distance(landmarks):
    """Euclidean distance between thumb tip and index tip, in pixels."""
    return _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])


def is_pinching(landmarks, threshold_px=45):
    return pinch_distance(landmarks) < threshold_px


def palm_center(landmarks):
    """Rough palm centroid, used to anchor the sphere overlay."""
    pts = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[PINKY_MCP], landmarks[MIDDLE_MCP]]
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return int(x), int(y)


class SwipeDetector:
    """
    Tracks palm-center x position over a short rolling window and fires
    a swipe event once cumulative horizontal motion crosses a threshold.
    Has a cooldown so one swipe doesn't fire multiple times.
    """

    def __init__(self, window=6, threshold_px=90, cooldown_frames=15):
        self._history = deque(maxlen=window)
        self._threshold = threshold_px
        self._cooldown = cooldown_frames
        self._cooldown_left = 0

    def update(self, x_pos):
        self._history.append(x_pos)
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return None

        if len(self._history) < self._history.maxlen:
            return None

        delta = self._history[-1] - self._history[0]
        if delta > self._threshold:
            self._cooldown_left = self._cooldown
            self._history.clear()
            return "right"
        elif delta < -self._threshold:
            self._cooldown_left = self._cooldown
            self._history.clear()
            return "left"
        return None


def _clamp01(x):
    return max(0.0, min(1.0, x))


def finger_extension_ratios(landmarks):
    """
    Continuous per-finger extension in [0, 1] — 0 fully curled, 1 fully
    extended — for the 5 fingers [thumb, index, middle, ring, pinky].
    Same tip/MCP distance idea as the boolean gesture checks above, but
    kept continuous (not thresholded) so it can drive a percentage display
    or a heat-map color instead of just a yes/no.
    """
    wrist = landmarks[WRIST]
    ratios = []
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        tip_dist = _dist(landmarks[tip_idx], wrist)
        mcp_dist = _dist(landmarks[mcp_idx], wrist)
        if mcp_dist < 1e-6:
            ratios.append(0.0)
            continue
        r = (tip_dist - mcp_dist * 0.9) / (mcp_dist * 1.4)
        ratios.append(_clamp01(r))
    return ratios


def joint_flexion_angle(landmarks, mcp_idx, pip_idx, tip_idx):
    """
    Real joint angle in degrees at the PIP joint, from the angle between
    vectors (PIP->MCP) and (PIP->TIP). ~180 degrees = finger straight,
    smaller angles = more flexed. Standard vector-geometry approach for
    estimating finger joint angle from 2D keypoints.
    """
    p_mcp = landmarks[mcp_idx][:2]
    p_pip = landmarks[pip_idx][:2]
    p_tip = landmarks[tip_idx][:2]
    v1 = (p_mcp[0] - p_pip[0], p_mcp[1] - p_pip[1])
    v2 = (p_tip[0] - p_pip[0], p_tip[1] - p_pip[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))
