"""Hand skeleton with a glow pass, colored per-finger by real extension ratio."""

import cv2
import numpy as np

from vision.hand_tracker import HAND_CONNECTIONS


def ratio_to_color(ratio):
    """0-1 -> red -> yellow -> green BGR (OpenCV hue 0=red, ~30=yellow, ~60=green)."""
    ratio = max(0.0, min(1.0, ratio))
    hue = int(ratio * 60)
    hsv_pixel = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def _connection_finger_index(a, b):
    for idx in (a, b):
        if 1 <= idx <= 4:
            return 0
        if 5 <= idx <= 8:
            return 1
        if 9 <= idx <= 12:
            return 2
        if 13 <= idx <= 16:
            return 3
        if 17 <= idx <= 20:
            return 4
    return None


def draw_glow_skeleton(frame, landmarks_px, extension_ratios=None):
    glow = frame.copy()
    for a, b in HAND_CONNECTIONS:
        pa, pb = landmarks_px[a][:2], landmarks_px[b][:2]
        finger_idx = _connection_finger_index(a, b)
        color = ratio_to_color(extension_ratios[finger_idx]) if (extension_ratios and finger_idx is not None) else (255, 255, 255)
        cv2.line(glow, pa, pb, color, 9, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.22, frame, 0.78, 0, dst=frame)

    for a, b in HAND_CONNECTIONS:
        pa, pb = landmarks_px[a][:2], landmarks_px[b][:2]
        finger_idx = _connection_finger_index(a, b)
        color = ratio_to_color(extension_ratios[finger_idx]) if (extension_ratios and finger_idx is not None) else (230, 230, 230)
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    for x, y, _ in landmarks_px:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 190, 60), 1, cv2.LINE_AA)
