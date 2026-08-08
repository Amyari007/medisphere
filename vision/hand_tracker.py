"""
Hand landmark tracking using MediaPipe's Tasks API.

Note: recent MediaPipe Python wheels (0.10.x on newer Python builds, and
all 1.x releases) no longer ship the old `mp.solutions.hands` API — only
the newer Tasks API remains. This wrapper targets that Tasks API and
needs a `hand_landmarker.task` model file on disk (see README setup).
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Standard 21-point hand connections, for manual skeleton drawing —
# mp.solutions.drawing_utils doesn't exist under the Tasks API either.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)


class HandTracker:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_hands: int = 1,
                 detection_conf: float = 0.6, tracking_conf: float = 0.5):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at:\n  {model_path}\n\n"
                "Download it (one-time, ~7.5 MB) with:\n\n"
                "  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task\n\n"
                "PowerShell equivalent:\n"
                "  curl.exe -L -o models\\hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task\n\n"
                "See README.md 'Setup' section for details."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, frame_bgr):
        """Runs detection on a BGR frame, returns a HandLandmarkerResult.
        Access detected hands via `result.hand_landmarks` (list of hands,
        each a list of 21 NormalizedLandmark objects)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def landmarks_to_pixels(hand_landmarks, frame_width, frame_height):
        """hand_landmarks: one detected hand's list of NormalizedLandmark."""
        return [
            (int(lm.x * frame_width), int(lm.y * frame_height), lm.z)
            for lm in hand_landmarks
        ]

    @staticmethod
    def draw_skeleton(frame_bgr, landmarks_px):
        """landmarks_px: output of landmarks_to_pixels (already pixel-space)."""
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, landmarks_px[a][:2], landmarks_px[b][:2], (0, 200, 0), 2, cv2.LINE_AA)
        for x, y, _ in landmarks_px:
            cv2.circle(frame_bgr, (x, y), 4, (0, 140, 255), -1, cv2.LINE_AA)

    def close(self):
        self._landmarker.close()
