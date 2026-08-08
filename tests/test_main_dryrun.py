"""
Runs main.py's actual main() function end-to-end with the camera and
MediaPipe detection monkeypatched out, feeding synthetic landmark
sequences instead. This is the closest thing to a real run available
without physical camera hardware, and it exercises main.py's own glue
code directly (key handling, state resets, the hologram/sphere branch,
rep-completion logging) rather than just the underlying modules.
"""
import sys
import os
import math
import shutil
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

import main as app_main
from tests.test_integration import make_hand


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeResult:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class _FakeCapture:
    """Stands in for cv2.VideoCapture, yielding blank frames indefinitely."""

    def __init__(self, *a, **kw):
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self._opened = False


def _make_fake_tracker_process(width, height):
    """Returns a process() replacement cycling through an open->close rep,
    normalized to [0,1] pixel-space landmarks as MediaPipe would provide."""
    state = {"frame": 0}
    cycle_len = 90  # ~3s at 30fps: ramp open, hold, ramp closed

    def process(self_unused, frame_bgr):
        i = state["frame"] % cycle_len
        state["frame"] += 1
        if i < 30:
            openness = i / 30.0
        elif i < 60:
            openness = 1.0
        else:
            openness = max(0.0, 1.0 - (i - 60) / 30.0)

        px_landmarks = make_hand(openness, wrist=(320, 400))  # returns (x,y,z) in pixel space already
        # HandTracker.landmarks_to_pixels expects *normalized* landmarks with
        # .x/.y in [0,1] and multiplies by width/height, so normalize here.
        norm_landmarks = [_FakeLandmark(x / width, y / height, z) for (x, y, z) in px_landmarks]
        return _FakeResult(hand_landmarks=[norm_landmarks])

    return process


def test_main_runs_end_to_end_without_crashing():
    test_dir = "/tmp/medisphere_main_dryrun"
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(test_dir)  # so session_data/ lands somewhere disposable
    os.makedirs("models", exist_ok=True)
    _asset_src = os.path.join(prev_cwd, "models", "test_asset.glb")
    if not os.path.exists(_asset_src):
        import trimesh
        trimesh.creation.icosphere(subdivisions=2, radius=1.0).export(_asset_src)
    shutil.copy(_asset_src, "models/brain_hologram.glb")

    real_video_capture = cv2.VideoCapture
    real_imshow = cv2.imshow
    real_waitkey = cv2.waitKey
    real_destroy = cv2.destroyAllWindows
    real_process = app_main.HandTracker.process
    real_init = app_main.HandTracker.__init__
    real_close = app_main.HandTracker.close

    frame_counter = {"n": 0}
    QUIT_AFTER = 200  # ~a few completed reps worth

    def fake_waitkey(_delay):
        frame_counter["n"] += 1
        if frame_counter["n"] >= QUIT_AFTER:
            return ord('q')
        return 0xFF  # no key pressed (matches "& 0xFF" masking used in main.py)

    try:
        cv2.VideoCapture = _FakeCapture
        cv2.imshow = lambda *a, **kw: None
        cv2.waitKey = fake_waitkey
        cv2.destroyAllWindows = lambda: None
        app_main.HandTracker.__init__ = lambda self, *a, **kw: None
        app_main.HandTracker.process = _make_fake_tracker_process(640, 480)
        app_main.HandTracker.close = lambda self: None

        exit_code = app_main.main(camera_index=0)
        assert exit_code == 0, f"main() should exit cleanly, got {exit_code}"

        # verify a session db + report actually got written by the real logger
        session_dirs = [d for d in os.listdir(test_dir) if d == "session_data"]
        assert session_dirs, f"expected session_data/ to be created, found: {os.listdir(test_dir)}"
        files = os.listdir(os.path.join(test_dir, "session_data"))
        assert any(f.endswith(".db") for f in files), f"expected a .db file, found: {files}"
        assert any(f.endswith("_report.txt") for f in files), f"expected a report file, found: {files}"

    finally:
        cv2.VideoCapture = real_video_capture
        cv2.imshow = real_imshow
        cv2.waitKey = real_waitkey
        cv2.destroyAllWindows = real_destroy
        app_main.HandTracker.process = real_process
        app_main.HandTracker.__init__ = real_init
        app_main.HandTracker.close = real_close
        os.chdir(prev_cwd)
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_main_runs_end_to_end_without_crashing()
    print("PASS  test_main_runs_end_to_end_without_crashing")
