import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from biomechanics.joints import (
    finger_joint_angles,
    all_finger_joint_angles,
    hand_openness,
    ROMTracker,
    FingerIndependenceTracker,
)
from biomechanics.kinematics import KinematicsTracker


def _straight_index_landmarks():
    lm = [(0, 0, 0)] * 21
    lm[0] = (0, 0, 0)
    lm[5] = (100, 0, 0)
    lm[6] = (150, 0, 0)
    lm[7] = (180, 0, 0)
    lm[8] = (200, 0, 0)  # fully colinear -> all joints ~180
    return lm


def _bent_index_landmarks():
    lm = [(0, 0, 0)] * 21
    lm[0] = (0, 0, 0)
    lm[5] = (100, 0, 0)
    lm[6] = (150, 0, 0)
    lm[7] = (150, 50, 0)   # 90 deg bend at joint2 (PIP)
    lm[8] = (150, 100, 0)  # straight continuation from joint2's new direction
    return lm


def test_straight_finger_all_joints_near_180():
    lm = _straight_index_landmarks()
    angles = finger_joint_angles(lm, "index")
    for name, val in angles.items():
        assert val > 170, f"{name} should be ~180 for a straight finger, got {val}"


def test_bent_finger_joint2_near_90():
    lm = _bent_index_landmarks()
    angles = finger_joint_angles(lm, "index")
    assert 80 <= angles["joint2"] <= 100, f"expected ~90 at joint2, got {angles['joint2']}"


def test_all_fingers_present():
    lm = _straight_index_landmarks()
    result = all_finger_joint_angles(lm)
    assert set(result.keys()) == {"thumb", "index", "middle", "ring", "pinky"}


def test_hand_openness_higher_for_straight_fingers():
    straight = [(0, 0, 0)] * 21
    curled = [(0, 0, 0)] * 21
    for base, mcp, pip, dip, tip in [
        (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)
    ]:
        straight[mcp] = (100, 0, 0)
        straight[pip] = (150, 0, 0)
        straight[dip] = (180, 0, 0)
        straight[tip] = (200, 0, 0)

        curled[mcp] = (100, 0, 0)
        curled[pip] = (150, 0, 0)
        curled[dip] = (140, 30, 0)   # sharply bent back
        curled[tip] = (120, 40, 0)

    open_score = hand_openness(straight)
    curled_score = hand_openness(curled)
    assert open_score > curled_score, f"straight hand ({open_score}) should score higher than curled ({curled_score})"


def test_rom_tracker_records_swing():
    tracker = ROMTracker()
    tracker.update(_straight_index_landmarks())
    tracker.update(_bent_index_landmarks())
    rom = tracker.rom_degrees()
    assert rom is not None
    assert rom["index"] > 60, f"expected a large index ROM swing, got {rom['index']}"


def test_finger_independence_low_when_synchronized():
    tracker = FingerIndependenceTracker(window=30)
    import random
    random.seed(1)
    for _ in range(30):
        v = random.uniform(0, 1)
        tracker.update([v, v, v, v, v])  # all fingers move identically -> perfectly correlated
    score = tracker.score()
    assert score is not None
    assert score < 15, f"perfectly synchronized fingers should score low independence, got {score}"


def test_finger_independence_high_when_uncorrelated():
    tracker = FingerIndependenceTracker(window=60)
    import random
    random.seed(2)
    for _ in range(60):
        tracker.update([random.uniform(0, 1) for _ in range(5)])  # independent random signals
    score = tracker.score()
    assert score is not None
    assert score > 60, f"independently moving fingers should score high independence, got {score}"


def test_kinematics_velocity_matches_known_speed():
    tracker = KinematicsTracker(window_seconds=2.0)
    # move at exactly 100 px/s along x
    for i in range(10):
        t = i * 0.1
        tracker.update((100.0 * t, 0.0), t)
    snap = tracker.snapshot()
    assert snap["velocity_px_s"] is not None
    assert abs(snap["velocity_px_s"] - 100.0) < 5.0, f"expected ~100 px/s, got {snap['velocity_px_s']}"


def test_kinematics_smoothness_higher_for_smooth_motion():
    smooth = KinematicsTracker(window_seconds=2.0)
    jerky = KinematicsTracker(window_seconds=2.0)
    import random
    random.seed(3)
    for i in range(60):
        t = i * 0.033
        # smooth: constant-velocity straight line
        smooth.update((100.0 * t, 0.0), t)
        # jerky: same overall path but with random noise added each step
        jerky.update((100.0 * t + random.uniform(-8, 8), random.uniform(-8, 8)), t)

    s_smooth = smooth.snapshot()["smoothness"]
    s_jerky = jerky.snapshot()["smoothness"]
    assert s_smooth is not None and s_jerky is not None
    assert s_smooth > s_jerky, f"smooth motion ({s_smooth}) should score higher than jerky motion ({s_jerky})"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
