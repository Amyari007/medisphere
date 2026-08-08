"""
Sanity tests for gesture geometry — no camera needed. Synthetic
landmark sets stand in for a real MediaPipe detection.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.gestures import is_open_palm, is_closed_fist, is_pinching, is_point_finger, SwipeDetector


def make_landmarks(finger_states, wrist=(300, 400)):
    """
    Builds a fake 21-point hand. `finger_states` is a 5-bool list
    [thumb, index, middle, ring, pinky] — True = extended outward,
    False = curled back near the wrist.
    """
    wx, wy = wrist
    # base offsets for each finger's MCP, PIP, TIP when curled vs extended
    directions = [(-40, -10), (-15, -60), (0, -70), (15, -60), (30, -50)]
    lm = [(wx, wy, 0.0)] * 21
    lm = list(lm)
    lm[0] = (wx, wy, 0.0)

    tip_idx = [4, 8, 12, 16, 20]
    pip_idx = [3, 6, 10, 14, 18]
    mcp_idx = [2, 5, 9, 13, 17]

    for i, extended in enumerate(finger_states):
        dx, dy = directions[i]
        mcp = (wx + dx * 0.4, wy + dy * 0.4, 0.0)
        if extended:
            pip = (wx + dx * 0.7, wy + dy * 0.7, 0.0)
            tip = (wx + dx * 1.4, wy + dy * 1.4, 0.0)
        else:
            # curled: tip folds back close to wrist, closer than pip
            pip = (wx + dx * 0.6, wy + dy * 0.6, 0.0)
            tip = (wx + dx * 0.3, wy + dy * 0.3, 0.0)
        lm[mcp_idx[i]] = mcp
        lm[pip_idx[i]] = pip
        lm[tip_idx[i]] = tip

    return lm


def test_open_palm():
    lm = make_landmarks([True, True, True, True, True])
    assert is_open_palm(lm), "all fingers extended should register as open palm"
    assert not is_closed_fist(lm)


def test_closed_fist():
    lm = make_landmarks([False, False, False, False, False])
    assert is_closed_fist(lm), "all fingers curled should register as fist"
    assert not is_open_palm(lm)


def test_point_finger():
    lm = make_landmarks([False, True, False, False, False])
    assert is_point_finger(lm), "only index extended should register as pointing"
    assert not is_open_palm(lm)
    assert not is_closed_fist(lm)


def test_pinch():
    lm = make_landmarks([True, True, True, True, True])
    # force thumb tip and index tip close together
    lm[4] = (300, 350, 0.0)
    lm[8] = (305, 352, 0.0)
    assert is_pinching(lm), "thumb and index tip close together should register as pinch"


def test_no_pinch_when_far_apart():
    lm = make_landmarks([True, True, True, True, True])
    lm[4] = (250, 300, 0.0)
    lm[8] = (400, 500, 0.0)
    assert not is_pinching(lm)


def test_swipe_right_detected():
    det = SwipeDetector(window=5, threshold_px=50, cooldown_frames=3)
    result = None
    for x in [100, 110, 130, 160, 200]:
        result = det.update(x)
    assert result == "right", f"expected 'right' swipe, got {result}"


def test_swipe_left_detected():
    det = SwipeDetector(window=5, threshold_px=50, cooldown_frames=3)
    result = None
    for x in [200, 190, 160, 130, 90]:
        result = det.update(x)
    assert result == "left", f"expected 'left' swipe, got {result}"


def test_no_swipe_on_small_motion():
    det = SwipeDetector(window=5, threshold_px=90, cooldown_frames=3)
    result = None
    for x in [200, 205, 202, 208, 204]:
        result = det.update(x)
    assert result is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
