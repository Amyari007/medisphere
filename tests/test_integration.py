"""
Simulates a realistic session (hand opening and closing over several
reps, with a synthetic tremor injected) through the exact modules
main.py wires together — biomechanics, kinematics, signal processing,
session tracking — to catch integration bugs that per-module unit tests
can't see. No camera needed: this drives everything with synthetic
landmark sequences.
"""
import sys
import os
import math
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.gestures import palm_center, finger_extension_ratios, INDEX_TIP
from biomechanics.joints import hand_openness, ROMTracker, FingerIndependenceTracker
from biomechanics.kinematics import KinematicsTracker
from signal_processing.stability import StabilityTracker
from signal_processing.tremor import TremorAnalyzer
from session.reps import RepetitionTracker
from session.fatigue import FatigueTracker
from session.motor_score import compute_motor_score
from session.logger import SessionLogger


def make_hand(openness, wrist=(300, 400), tremor_offset=(0.0, 0.0)):
    """
    Builds a synthetic 21-point hand at a given openness level (0=fully
    curled, 1=fully extended). Unlike a naive colinear model, this
    actually bends the DIP/TIP segment relative to the MCP-PIP segment
    by an angle that shrinks toward 0 (straight) as openness -> 1, so
    joint angles (and therefore ROM) genuinely vary with openness -
    matching how a real curling finger behaves.
    """
    wx, wy = wrist
    wx += tremor_offset[0]
    wy += tremor_offset[1]
    directions = [(-40, -10), (-15, -60), (0, -70), (15, -60), (30, -50)]
    lm = [(wx, wy, 0.0)] * 21
    lm = list(lm)
    lm[0] = (wx, wy, 0.0)

    fingers = {
        0: (1, 2, 3, 4),
        1: (5, 6, 7, 8),
        2: (9, 10, 11, 12),
        3: (13, 14, 15, 16),
        4: (17, 18, 19, 20),
    }
    bend_deg = (1.0 - openness) * 160.0  # fully curled -> tip folds back toward wrist, fully open -> straight
    bend_rad = math.radians(bend_deg)

    for i, (dx, dy) in enumerate(directions):
        mcp, pip, dip, tip = fingers[i]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length  # unit direction, MCP->PIP

        mcp_pos = (wx + ux * length * 0.4, wy + uy * length * 0.4)
        pip_pos = (wx + ux * length * 0.7, wy + uy * length * 0.7)

        # rotate the outward direction by bend_rad to fold DIP/TIP inward
        rux = ux * math.cos(bend_rad) - uy * math.sin(bend_rad)
        ruy = ux * math.sin(bend_rad) + uy * math.cos(bend_rad)

        dip_pos = (pip_pos[0] + rux * length * 0.3, pip_pos[1] + ruy * length * 0.3)
        tip_pos = (dip_pos[0] + rux * length * 0.3, dip_pos[1] + ruy * length * 0.3)

        lm[mcp] = (mcp_pos[0], mcp_pos[1], 0.0)
        lm[pip] = (pip_pos[0], pip_pos[1], 0.0)
        lm[dip] = (dip_pos[0], dip_pos[1], 0.0)
        lm[tip] = (tip_pos[0], tip_pos[1], 0.0)
    return lm


def test_full_session_pipeline_runs_and_produces_sane_values():
    test_dir = "/tmp/medisphere_integration_test"
    shutil.rmtree(test_dir, ignore_errors=True)

    rom = ROMTracker()
    independence = FingerIndependenceTracker()
    kinematics = KinematicsTracker()
    stability = StabilityTracker()
    tremor = TremorAnalyzer()
    reps = RepetitionTracker()
    fatigue = FatigueTracker(min_reps=3)
    logger = SessionLogger(db_path=os.path.join(test_dir, "t.db"), csv_dir=test_dir)

    t = 0.0
    dt = 1.0 / 30.0
    rep_target = 5
    motor_scores = []

    for rep_i in range(rep_target):
        # ramp open over 20 frames, hold, ramp closed over 20 frames
        openness_curve = (
            [i / 20.0 for i in range(20)] + [1.0] * 10 + [1.0 - i / 20.0 for i in range(20)] + [0.0] * 5
        )
        for openness in openness_curve:
            landmarks = make_hand(openness)
            palm = palm_center(landmarks)
            ext_ratios = finger_extension_ratios(landmarks)
            ext_pct = round(sum(ext_ratios) / len(ext_ratios) * 100)

            kinematics.update(palm, t)
            kin = kinematics.snapshot()
            stability.update(palm, t)
            stab_score, _ = stability.score()
            tremor.update(landmarks[INDEX_TIP][:2], t)
            rom.update(landmarks)
            independence.update(ext_ratios)

            is_open = openness > 0.75
            is_closed = openness < 0.25
            reps.update(is_open=is_open, is_closed=is_closed, timestamp=t, openness=openness,
                        velocity_px_s=kin["velocity_px_s"])

            rom_dict = rom.rom_degrees()
            rom_avg = sum(rom_dict.values()) / len(rom_dict) if rom_dict else None
            score = compute_motor_score(ext_pct, stab_score, rom_avg, kin["smoothness"])
            if score is not None:
                motor_scores.append(score)

            t += dt

    # --- sanity assertions on the whole pipeline ---
    assert reps.count == rep_target, f"expected {rep_target} completed reps, got {reps.count}"
    assert len(reps.completed_reps) == rep_target

    for i, rep_data in enumerate(reps.completed_reps):
        logger.log_rep(i, rep_data)
    fatigue_result = fatigue.analyze(reps.completed_reps, key="peak_openness")
    assert fatigue_result is not None
    assert fatigue_result["declining"] is False, "constant-amplitude reps shouldn't register as fatigue"

    logger.finalize_session(total_reps=reps.count, motor_score=motor_scores[-1] if motor_scores else None,
                             fatigue_declining=fatigue_result["declining"])

    logged_rows = logger.read_all_reps()
    assert len(logged_rows) == rep_target, f"expected {rep_target} logged rows, got {len(logged_rows)}"

    rom_final = rom.rom_degrees()
    assert rom_final is not None
    assert all(v > 20 for v in rom_final.values()), f"expected meaningful ROM swings, got {rom_final}"

    indep_score = independence.score()
    assert indep_score is not None
    # all fingers moved in lockstep in this synthetic hand -> low independence
    # expected among whichever fingers had enough variance to correlate
    assert indep_score < 40, f"synchronized fingers should score low independence, got {indep_score}"

    assert len(motor_scores) > 0
    assert all(0 <= s <= 100 for s in motor_scores)

    report_path, report_text = logger.generate_report()
    assert f"Total repetitions: {rep_target}" in report_text

    shutil.rmtree(test_dir, ignore_errors=True)


def test_tremor_detected_when_injected_into_session():
    """Same pipeline, but with a real 6Hz tremor injected into the wrist
    position throughout — verifies tremor detection survives the full
    integration, not just an isolated unit test. analyze() is called every
    frame (matching main.py), so the sustained oscillation should confirm
    detection well before the buffer ends."""
    tremor = TremorAnalyzer()
    t = 0.0
    fs = 30.0
    result = None
    for i in range(int(fs * 4.5)):
        t = i / fs
        tremor_offset = (6.0 * math.sin(2 * math.pi * 6.0 * t), 0.0)
        landmarks = make_hand(openness=0.6, tremor_offset=tremor_offset)
        tremor.update(landmarks[INDEX_TIP][:2], t)
        result = tremor.analyze()
    assert result is not None
    assert result["detected"], f"expected sustained tremor detected in integration context, got {result}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
