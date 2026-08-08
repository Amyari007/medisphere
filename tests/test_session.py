import sys
import os
import shutil
import csv as csv_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from session.reps import RepetitionTracker
from session.reaction import ReactionTimer
from session.fatigue import FatigueTracker
from session.motor_score import compute_motor_score
from session.logger import SessionLogger


def test_repetition_tracker_counts_full_open_close_cycles():
    tracker = RepetitionTracker(open_hold_s=0.05, close_hold_s=0.05)
    t = 0.0
    # cycle 1: hold open past the debounce, then hold closed past it
    tracker.update(is_open=True, is_closed=False, timestamp=t, openness=0.9, velocity_px_s=50); t += 0.1
    tracker.update(is_open=True, is_closed=False, timestamp=t, openness=1.0, velocity_px_s=60); t += 0.1
    tracker.update(is_open=False, is_closed=True, timestamp=t); t += 0.1
    tracker.update(is_open=False, is_closed=True, timestamp=t)  # sustains closed past hold
    assert tracker.count == 1, f"expected 1 completed rep, got {tracker.count}"

    # cycle 2
    t += 0.1
    tracker.update(is_open=True, is_closed=False, timestamp=t, openness=0.8, velocity_px_s=40); t += 0.1
    tracker.update(is_open=True, is_closed=False, timestamp=t, openness=0.8, velocity_px_s=40); t += 0.1
    tracker.update(is_open=False, is_closed=True, timestamp=t); t += 0.1
    tracker.update(is_open=False, is_closed=True, timestamp=t)
    assert tracker.count == 2, f"expected 2 completed reps, got {tracker.count}"
    assert len(tracker.completed_reps) == 2
    assert tracker.completed_reps[0]["peak_openness"] == 1.0


def test_repetition_tracker_ignores_partial_open_without_close():
    tracker = RepetitionTracker(open_hold_s=0.05, close_hold_s=0.05)
    tracker.update(is_open=True, is_closed=False, timestamp=0.0, openness=0.5)
    tracker.update(is_open=True, is_closed=False, timestamp=0.1, openness=0.6)
    assert tracker.count == 0, "a rep that never closes shouldn't count yet"


def test_repetition_tracker_ignores_brief_threshold_jitter():
    """Regression test for a real observed bug: rapid flicker across the
    open/close threshold (landmark jitter) inflated the rep count to 79
    in under 4 minutes. A transition must be HELD for the debounce
    duration to count - single-frame spikes shouldn't register at all."""
    tracker = RepetitionTracker(open_hold_s=0.15, close_hold_s=0.15)
    t = 0.0
    # hand genuinely closed, but openness jitters briefly above the open
    # threshold for a single frame every so often - each spike is shorter
    # than open_hold_s, so none of them should count as a real transition
    import random
    random.seed(5)
    for _ in range(200):
        jitter_open = random.random() < 0.1  # 10% of frames spike open briefly
        tracker.update(is_open=jitter_open, is_closed=not jitter_open, timestamp=t, openness=0.8 if jitter_open else 0.1)
        t += 0.02  # 50fps-equivalent, so each spike lasts ~20ms, well under the 150ms hold
    assert tracker.count == 0, f"brief threshold jitter should never count as a rep, got {tracker.count}"


def test_repetition_tracker_counts_genuine_slow_cycle_despite_debounce():
    """A real, deliberate open-close cycle (each phase held well past the
    debounce window) should still count normally."""
    tracker = RepetitionTracker(open_hold_s=0.15, close_hold_s=0.15)
    t = 0.0
    for _ in range(20):  # hold open for 0.4s
        tracker.update(is_open=True, is_closed=False, timestamp=t, openness=0.95)
        t += 0.02
    for _ in range(20):  # hold closed for 0.4s
        tracker.update(is_open=False, is_closed=True, timestamp=t, openness=0.05)
        t += 0.02
    assert tracker.count == 1, f"a genuine slow open-close cycle should count once, got {tracker.count}"


def test_reaction_timer_measures_real_elapsed_time():
    timer = ReactionTimer()
    timer.issue_cue(timestamp=10.0)
    assert timer.is_waiting_for_response()
    elapsed = timer.register_response(timestamp=10.35)
    assert abs(elapsed - 0.35) < 1e-6, f"expected 0.35s elapsed, got {elapsed}"
    assert not timer.is_waiting_for_response()


def test_reaction_timer_ignores_response_without_cue():
    timer = ReactionTimer()
    result = timer.register_response(timestamp=5.0)
    assert result is None


def test_fatigue_detects_real_declining_trend():
    tracker = FatigueTracker(min_reps=4)
    reps = [{"peak_openness": v} for v in [1.0, 0.9, 0.8, 0.7, 0.6]]  # clearly declining
    result = tracker.analyze(reps, key="peak_openness")
    assert result is not None
    assert result["declining"], f"expected declining=True for a clear downward trend, got {result}"


def test_fatigue_does_not_flag_stable_performance():
    tracker = FatigueTracker(min_reps=4)
    reps = [{"peak_openness": v} for v in [0.9, 0.91, 0.89, 0.90, 0.90]]  # flat/stable
    result = tracker.analyze(reps, key="peak_openness")
    assert result is not None
    assert not result["declining"], f"expected declining=False for stable performance, got {result}"


def test_fatigue_returns_none_with_too_few_reps():
    tracker = FatigueTracker(min_reps=4)
    result = tracker.analyze([{"peak_openness": 1.0}], key="peak_openness")
    assert result is None


def test_motor_score_higher_for_better_performance():
    good = compute_motor_score(avg_extension_pct=90, stability_score=95, rom_avg_degrees=110, smoothness=10)
    poor = compute_motor_score(avg_extension_pct=30, stability_score=25, rom_avg_degrees=20, smoothness=-15)
    assert good is not None and poor is not None
    assert good > poor, f"good performance ({good}) should score higher than poor ({poor})"
    assert 0 <= good <= 100 and 0 <= poor <= 100


def test_motor_score_handles_missing_submetrics():
    score = compute_motor_score(avg_extension_pct=80, stability_score=None, rom_avg_degrees=None, smoothness=None)
    assert score is not None
    assert 0 <= score <= 100


def test_session_logger_round_trip():
    test_dir = "/tmp/medisphere_test_session"
    shutil.rmtree(test_dir, ignore_errors=True)
    logger = SessionLogger(db_path=os.path.join(test_dir, "test.db"), csv_dir=test_dir)

    logger.log_rep(0, {"duration_s": 1.2, "peak_openness": 0.9, "peak_velocity_px_s": 55.0})
    logger.log_rep(1, {"duration_s": 1.5, "peak_openness": 0.85, "peak_velocity_px_s": 60.0})
    logger.finalize_session(total_reps=2, motor_score=78.5, fatigue_declining=False)

    # verify SQLite round-trip
    rows = logger.read_all_reps()
    assert len(rows) == 2, f"expected 2 rows back from SQLite, got {len(rows)}"
    assert rows[0]["duration_s"] == 1.2
    assert rows[1]["peak_velocity_px_s"] == 60.0

    # verify CSV round-trip
    csv_path = os.path.join(test_dir, f"{logger.session_id}_reps.csv")
    assert os.path.exists(csv_path), "CSV file should have been written"
    with open(csv_path) as f:
        csv_rows = list(csv_module.DictReader(f))
    assert len(csv_rows) == 2
    assert float(csv_rows[0]["duration_s"]) == 1.2

    # verify report generation
    report_path, report_text = logger.generate_report()
    assert os.path.exists(report_path)
    assert "Total repetitions: 2" in report_text
    assert "78.5" in report_text

    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
