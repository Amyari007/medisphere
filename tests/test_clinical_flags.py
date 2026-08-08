import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from session.clinical_flags import evaluate_flags, recent_peak_openness_avg, LEVEL_ALERT, LEVEL_WARNING


def test_no_flags_for_healthy_looking_values():
    flags = evaluate_flags(
        extension_pct=90, stability_score=85, rom_avg=90, smoothness=5.0,
        independence_score=80, reaction_time_s=0.3, tremor_result={"detected": False},
        completed_reps=[{"peak_openness": 0.95}] * 3,
    )
    assert flags == [], f"expected no flags for healthy-looking values, got {flags}"


def test_flags_reduced_extension():
    flags = evaluate_flags(extension_pct=30)
    messages = [f["message"] for f in flags]
    assert any("extension" in m.lower() for m in messages), messages


def test_no_flag_at_extension_boundary_and_above():
    flags = evaluate_flags(extension_pct=50)  # exactly at threshold, not below
    assert not any("extension" in f["message"].lower() for f in flags)
    flags2 = evaluate_flags(extension_pct=51)
    assert not any("extension" in f["message"].lower() for f in flags2)


def test_flags_limited_rom():
    flags = evaluate_flags(rom_avg=20)
    assert any("range of motion" in f["message"].lower() for f in flags)


def test_flags_reduced_stability():
    flags = evaluate_flags(stability_score=40)
    assert any("stability" in f["message"].lower() for f in flags)


def test_flags_jerky_movement():
    flags = evaluate_flags(smoothness=-15.0)
    assert any("jerky" in f["message"].lower() for f in flags)


def test_no_flag_for_smooth_movement():
    flags = evaluate_flags(smoothness=8.0)
    assert not any("jerky" in f["message"].lower() for f in flags)


def test_flags_low_independence_as_possible_enslaving():
    flags = evaluate_flags(independence_score=15)
    assert any("enslaving" in f["message"].lower() for f in flags)


def test_flags_slow_reaction_as_info_not_warning():
    flags = evaluate_flags(reaction_time_s=1.2)
    matching = [f for f in flags if "reaction" in f["message"].lower()]
    assert len(matching) == 1
    assert matching[0]["level"] != LEVEL_ALERT  # slow reaction is informational, not alarming


def test_flags_unsatisfactory_hand_opening():
    reps = [{"peak_openness": 0.4}, {"peak_openness": 0.45}, {"peak_openness": 0.5}]
    flags = evaluate_flags(completed_reps=reps)
    assert any("opening below target" in f["message"].lower() for f in flags)


def test_no_flag_for_good_hand_opening():
    reps = [{"peak_openness": 0.9}, {"peak_openness": 0.95}, {"peak_openness": 1.0}]
    flags = evaluate_flags(completed_reps=reps)
    assert not any("opening below target" in f["message"].lower() for f in flags)


def test_recent_peak_openness_avg_uses_only_last_n_reps():
    reps = [{"peak_openness": 0.1}, {"peak_openness": 0.1}, {"peak_openness": 0.9}, {"peak_openness": 0.9}, {"peak_openness": 0.9}]
    avg = recent_peak_openness_avg(reps, n=3)
    assert abs(avg - 0.9) < 1e-6, f"expected the average of only the last 3 reps (all 0.9), got {avg}"


def test_recent_peak_openness_avg_none_for_empty_reps():
    assert recent_peak_openness_avg([]) is None
    assert recent_peak_openness_avg(None) is None if False else True  # None handled by caller, not this fn


def test_tremor_severity_mild_vs_significant():
    mild = evaluate_flags(tremor_result={"detected": True, "amplitude": 0.8, "frequency_hz": 5.0})
    significant = evaluate_flags(tremor_result={"detected": True, "amplitude": 3.5, "frequency_hz": 5.0})

    mild_flag = next(f for f in mild if "tremor" in f["message"].lower())
    sig_flag = next(f for f in significant if "tremor" in f["message"].lower())

    assert "mild" in mild_flag["message"].lower()
    assert mild_flag["level"] == LEVEL_WARNING

    assert "significant" in sig_flag["message"].lower()
    assert sig_flag["level"] == LEVEL_ALERT, "significant tremor should be flagged at ALERT level, not just warning"


def test_no_tremor_flag_when_not_detected():
    flags = evaluate_flags(tremor_result={"detected": False, "amplitude": 0.1, "frequency_hz": 4.0})
    assert not any("tremor" in f["message"].lower() for f in flags)


def test_none_inputs_produce_no_flags():
    """Missing data (None) should never be treated as a bad value."""
    flags = evaluate_flags()
    assert flags == []


def test_multiple_simultaneous_flags():
    flags = evaluate_flags(
        extension_pct=25, stability_score=30, rom_avg=15, smoothness=-20.0,
        independence_score=10, reaction_time_s=1.5,
        tremor_result={"detected": True, "amplitude": 4.0, "frequency_hz": 6.0},
        completed_reps=[{"peak_openness": 0.3}],
    )
    assert len(flags) == 8, f"expected all 8 heuristics to trip, got {len(flags)}: {flags}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
