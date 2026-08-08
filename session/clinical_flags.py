"""
Flags movement-quality patterns against heuristic thresholds, based on
general characterizations from movement-science / stroke-rehab
literature (reduced extension, limited ROM, jerky movement, low finger
independence/"enslaving", tremor, slow reaction, unsatisfactory hand
opening).

IMPORTANT: these thresholds are NOT validated clinical cutoffs. They
were not derived from patient data or a clinical study — they're
reasonable heuristic values based on the general direction findings in
the literature (e.g. "post-stroke smoothness is often markedly reduced")
translated into a concrete number so the app can show *something*
actionable. Do not use these flags to make clinical decisions. See
README.md for the full caveat.
"""

THRESHOLDS = {
    "extension_low_pct": 50,       # below this -> reduced extension flag
    "rom_low_deg": 50,             # avg ROM below this -> limited ROM flag
    "stability_low": 60,           # stability score below this -> reduced stability flag
    "smoothness_low": -5.0,        # smoothness below this -> jerky movement flag
    "independence_low": 40,        # independence score below this -> possible enslaving flag
    "reaction_slow_s": 0.6,        # reaction time above this -> slow reaction flag
    "openness_unsatisfactory": 0.6,  # avg peak openness across recent reps below this -> flag
    "tremor_significant_amp": 2.0,   # tremor amplitude above this -> "significant" vs "mild"
}

LEVEL_INFO = "info"
LEVEL_WARNING = "warning"
LEVEL_ALERT = "alert"


def recent_peak_openness_avg(completed_reps, n=3):
    """Average peak openness across the last n completed reps, or None."""
    if not completed_reps:
        return None
    recent = completed_reps[-n:]
    values = [r["peak_openness"] for r in recent if r.get("peak_openness") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_flags(extension_pct=None, stability_score=None, rom_avg=None, smoothness=None,
                    independence_score=None, reaction_time_s=None, tremor_result=None,
                    completed_reps=None, thresholds=None):
    """
    Returns a list of {"level": ..., "message": ...} dicts for whichever
    heuristics the current values trip. Any input left as None is simply
    skipped (not enough data yet), never treated as a bad value.
    """
    t = thresholds or THRESHOLDS
    flags = []

    if extension_pct is not None and extension_pct < t["extension_low_pct"]:
        flags.append({"level": LEVEL_WARNING, "message": f"Reduced finger extension ({extension_pct}%)"})

    if rom_avg is not None and rom_avg < t["rom_low_deg"]:
        flags.append({"level": LEVEL_WARNING, "message": f"Limited range of motion ({rom_avg} deg avg)"})

    if stability_score is not None and stability_score < t["stability_low"]:
        flags.append({"level": LEVEL_WARNING, "message": f"Reduced hold stability ({stability_score}%)"})

    if smoothness is not None and smoothness < t["smoothness_low"]:
        flags.append({"level": LEVEL_WARNING, "message": "Jerky / segmented movement pattern"})

    if independence_score is not None and independence_score < t["independence_low"]:
        flags.append({"level": LEVEL_WARNING,
                       "message": f"Low finger independence ({independence_score}%) - possible enslaving"})

    if reaction_time_s is not None and reaction_time_s > t["reaction_slow_s"]:
        flags.append({"level": LEVEL_INFO, "message": f"Slow reaction time ({reaction_time_s}s)"})

    avg_open = recent_peak_openness_avg(completed_reps) if completed_reps else None
    if avg_open is not None and avg_open < t["openness_unsatisfactory"]:
        flags.append({"level": LEVEL_WARNING,
                       "message": f"Hand opening below target ({int(avg_open * 100)}% avg over recent reps)"})

    if tremor_result and tremor_result.get("detected"):
        amp = tremor_result.get("amplitude", 0.0)
        freq = tremor_result.get("frequency_hz", 0.0)
        significant = amp > t["tremor_significant_amp"]
        severity = "significant" if significant else "mild"
        flags.append({
            "level": LEVEL_ALERT if significant else LEVEL_WARNING,
            "message": f"Tremor detected - {severity} ({freq} Hz)",
        })

    return flags
