"""
A single 0-100 composite "Motor Score" combining real sub-metrics.

IMPORTANT: this is a custom index defined for this project, not a
validated clinical assessment scale. It is NOT the Fugl-Meyer Assessment,
Action Research Arm Test, or any other standardized clinical instrument —
those require validated protocols and trained assessors. Treat this as a
convenient single-number summary of the session's real measurements, not
a diagnostic or clinical-grade score.
"""


def compute_motor_score(avg_extension_pct, stability_score, rom_avg_degrees, smoothness):
    """
    All inputs are real measured values (or None if not yet available).
    Missing sub-scores are simply excluded from the average rather than
    treated as zero, so an incomplete snapshot doesn't unfairly tank the
    score.
    """
    components = []

    if avg_extension_pct is not None:
        components.append(max(0.0, min(100.0, avg_extension_pct)))

    if stability_score is not None:
        components.append(max(0.0, min(100.0, stability_score)))

    if rom_avg_degrees is not None:
        # normalize against an empirical ~120 degree "good" full-session ROM
        rom_score = max(0.0, min(100.0, (rom_avg_degrees / 120.0) * 100.0))
        components.append(rom_score)

    if smoothness is not None:
        # smoothness (log-dimensionless-jerk-style) typically ranges very
        # negative (jerky) to positive (smooth); map a practical range to 0-100
        smooth_score = max(0.0, min(100.0, (smoothness + 20.0) / 40.0 * 100.0))
        components.append(smooth_score)

    if not components:
        return None
    return round(sum(components) / len(components), 1)
