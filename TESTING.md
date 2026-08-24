# Testing

This document lists every automated test in MediSphere, organized by module, with a plain explanation of what each test checks and why it exists.

**Current result: 57 / 57 tests passing.**

Run them yourself:
```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Summary

| Module | File | Test Count |
|---|---|---|
| Gesture recognition | `tests/test_gestures.py` | 8 |
| Biomechanics & kinematics | `tests/test_biomechanics.py` | 9 |
| Signal processing (tremor, filtering) | `tests/test_signal_processing.py` | 5 |
| Audio feedback | `tests/test_audio.py` | 3 |
| Session logic (reps, reaction, fatigue, scoring, logging) | `tests/test_session.py` | 12 |
| Clinical flags | `tests/test_clinical_flags.py` | 17 |
| Full pipeline integration | `tests/test_integration.py` | 2 |
| Application entry point | `tests/test_main_dryrun.py` | 1 |
| **Total** | | **57** |

---

## Gesture Recognition — `test_gestures.py`

| Test | What it checks |
|---|---|
| `test_open_palm` | An open-hand landmark pattern is correctly classified as "open palm" |
| `test_closed_fist` | A curled-hand landmark pattern is correctly classified as "closed fist" |
| `test_point_finger` | A single-extended-finger pattern is correctly classified as "pointing" |
| `test_pinch` | Thumb and index tip brought close together is correctly classified as a "pinch" |
| `test_no_pinch_when_far_apart` | Thumb and index tip held apart is correctly classified as *not* a pinch |
| `test_swipe_right_detected` | A hand moving steadily rightward is correctly classified as a right swipe |
| `test_swipe_left_detected` | A hand moving steadily leftward is correctly classified as a left swipe |
| `test_no_swipe_on_small_motion` | Small, non-directional hand movement does not falsely register as a swipe |

---

## Biomechanics & Kinematics — `test_biomechanics.py`

| Test | What it checks |
|---|---|
| `test_straight_finger_all_joints_near_180` | A fully straight finger's joint angles read close to 180° |
| `test_bent_finger_joint2_near_90` | A finger bent at a right angle reads close to 90° at that joint |
| `test_all_fingers_present` | Joint-angle output includes all five fingers, every frame |
| `test_hand_openness_higher_for_straight_fingers` | The openness score is higher for extended fingers than curled ones |
| `test_rom_tracker_records_swing` | Range-of-motion tracking correctly records the difference between a finger's most-curled and most-extended positions |
| `test_finger_independence_low_when_synchronized` | Fingers that move identically together score low on the independence metric |
| `test_finger_independence_high_when_uncorrelated` | Fingers that move independently of each other score high on the independence metric |
| `test_kinematics_velocity_matches_known_speed` | A point moved at an exact 100 px/s is measured as ≈100 px/s |
| `test_kinematics_smoothness_higher_for_smooth_motion` | A constant-velocity path scores higher on smoothness than a randomly jittered path over the same route |

---

## Signal Processing — `test_signal_processing.py`

| Test | What it checks |
|---|---|
| `test_lowpass_filter_removes_high_frequency_noise` | A 25 Hz noise component mixed into a 2 Hz signal is suppressed by the filter, leaving the 2 Hz content intact |
| `test_tremor_detects_known_frequency_after_filtering` | A deliberately injected 6 Hz oscillation is correctly detected at ≈6 Hz |
| `test_tremor_still_ignores_still_hand_after_filtering` | A perfectly stationary signal does not register as tremor |
| `test_tremor_ignores_a_single_noisy_frame_spike` | Four seconds of pure random jitter (no real periodic pattern) never triggers a false tremor detection |
| `test_stability_score_sane_range` | The stability score stays within its defined 0–100 bounds |

---

## Audio Feedback — `test_audio.py`

| Test | What it checks |
|---|---|
| `test_pitch_decreases_as_hand_opens` | The pitch-mapping function produces a lower frequency as hand openness increases |
| `test_generated_waveform_actually_contains_target_frequency` | A requested 440 Hz tone is generated, then independently re-analyzed with an FFT, confirming the output actually contains 440 Hz |
| `test_consecutive_blocks_are_phase_continuous` | Back-to-back audio buffer chunks connect smoothly, without an audible click at the seam |

---

## Session Logic — `test_session.py`

| Test | What it checks |
|---|---|
| `test_repetition_tracker_counts_full_open_close_cycles` | A complete open-then-close hand cycle is counted as one repetition |
| `test_repetition_tracker_ignores_partial_open_without_close` | An open hand that never closes again is not counted as a repetition |
| `test_repetition_tracker_ignores_brief_threshold_jitter` | 200 frames of rapid flicker across the open/close boundary produce **zero** false repetitions |
| `test_repetition_tracker_counts_genuine_slow_cycle_despite_debounce` | One deliberate, sustained open-close cycle is still counted correctly despite the jitter filter above |
| `test_reaction_timer_measures_real_elapsed_time` | The time between a cue and a response is measured correctly |
| `test_reaction_timer_ignores_response_without_cue` | A response with no prior cue does not register a reaction time |
| `test_fatigue_detects_real_declining_trend` | A clearly declining sequence of rep values is flagged as a fatigue trend |
| `test_fatigue_does_not_flag_stable_performance` | A flat, non-declining sequence is not flagged as fatigue |
| `test_fatigue_returns_none_with_too_few_reps` | Fatigue analysis withholds a result until enough repetitions have occurred |
| `test_motor_score_higher_for_better_performance` | Better sub-metric inputs produce a higher composite score than worse ones |
| `test_motor_score_handles_missing_submetrics` | The composite score still computes sensibly when one or more sub-metrics are unavailable |
| `test_session_logger_round_trip` | Data written to SQLite and CSV is read back identically to what was written |

---

## Clinical Flags — `test_clinical_flags.py`

| Test | What it checks |
|---|---|
| `test_no_flags_for_healthy_looking_values` | No flags fire when all metrics are within normal-looking ranges |
| `test_flags_reduced_extension` | A low extension value triggers the reduced-extension flag |
| `test_no_flag_at_extension_boundary_and_above` | The extension flag does not fire exactly at or above its threshold |
| `test_flags_limited_rom` | A low range-of-motion value triggers the limited-ROM flag |
| `test_flags_reduced_stability` | A low stability value triggers the reduced-stability flag |
| `test_flags_jerky_movement` | A poor smoothness value triggers the jerky-movement flag |
| `test_no_flag_for_smooth_movement` | A good smoothness value does not trigger the jerky-movement flag |
| `test_flags_low_independence_as_possible_enslaving` | A low independence score triggers the possible-enslaving flag |
| `test_flags_slow_reaction_as_info_not_warning` | A slow reaction time is flagged at "info" level, not treated as a warning |
| `test_flags_unsatisfactory_hand_opening` | Consistently low peak openness across recent reps triggers a flag |
| `test_no_flag_for_good_hand_opening` | Consistently high peak openness does not trigger that flag |
| `test_recent_peak_openness_avg_uses_only_last_n_reps` | The rolling openness average correctly uses only the most recent reps, not the whole session |
| `test_recent_peak_openness_avg_none_for_empty_reps` | The rolling average returns nothing when there is no rep data yet |
| `test_tremor_severity_mild_vs_significant` | Tremor amplitude correctly determines "mild" vs. "significant" severity |
| `test_no_tremor_flag_when_not_detected` | No tremor flag appears when tremor was not detected |
| `test_none_inputs_produce_no_flags` | Missing data never produces a false flag |
| `test_multiple_simultaneous_flags` | Multiple flags can fire at once, independently, without interfering with each other |

---

## Full Pipeline Integration — `test_integration.py`

| Test | What it checks |
|---|---|
| `test_full_session_pipeline_runs_and_produces_sane_values` | A complete simulated session — multiple reps, biomechanics, kinematics, session tracking together — runs and produces values within expected ranges |
| `test_tremor_detected_when_injected_into_session` | A tremor injected into a full simulated session is still correctly detected end-to-end, not just in an isolated unit test |

---

## Application Entry Point — `test_main_dryrun.py`

| Test | What it checks |
|---|---|
| `test_main_runs_end_to_end_without_crashing` | `main.py`'s actual code runs start to finish — camera and hand-tracking calls swapped for deterministic test input — including writing a session report to disk, without error |

---

## Two issues these tests were written to catch after they were first found live

- **Repetition count inflation:** an early version counted 79 repetitions in under 4 minutes of real use, caused by landmark jitter flickering across the open/close threshold. `test_repetition_tracker_ignores_brief_threshold_jitter` now guards against this specifically.
- **Tremor false positives:** a single noisy analysis frame could occasionally register a false tremor detection. `test_tremor_ignores_a_single_noisy_frame_spike` now guards against this specifically.
