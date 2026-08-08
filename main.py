"""
MediSphere — real-time post-stroke hand rehabilitation platform.

Run:
    python main.py

Keys:
    c - issue a reaction-time cue ("OPEN NOW")
    r - reset the session (clears reps/history, starts a fresh log)
    q - quit (writes a session report before exiting)

Every number in the HUD is computed live from tracked hand landmarks —
see biomechanics/, signal_processing/, and session/ for the underlying
math, and the matching tests/test_*.py files for validation against
known synthetic ground truth. Audio feedback (pitch mapped to hand
openness) requires a working audio output device; if none is found the
app continues silently rather than crashing.
"""

import sys
import time

import cv2

from vision.hand_tracker import HandTracker
from vision.gestures import palm_center, finger_extension_ratios, INDEX_TIP

from biomechanics.joints import hand_openness, ROMTracker, FingerIndependenceTracker
from biomechanics.kinematics import KinematicsTracker

from signal_processing.stability import StabilityTracker
from signal_processing.tremor import TremorAnalyzer

from session.reps import RepetitionTracker
from session.reaction import ReactionTimer
from session.fatigue import FatigueTracker
from session.motor_score import compute_motor_score
from session.clinical_flags import evaluate_flags
from session.logger import SessionLogger

from audio.feedback import ToneFeedback

from visualization.neurosphere import NeuroSphere
from visualization.hologram import HologramRenderer, composite_rgba_onto_bgr
from visualization.hud import draw_clinical_hud
from visualization.skeleton import draw_glow_skeleton

TARGET_REPS = 10
OPEN_THRESHOLD = 0.75
CLOSE_THRESHOLD = 0.25
REACTION_TIMEOUT_S = 4.0
HOLOGRAM_PATH = "models/brain_hologram.glb"  # drop your GLB here; falls back to the procedural sphere if absent


class SessionState:
    """Holds everything that should reset on 'r'."""

    def __init__(self):
        self.rom = ROMTracker()
        self.independence = FingerIndependenceTracker()
        self.kinematics = KinematicsTracker()
        self.stability = StabilityTracker()
        self.tremor = TremorAnalyzer()
        self.reps = RepetitionTracker()
        self.reaction = ReactionTimer()
        self.fatigue = FatigueTracker()
        self.logger = SessionLogger()
        self.motor_score_history = []
        self.cue_active = False
        self.cue_issued_at = None
        self.last_reaction_time = None
        self.start_time = time.time()
        self._was_open = False
        self.printed_flags = set()


def main(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}.", file=sys.stderr)
        return 1

    tracker = HandTracker(max_hands=1)
    sphere = NeuroSphere()
    hologram = HologramRenderer(HOLOGRAM_PATH, render_w=640, render_h=640)
    if hologram.available:
        print(f"[hologram] using 3D model at {HOLOGRAM_PATH}")
    else:
        print("[hologram] using procedural NeuroSphere (3D hologram unavailable)")
    audio = ToneFeedback()
    audio_started = audio.start()

    state = SessionState()
    last_frame_time = time.time()

    print("MediSphere running. 'c' = reaction cue, 'r' = reset session, 'q' = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame grab failed; exiting.", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        dt = max(1e-3, now - last_frame_time)
        last_frame_time = now

        result = tracker.process(frame)
        mode_text = "No hand detected"
        openness = 0.0
        extension_ratios = None
        extension_pct = None
        stability_score = None
        tremor_result = None
        rom = None
        rom_avg = None
        independence_score = None
        velocity = None
        smoothness = None

        if result.hand_landmarks:
            landmarks = HandTracker.landmarks_to_pixels(result.hand_landmarks[0], w, h)
            mode_text = "Tracking"

            openness = hand_openness(landmarks)
            extension_ratios = finger_extension_ratios(landmarks)
            extension_pct = round(sum(extension_ratios) / len(extension_ratios) * 100)

            palm = palm_center(landmarks)
            state.kinematics.update(palm, now)
            kin_snap = state.kinematics.snapshot()
            velocity = None if kin_snap["velocity_px_s"] is None else round(kin_snap["velocity_px_s"], 1)
            smoothness = None if kin_snap["smoothness"] is None else round(kin_snap["smoothness"], 1)

            state.stability.update(palm, now)
            stability_raw, _ = state.stability.score()
            stability_score = None if stability_raw is None else round(stability_raw)

            index_tip = landmarks[INDEX_TIP][:2]
            state.tremor.update(index_tip, now)
            tremor_result = state.tremor.analyze()

            state.rom.update(landmarks)
            rom = state.rom.rom_degrees()
            if rom:
                rom_avg = round(sum(rom.values()) / len(rom.values()), 1)

            state.independence.update(extension_ratios)
            independence_raw = state.independence.score()
            independence_score = None if independence_raw is None else round(independence_raw)

            draw_glow_skeleton(frame, landmarks, extension_ratios)

            # --- discrete open/close state for rep counting + reaction timing ---
            is_open = openness > OPEN_THRESHOLD
            is_closed = openness < CLOSE_THRESHOLD

            if state.cue_active and is_open:
                rt = state.reaction.register_response(now)
                if rt is not None:
                    state.last_reaction_time = rt
                    state.cue_active = False

            prev_open = state._was_open
            state.reps.update(is_open=is_open, is_closed=is_closed, timestamp=now, openness=openness,
                               velocity_px_s=kin_snap["velocity_px_s"])
            if is_open:
                state._was_open = True
            elif is_closed:
                state._was_open = False

            # a rep just completed if count increased since last frame
            if state.reps.completed_reps and (not hasattr(state, "_last_logged_count") or state.reps.count != getattr(state, "_last_logged_count", 0)):
                new_rep = state.reps.completed_reps[-1]
                state.logger.log_rep(state.reps.count - 1, new_rep)
                state._last_logged_count = state.reps.count

        else:
            state.independence.update([0.0] * 5)

        if audio_started:
            audio.set_openness(openness)

        motor_score = compute_motor_score(
            avg_extension_pct=extension_pct,
            stability_score=stability_score,
            rom_avg_degrees=rom_avg,
            smoothness=smoothness,
        )
        if motor_score is not None:
            if not state.motor_score_history or abs(state.motor_score_history[-1] - motor_score) > 0.01:
                state.motor_score_history.append(motor_score)
                state.motor_score_history = state.motor_score_history[-40:]

        fatigue_result = state.fatigue.analyze(state.reps.completed_reps, key="peak_openness")
        fatigue_declining = bool(fatigue_result and fatigue_result["declining"])

        flags = evaluate_flags(
            extension_pct=extension_pct,
            stability_score=stability_score,
            rom_avg=rom_avg,
            smoothness=smoothness,
            independence_score=independence_score,
            reaction_time_s=state.last_reaction_time,
            tremor_result=tremor_result,
            completed_reps=state.reps.completed_reps,
        )
        # print each newly-appearing flag once to the console, rather than
        # spamming every frame while a condition persists
        new_messages = {f["message"] for f in flags} - state.printed_flags
        for f in flags:
            if f["message"] in new_messages:
                print(f"[flag:{f['level']}] {f['message']}")
        state.printed_flags = {f["message"] for f in flags}

        if tremor_result is None:
            tremor_text = "measuring..."
        elif tremor_result["detected"]:
            tremor_text = f"{tremor_result['frequency_hz']} Hz (detected)"
        else:
            tremor_text = "none detected"

        # sphere/hologram always drawn, radius/particles/scale driven by real openness
        center = (w // 2, h // 2)
        if hologram.available:
            state.hologram_rotation = getattr(state, "hologram_rotation", 0.0) + dt * 35.0
            hologram_scale = 0.25 + 0.9 * openness  # shrinks toward a point as hand closes
            render_px = int(180 + 260 * hologram_scale)  # bigger, more visible against real backgrounds

            # dark backdrop "stage" so the hologram reads clearly against a
            # real, cluttered room instead of blending into it
            stage_r = int(render_px * 0.65) + 20
            stage_overlay = frame.copy()
            cv2.circle(stage_overlay, center, stage_r, (8, 6, 6), -1, cv2.LINE_AA)
            cv2.addWeighted(stage_overlay, 0.6, frame, 0.4, 0, dst=frame)

            overlay = hologram.render_overlay(rotation_deg=state.hologram_rotation, scale=1.0)
            composite_rgba_onto_bgr(frame, overlay, center, overlay_scale=render_px / hologram.render_width)
        else:
            sphere.update(openness, dt)
            sphere.draw(frame, center)

        if state.cue_active and (now - state.cue_issued_at) > REACTION_TIMEOUT_S:
            state.cue_active = False  # timed out, no response

        session_view = {
            "mode_text": mode_text,
            "openness": round(openness, 2),
            "extension_pct": extension_pct,
            "stability": stability_score,
            "velocity": velocity,
            "smoothness": smoothness,
            "independence": independence_score,
            "tremor_text": tremor_text,
            "rom_avg": rom_avg,
            "reaction_time_s": state.last_reaction_time,
            "rep_count": state.reps.count,
            "target_reps": TARGET_REPS,
            "elapsed_s": now - state.start_time,
            "motor_score": motor_score,
            "motor_score_history": state.motor_score_history,
            "fatigue_declining": fatigue_declining,
            "audio_enabled": audio_started,
            "cue_active": state.cue_active,
            "flags": flags,
        }
        draw_clinical_hud(frame, session_view)

        cv2.imshow("MediSphere", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not state.cue_active:
            if openness > CLOSE_THRESHOLD:
                # Reaction time only means something measured from a closed
                # hand — issuing the cue while already open would measure
                # near-zero "reaction" that isn't really a reaction to anything.
                print("Close your hand first, then press 'c' for a reaction cue.")
            else:
                state.cue_active = True
                state.cue_issued_at = now
                state.reaction.issue_cue(now)
        elif key == ord('r'):
            print("Session reset.")
            state = SessionState()

    total_reps = state.reps.count
    final_score = state.motor_score_history[-1] if state.motor_score_history else None
    state.logger.finalize_session(total_reps=total_reps, motor_score=final_score, fatigue_declining=fatigue_declining)
    report_path, report_text = state.logger.generate_report()
    print("\n" + report_text)
    print(f"\nSession report saved to: {report_path}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    audio.stop()
    hologram.close()
    return 0


if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(main(cam_idx))
