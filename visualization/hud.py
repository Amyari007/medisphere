"""
Clinical AR-style HUD. Every number here comes from the real biomechanics/
kinematics/signal_processing/session modules — nothing decorative except
the panel styling itself.
"""

import time

import cv2
import numpy as np


ACCENT = (255, 190, 60)
GOOD = (140, 255, 170)
WARN = (110, 190, 255)
DIM = (140, 140, 140)


def _panel(frame, x0, y0, x1, y1, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 70), 1, cv2.LINE_AA)


def draw_trend_graph(frame, values, rect, color=ACCENT, label=""):
    """Simple line graph of recent values inside rect=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = rect
    _panel(frame, x0, y0, x1, y1, alpha=0.45)
    if label:
        cv2.putText(frame, label, (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)

    if not values or len(values) < 2:
        return
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    graph_top = y0 + 22
    graph_bottom = y1 - 6
    graph_h = max(1, graph_bottom - graph_top)
    graph_left = x0 + 8
    graph_right = x1 - 8
    graph_w = max(1, graph_right - graph_left)

    pts = []
    for i, v in enumerate(values):
        px = int(graph_left + (i / (len(values) - 1)) * graph_w)
        py = int(graph_bottom - ((v - vmin) / (vmax - vmin)) * graph_h)
        pts.append((px, py))
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
    cv2.circle(frame, pts[-1], 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_clinical_hud(frame, session_view):
    """
    session_view: dict with the real values to display —
        mode_text, openness, extension_pct, stability, tremor, rom,
        velocity, smoothness, independence, rep_count, target_reps,
        elapsed_s, motor_score, motor_score_history, reaction_time_s,
        fatigue_declining, audio_enabled
    """
    h, w = frame.shape[:2]

    # top status bar
    _panel(frame, 0, 0, w, 46)
    cv2.putText(frame, "MediSphere - Rehab", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, session_view.get("mode_text", ""), (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)

    elapsed = session_view.get("elapsed_s", 0.0)
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    timer_text = f"{mm:02d}:{ss:02d}"
    (tw, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, timer_text, (w - tw - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # left metrics panel
    panel_w = 250
    panel_h = 300
    _panel(frame, 0, 50, panel_w, 50 + panel_h)

    def fmt(v, suffix="", none_text="measuring..."):
        return none_text if v is None else f"{v}{suffix}"

    lines = [
        ("Openness", fmt(session_view.get("openness"))),
        ("Extension", fmt(session_view.get("extension_pct"), "%")),
        ("Stability", fmt(session_view.get("stability"), "%")),
        ("Velocity", fmt(session_view.get("velocity"), " px/s")),
        ("Smoothness", fmt(session_view.get("smoothness"))),
        ("Independence", fmt(session_view.get("independence"), "%")),
        ("Tremor", session_view.get("tremor_text", "measuring...")),
        ("ROM (avg)", fmt(session_view.get("rom_avg"), " deg")),
        ("Reaction", fmt(session_view.get("reaction_time_s"), " s", "n/a")),
    ]
    for i, (k, v) in enumerate(lines):
        y = 78 + i * 27
        color = DIM if "measuring" in str(v) or v == "n/a" else GOOD
        cv2.putText(frame, f"{k}:", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, str(v), (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # right panel: reps, motor score, fatigue
    rp_w = 230
    _panel(frame, w - rp_w, 50, w, 50 + 150)
    rep_count = session_view.get("rep_count", 0)
    target = session_view.get("target_reps")
    rep_text = f"{rep_count}/{target}" if target else f"{rep_count}"
    cv2.putText(frame, "Repetitions", (w - rp_w + 12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, rep_text, (w - rp_w + 12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    motor_score = session_view.get("motor_score")
    ms_text = "measuring..." if motor_score is None else f"{motor_score}"
    cv2.putText(frame, "Motor Score", (w - rp_w + 12, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, ms_text, (w - rp_w + 12, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)

    if session_view.get("fatigue_declining"):
        cv2.putText(frame, "Fatigue trend detected", (w - rp_w + 12, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WARN, 1, cv2.LINE_AA)

    if not session_view.get("audio_enabled", False):
        cv2.putText(frame, "(audio feedback off)", (w - rp_w + 12, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)

    # trend graph, bottom-right
    history = session_view.get("motor_score_history", [])
    if history:
        draw_trend_graph(frame, history, (w - rp_w, 50 + 155, w, 50 + 155 + 90), label="Motor score trend")

    # cue banner (reaction-time trial)
    if session_view.get("cue_active"):
        cue_text = "OPEN NOW"
        (cw, ch), _ = cv2.getTextSize(cue_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cx = w // 2 - cw // 2
        cy = 110
        cv2.putText(frame, cue_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # clinical flags panel — heuristic movement-quality alerts, not diagnostic
    flags = session_view.get("flags", [])
    if flags:
        FLAG_COLORS = {"alert": (60, 60, 255), "warning": (50, 165, 255), "info": (255, 190, 120)}
        # alerts first, then warnings, then info; cap to avoid clutter
        priority = {"alert": 0, "warning": 1, "info": 2}
        shown = sorted(flags, key=lambda f: priority.get(f["level"], 3))[:4]

        panel_w = 420
        panel_h = 24 * len(shown) + 14
        px0 = w // 2 - panel_w // 2
        py0 = h - panel_h - 40
        _panel(frame, px0, py0, px0 + panel_w, py0 + panel_h, alpha=0.6)
        for i, f in enumerate(shown):
            color = FLAG_COLORS.get(f["level"], (200, 200, 200))
            y = py0 + 20 + i * 24
            cv2.circle(frame, (px0 + 14, y - 4), 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f["message"], (px0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    hint = "Press 'c' for a reaction-time cue | 'r' reset session | 'q' quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, hint, (w // 2 - hw // 2, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
