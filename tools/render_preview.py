"""
Renders a static preview of the NeuroSphere + HUD without a webcam
attached — useful for checking the visuals in CI or a headless
environment. This exercises the same visualization code main.py uses,
just fed synthetic session data instead of a live camera.

Run: python tools/render_preview.py
Output: tools/preview_neurosphere.png
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from visualization.neurosphere import NeuroSphere
from visualization.hud import draw_clinical_hud

W, H = 1000, 600

if __name__ == "__main__":
    sphere = NeuroSphere()
    for i in range(70):  # ramp the sphere open before capturing the frame
        sphere.update(min(1.0, i / 50.0), dt=1 / 30.0)

    frame = np.full((H, W, 3), (18, 18, 22), dtype=np.uint8)
    sphere.draw(frame, (W // 2, H // 2 - 40))

    session_view = {
        "mode_text": "Session active", "openness": 0.95, "extension_pct": 92,
        "stability": 84, "velocity": 38.5, "smoothness": 5.1, "independence": 68,
        "tremor_text": "none detected", "rom_avg": 71.2, "reaction_time_s": 0.38,
        "rep_count": 4, "target_reps": 10, "elapsed_s": 130, "motor_score": 79.4,
        "motor_score_history": [55, 61, 68, 72, 75, 79.4],
        "fatigue_declining": False, "audio_enabled": True, "cue_active": False,
    }
    draw_clinical_hud(frame, session_view)

    out_path = os.path.join(os.path.dirname(__file__), "preview_neurosphere.png")
    cv2.imwrite(out_path, frame)
    print(f"wrote {out_path}")
    print("Note: the 3D hologram renderer (visualization/hologram.py) needs a")
    print("working OpenGL context and isn't exercised by this simple preview.")
