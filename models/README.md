Two files go in this folder (neither is committed to git — see .gitignore):

## 1. Hand landmark model (required)

    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

PowerShell:

    curl.exe -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## 2. Brain hologram mesh (optional)

Copy your `brain_hologram.glb` file into this folder. If present, MediSphere
renders it as a real 3D hologram via OpenGL. If absent (or if OpenGL/GLFW
can't get a working context on your machine), the app automatically falls
back to a procedural 2D "NeuroSphere" instead — no crash either way.
