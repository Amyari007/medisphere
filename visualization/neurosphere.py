"""
The NeuroSphere: a continuous visual readout of real hand openness — not
a binary "summoned/dismissed" prop. It smoothly grows/rotates/emits
particles as the hand opens, and smoothly collapses to a small glowing
point as the hand closes. Every dynamic here (radius, rotation speed,
particle emission rate) is driven by the actual openness value computed
in biomechanics/joints.py, not decorative randomness.
"""

import math
import random

import cv2
import numpy as np

MIN_RADIUS = 6       # collapsed "glowing point" state, hand closed
MAX_RADIUS = 120      # fully open
SMOOTHING = 6.0        # higher = snappier tracking of the target radius
BASE_COLOR = (255, 190, 60)   # cyan-ish core (BGR)
GLOW_COLOR = (255, 220, 150)


class _Particle:
    __slots__ = ("angle", "dist", "speed", "life", "max_life")

    def __init__(self, max_life):
        self.angle = random.uniform(0, 2 * math.pi)
        self.dist = 0.0
        self.speed = random.uniform(40.0, 90.0)  # px/s outward
        self.max_life = max_life
        self.life = max_life


class NeuroSphere:
    def __init__(self):
        self.radius = MIN_RADIUS
        self.rotation = 0.0
        self._particles = []

    def update(self, openness, dt, emit=True):
        openness = max(0.0, min(1.0, openness))
        target_radius = MIN_RADIUS + openness * (MAX_RADIUS - MIN_RADIUS)
        self.radius += (target_radius - self.radius) * min(1.0, dt * SMOOTHING)

        self.rotation = (self.rotation + dt * (20.0 + 80.0 * openness)) % 360.0

        if emit and openness > 0.2:
            emit_rate = openness * 12.0  # particles/sec
            expected = emit_rate * dt
            n_new = int(expected) + (1 if random.random() < (expected % 1.0) else 0)
            for _ in range(n_new):
                self._particles.append(_Particle(max_life=random.uniform(0.6, 1.1)))

        alive = []
        for p in self._particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.dist += p.speed * dt
            alive.append(p)
        self._particles = alive

    def draw(self, frame, center):
        cx, cy = center
        r = int(self.radius)

        # solid dark "stage" backdrop behind the sphere, so it reads clearly
        # against any real background (a real room, not a flat test color)
        # instead of blending invisibly into it
        stage_r = int(r * 2.1) + 20
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), stage_r, (10, 8, 8), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        # bloom: several soft, decreasing-alpha rings behind the core
        for i, (mult, alpha) in enumerate([(1.8, 0.14), (1.4, 0.22), (1.15, 0.32)]):
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), int(r * mult), GLOW_COLOR, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

        # particles, fading with remaining life
        for p in self._particles:
            fade = max(0.0, p.life / p.max_life)
            px = int(cx + math.cos(p.angle) * (r + p.dist))
            py = int(cy + math.sin(p.angle) * (r + p.dist))
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                size = max(1, int(3 * fade))
                color = tuple(int(c * fade) for c in GLOW_COLOR)
                cv2.circle(frame, (px, py), size, color, -1, cv2.LINE_AA)

        # core sphere fill
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, BASE_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)

        # rotating highlight arc, visualizes continuous rotation
        if r > MIN_RADIUS + 2:
            start = self.rotation
            end = start + 50
            cv2.ellipse(frame, (cx, cy), (r, r), 0, start, end, (255, 255, 255), 3, cv2.LINE_AA)
