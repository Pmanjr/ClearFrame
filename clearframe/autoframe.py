"""Auto-framing: MediaPipe face detection on a small copy, then a smoothed
crop window driven by a critically damped spring."""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .settings import AutoFrameSettings, models_dir

log = logging.getLogger("clearframe.autoframe")

FACE_MODEL = "blaze_face_short_range.tflite"


class Spring:
    """Critically damped spring, implicit integration (stable at any dt)."""

    __slots__ = ("x", "v")

    def __init__(self, x: float = 0.0):
        self.x, self.v = x, 0.0

    def step(self, target: float, omega: float, dt: float) -> float:
        f = 1.0 + 2.0 * dt * omega
        oo = omega * omega
        hoo = dt * oo
        hhoo = dt * hoo
        det_inv = 1.0 / (f + hhoo)
        x = (f * self.x + dt * self.v + hhoo * target) * det_inv
        self.v = (self.v + hoo * (target - self.x)) * det_inv
        self.x = x
        return x


class AutoFramer:
    DETECT_EVERY = 3            # frames
    LOST_AFTER = 1.5            # seconds without a face -> ease back to full frame
    DET_W = 640

    def __init__(self):
        self._det = None
        self._failed = False
        self._ts = 0
        self._n = 0
        self._last_seen = 0.0
        self._last_t: float | None = None
        self.cx, self.cy, self.zoom = Spring(0.5), Spring(0.5), Spring(1.0)
        self._target = (0.5, 0.5, 1.0)
        self.face: tuple[float, float, float, float] | None = None   # normalised x, y, w, h

    def _detector(self):
        if self._det is None and not self._failed:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mpt
                from mediapipe.tasks.python import vision
                self._mp = mp
                path = models_dir() / FACE_MODEL
                opts = vision.FaceDetectorOptions(
                    base_options=mpt.BaseOptions(model_asset_buffer=path.read_bytes()),
                    running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.5)
                self._det = vision.FaceDetector.create_from_options(opts)
            except Exception as e:  # noqa: BLE001
                self._failed = True
                log.warning("Face detector unavailable (%s). Run scripts/download_models.py", e)
        return self._det

    def reset(self) -> None:
        self.cx, self.cy, self.zoom = Spring(0.5), Spring(0.5), Spring(1.0)
        self._target = (0.5, 0.5, 1.0)
        self._last_t = None

    def _detect(self, frame: np.ndarray) -> None:
        det = self._detector()
        if det is None:
            return
        h, w = frame.shape[:2]
        dw = self.DET_W
        dh = max(2, int(round(h * dw / w)))
        small = cv2.cvtColor(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._ts += 33
        res = det.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=small), self._ts)
        best = None
        for d in res.detections:
            b = d.bounding_box
            if best is None or b.width * b.height > best.width * best.height:
                best = b
        if best is None:
            self.face = None
            return
        self.face = (best.origin_x / dw, best.origin_y / dh, best.width / dw, best.height / dh)
        self._last_seen = time.perf_counter()

    def update(self, frame: np.ndarray, af: AutoFrameSettings, new_frame: bool = True) -> tuple[float, float, float]:
        """Advance tracking; returns (centre x, centre y, zoom) - centre is a
        fraction of the frame, zoom >= 1."""
        now = time.perf_counter()
        dt = 1 / 30 if self._last_t is None else float(np.clip(now - self._last_t, 1e-3, 0.25))
        self._last_t = now
        if new_frame:
            if self._n % self.DETECT_EVERY == 0:
                self._detect(frame)
            self._n += 1

        want_zoom = float(np.clip(af.zoom, 1.0, 4.0))
        tx, ty, tz = self._target
        if self.face is not None:
            fx, fy, fw, fh = self.face
            # keep the head from filling more than ~55% of the crop height
            z = float(np.clip(min(want_zoom, 0.48 / max(fh, 1e-3)), 1.0, want_zoom))
            cx, cy = fx + fw / 2, fy + fh * 0.5         # face centre; hair and shoulders fit around it
            # dead zone: ignore small shifts so the frame does not swim
            dead = 0.06 / z
            if abs(cx - tx) > dead or abs(cy - ty) > dead or abs(z - tz) > 0.08:
                tx, ty, tz = cx, cy, z
        elif now - self._last_seen > self.LOST_AFTER:
            tx, ty, tz = 0.5, 0.5, 1.0
        self._target = (tx, ty, tz)

        omega = 1.2 + float(np.clip(af.speed, 0.0, 1.0)) * 9.0
        z = max(1.0, self.zoom.step(tz, omega * 0.8, dt))
        half = 0.5 / z
        cx = float(np.clip(self.cx.step(tx, omega, dt), half, 1.0 - half))
        cy = float(np.clip(self.cy.step(ty, omega, dt), half, 1.0 - half))
        return cx, cy, z

    def close(self) -> None:
        if self._det is not None:
            try:
                self._det.close()
            except Exception:  # noqa: BLE001
                pass
            self._det = None
