"""Enhance Lighting: auto gamma, CLAHE on luma, auto white balance.

Statistics and the local-contrast map are computed on a small copy and
smoothed over time; the correction is then applied to the full-resolution
frame as a per-channel LUT plus a smooth gain map, so no detail is lost.
"""
from __future__ import annotations

import cv2
import numpy as np


class LightingEnhancer:
    SMALL_W = 320
    SMOOTH = 0.08               # temporal smoothing of the estimates

    def __init__(self):
        self._gamma = 1.0
        self._wb = np.ones(3, np.float32)
        self._gain: np.ndarray | None = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._init = False

    def reset(self) -> None:
        self._init = False
        self._gain = None

    def apply(self, frame: np.ndarray, strength: float, update: bool = True) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return frame
        H, W = frame.shape[:2]
        sw = self.SMALL_W
        sh = max(2, int(round(H * sw / W)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        if update or not self._init:
            k = 1.0 if not self._init else self.SMOOTH
            # --- auto white balance (grey world on mid-tones, limited range)
            mid = cv2.inRange(small, (30, 30, 30), (225, 225, 225))
            if cv2.countNonZero(mid) > small.shape[0] * small.shape[1] * 0.1:
                means = np.array(cv2.mean(small, mask=mid)[:3], np.float32)
                wb = np.clip(means.mean() / np.maximum(means, 1.0), 0.75, 1.35)
                self._wb += (wb - self._wb) * k
            # --- auto gamma: pull mean luma towards mid-grey
            y = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mean = float(np.clip(cv2.mean(y)[0] / 255.0, 0.02, 0.98))
            gamma = float(np.clip(np.log(0.46) / np.log(mean), 0.55, 1.6))   # out = in ** gamma
            self._gamma += (gamma - self._gamma) * k
            self._init = True

        # --- global LUT (white balance + gamma), blended by strength
        x = np.arange(256, dtype=np.float32) / 255.0
        g = 1.0 + (self._gamma - 1.0) * strength
        lut = np.empty((256, 1, 3), np.uint8)
        for c in range(3):
            gain = 1.0 + (float(self._wb[c]) - 1.0) * strength
            lut[:, 0, c] = np.clip(np.power(np.clip(x * gain, 0, 1), g) * 255.0 + 0.5, 0, 255)
        out = cv2.LUT(frame, lut)

        # --- local contrast: CLAHE on small luma -> smooth gain map -> full res
        if update or self._gain is None or self._gain.shape != (sh, sw):
            ys = cv2.cvtColor(cv2.LUT(small, lut), cv2.COLOR_BGR2GRAY)
            eq = self._clahe.apply(ys)
            gain = (eq.astype(np.float32) + 8.0) / (ys.astype(np.float32) + 8.0)
            gain = cv2.GaussianBlur(np.clip(gain, 0.7, 1.8), (0, 0), 3.0)
            if self._gain is None or self._gain.shape != gain.shape:
                self._gain = gain
            else:
                self._gain += (gain - self._gain) * 0.25
        gmap = 1.0 + (self._gain - 1.0) * (0.75 * strength)
        g8 = cv2.convertScaleAbs(gmap, alpha=64.0)                 # 64 == 1.0x
        g8 = cv2.resize(g8, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.multiply(out, cv2.merge((g8, g8, g8)), scale=1.0 / 64.0)
