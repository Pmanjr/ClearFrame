"""Noise reduction and sharpening at full resolution.

Small webcam sensors are noisy indoors. Averaging frames over time removes
that noise without blurring detail, as long as it is only done where nothing
moves - so the blend weight is driven by a local motion measure, with the
noise floor estimated from the image itself. Moving areas fall back to a
light chroma-only smoothing (colour speckle is the ugliest part of the
noise, and the eye does not miss chroma resolution).
"""
from __future__ import annotations

import cv2
import numpy as np

from .compositor import blend


class Denoiser:
    def __init__(self):
        self._acc: np.ndarray | None = None
        self._floor = 4.0

    def reset(self) -> None:
        self._acc = None

    def apply(self, frame: np.ndarray, strength: float, new_frame: bool = True) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            self._acc = None
            return frame
        if not new_frame and self._acc is not None and self._acc.shape == frame.shape:
            return self._acc
        cur = self._chroma_smooth(frame, strength)
        acc = self._acc
        if acc is None or acc.shape != cur.shape:
            self._acc = cur
            return cur
        H, W = cur.shape[:2]
        # local motion measure on a 1/4 grid (noise averages out, motion does not)
        d = cv2.cvtColor(cv2.absdiff(cur, acc), cv2.COLOR_BGR2GRAY)
        m = cv2.resize(d, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        m = cv2.blur(m, (5, 5))
        floor = float(np.percentile(m[::4, ::4], 30))
        self._floor += (float(np.clip(floor, 0.5, 20.0)) - self._floor) * 0.1
        t0, t1 = self._floor * 1.6 + 0.5, self._floor * 4.0 + 3.0
        w_min = 1.0 - 0.85 * strength              # weight of the new frame where static
        lut = np.clip((np.arange(256, dtype=np.float32) - t0) / (t1 - t0), 0.0, 1.0)
        lut = ((w_min + (1.0 - w_min) * lut) * 255.0 + 0.5).astype(np.uint8)
        w = cv2.LUT(m, lut)
        w = cv2.dilate(w, np.ones((3, 3), np.uint8))          # be generous around motion
        w = cv2.resize(w, (W, H), interpolation=cv2.INTER_LINEAR)
        out = blend(cur, acc, w)
        self._acc = out
        return out

    @staticmethod
    def _chroma_smooth(frame: np.ndarray, strength: float) -> np.ndarray:
        """Smooth only the colour-difference channels (through a 1/4-size
        copy); the full-resolution luma, where the detail lives, is put back
        untouched."""
        H, W = frame.shape[:2]
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        d = 4 if strength < 0.7 else 6
        small = cv2.resize(ycc, (max(2, W // d), max(2, H // d)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.mixChannels([ycc], [up], [0, 0])
        return cv2.cvtColor(up, cv2.COLOR_YCrCb2BGR)


def sharpen(frame: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask in a single 3x3 pass. amount 0..1."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return frame
    a = 1.6 * amount
    k = np.full((3, 3), -a / 9.0, np.float32)
    k[1, 1] += 1.0 + a
    return cv2.filter2D(frame, -1, k)
