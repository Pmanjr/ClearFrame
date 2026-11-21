"""Splash / idle image shown when the camera is paused, asleep or off."""
from __future__ import annotations

import cv2
import numpy as np

from .compositor import place_media
from .media import imread_any


def builtin_splash(size: tuple[int, int], text: str = "Camera paused") -> np.ndarray:
    W, H = size
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    g = 0.5 * (1 - y) + 0.5 * x
    img = np.empty((H, W, 3), np.uint8)
    img[..., 0] = (38 + 30 * g).astype(np.uint8)
    img[..., 1] = (26 + 14 * g).astype(np.uint8)
    img[..., 2] = (22 + 10 * g).astype(np.uint8)
    # camera-off glyph
    s = H / 1080.0
    cx, cy = W // 2, int(H * 0.44)
    col = (235, 235, 235)
    bw, bh = int(150 * s), int(100 * s)
    cv2.rectangle(img, (cx - bw, cy - bh), (cx + int(bw * 0.45), cy + bh), col, max(2, int(10 * s)), cv2.LINE_AA)
    pts = np.array([[cx + int(bw * 0.45), cy - int(30 * s)], [cx + bw, cy - int(80 * s)],
                    [cx + bw, cy + int(80 * s)], [cx + int(bw * 0.45), cy + int(30 * s)]], np.int32)
    cv2.polylines(img, [pts], True, col, max(2, int(10 * s)), cv2.LINE_AA)
    cv2.line(img, (cx - int(190 * s), cy + int(150 * s)), (cx + int(190 * s), cy - int(150 * s)),
             (90, 90, 240), max(2, int(14 * s)), cv2.LINE_AA)
    scale = 1.6 * s
    th = max(1, int(3 * s))
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    cv2.putText(img, text, (cx - tw // 2, int(H * 0.70)), cv2.FONT_HERSHEY_DUPLEX, scale, col, th, cv2.LINE_AA)
    return img


class Splash:
    """Caches the splash at the current output size."""

    def __init__(self):
        self._key = None
        self._img: np.ndarray | None = None

    def get(self, size: tuple[int, int], custom_path: str = "", text: str = "Camera paused") -> np.ndarray:
        key = (size, custom_path, text)
        if key != self._key or self._img is None:
            img = None
            if custom_path:
                src = imread_any(custom_path)
                if src is not None:
                    img = np.ascontiguousarray(place_media(src, size, "fill"))
            self._img = img if img is not None else builtin_splash(size, text)
            self._key = key
        return self._img
