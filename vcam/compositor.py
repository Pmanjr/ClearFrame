"""Full-resolution compositing. All hot paths use OpenCV's SIMD/multithreaded
uint8 kernels; the person's pixels pass through bit-exact where alpha is 255.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

from .media import MediaSource
from .settings import BackgroundSettings


# ---------------------------------------------------------------------------
# colour adjustments (shared by camera controls and the background layer)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _tone_lut(brightness: float, contrast: float, gamma: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    if gamma != 1.0:
        x = np.power(x, 1.0 / max(gamma, 0.05))
    x = (x - 0.5) * (1.0 + contrast / 100.0) + 0.5 + brightness / 200.0
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def _hs_lut(hue: float, saturation: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    lut = np.empty((256, 1, 3), np.uint8)
    lut[:, 0, 0] = np.mod(x + hue / 2.0, 180.0).astype(np.uint8)       # OpenCV hue is 0..179
    lut[:, 0, 1] = np.clip(x * (1.0 + saturation / 100.0), 0, 255).astype(np.uint8)
    lut[:, 0, 2] = x.astype(np.uint8)
    return lut


def adjust(img: np.ndarray, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0,
           gamma=1.0) -> np.ndarray:
    """Brightness/contrast (-100..100), saturation (-100..100), hue (deg), gamma.
    Returns the input untouched when everything is neutral."""
    if brightness or contrast or gamma != 1.0:
        img = cv2.LUT(img, _tone_lut(float(brightness), float(contrast), float(gamma)))
    if saturation or hue:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.LUT(hsv, _hs_lut(float(hue), float(saturation)))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------
def blend(fg: np.ndarray, bg: np.ndarray, a8: np.ndarray) -> np.ndarray:
    """out = fg*a + bg*(1-a), uint8, full resolution."""
    a3 = cv2.merge((a8, a8, a8))
    out = cv2.multiply(fg, a3, scale=1.0 / 255.0)
    cv2.bitwise_not(a3, dst=a3)
    cv2.add(out, cv2.multiply(bg, a3, scale=1.0 / 255.0), dst=out)
    return out


def blend_color(fg: np.ndarray, color_bgr: tuple[int, int, int], a8: np.ndarray) -> np.ndarray:
    bg = np.empty_like(fg)
    bg[:] = color_bgr
    return blend(fg, bg, a8)


def to_rgba(fg_bgr: np.ndarray, a8: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2RGBA)
    rgba[..., 3] = a8
    return rgba


def overlay_bgra(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float = 1.0,
                 dst_alpha: np.ndarray | None = None) -> None:
    """Alpha-blend BGR/BGRA `src` onto `dst` in place at (x, y), clipped.
    If dst_alpha is given (RGBA output), the overlay makes it opaque there."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = dst[y0:y1, x0:x1]
    if s.shape[2] == 4:
        a = s[..., 3]
        if opacity < 1.0:
            a = cv2.convertScaleAbs(a, alpha=opacity)
        roi[:] = blend(np.ascontiguousarray(s[..., :3]), roi, np.ascontiguousarray(a))
    else:
        if opacity < 1.0:
            cv2.addWeighted(s, opacity, roi, 1.0 - opacity, 0.0, dst=roi)
        else:
            roi[:] = s
        a = None
    if dst_alpha is not None:
        ra = dst_alpha[y0:y1, x0:x1]
        if a is None:
            ra[:] = max(int(255 * opacity), 0)
        else:
            cv2.max(ra, a, dst=ra)


# ---------------------------------------------------------------------------
# blur
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def _disc(radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)).astype(np.float32)
    return k / k.sum()


def blur_strength_to_sigma(strength: float, width: int) -> float:
    s = float(np.clip(strength, 0, 100)) / 100.0
    return 1.5 + (s ** 1.4) * 0.035 * width


def blur_background(frame: np.ndarray, a8: np.ndarray | None, strength: float,
                    bokeh: bool = False) -> np.ndarray:
    """Blurred copy of `frame` for use as the background layer.

    The blur is foreground-aware: pixels belonging to the person are excluded
    (normalised convolution), so there is no coloured halo of the person
    smeared into the background around their outline. Works at reduced
    resolution - it is only the background, and it is being blurred anyway.
    """
    H, W = frame.shape[:2]
    sigma_full = blur_strength_to_sigma(strength, W)
    ds = int(np.clip(2 ** int(np.floor(np.log2(max(sigma_full / 3.0, 1.0)))), 2, 8))
    sw, sh = max(W // ds, 8), max(H // ds, 8)
    sigma = sigma_full / ds
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)

    if a8 is not None:
        am = cv2.resize(a8, (sw, sh), interpolation=cv2.INTER_AREA)
        am = cv2.dilate(am, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        w = (255.0 - am.astype(np.float32)) * (1.0 / 255.0)
        w += 1e-3
    else:
        w = np.ones((sh, sw), np.float32)

    if bokeh:
        lum = small.max(axis=2) * (1.0 / 255.0)
        w = w * (0.25 + 6.0 * lum ** 4)        # highlights bloom into discs
        kern = _disc(max(1, int(round(sigma * 1.5))))
        num = cv2.filter2D(small * w[..., None], -1, kern, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(w, -1, kern, borderType=cv2.BORDER_REFLECT)
    else:
        num = cv2.GaussianBlur(small * w[..., None], (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    out = num / np.maximum(den, 1e-6)[..., None]
    out = cv2.convertScaleAbs(out)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC if ds > 2 else cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# background layer (replace mode)
# ---------------------------------------------------------------------------
def place_media(img: np.ndarray, size: tuple[int, int], fit: str = "fill", zoom: float = 1.0,
                move_x: float = 0.0, move_y: float = 0.0, flip: bool = False) -> np.ndarray:
    """Scale `img` (BGR) to the WxH canvas using fit/fill + zoom + offset."""
    W, H = size
    if img.shape[2] == 4:
        # flatten transparency over black
        a = np.ascontiguousarray(img[..., 3])
        bgr = np.ascontiguousarray(img[..., :3])
        img = cv2.multiply(bgr, cv2.merge((a, a, a)), scale=1.0 / 255.0)
    if flip:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    s = (max if fit == "fill" else min)(W / w, H / h) * max(zoom, 0.05)
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (sw, sh) != (w, h):
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
    x0 = int(round(W / 2 + move_x * W - sw / 2))
    y0 = int(round(H / 2 + move_y * H - sh / 2))
    if x0 == 0 and y0 == 0 and (sw, sh) == (W, H):
        return img
    canvas = np.zeros((H, W, 3), np.uint8)
    dx0, dy0, dx1, dy1 = max(x0, 0), max(y0, 0), min(x0 + sw, W), min(y0 + sh, H)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = img[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    return canvas


class BackgroundLayer:
    """Renders the replacement background at output size, caching stills."""

    def __init__(self):
        self._key = None
        self._cached: np.ndarray | None = None

    def invalidate(self) -> None:
        self._key = None

    def render(self, src: MediaSource | None, bs: BackgroundSettings,
               size: tuple[int, int], now: float | None = None) -> np.ndarray:
        W, H = size
        if src is None:
            if self._key != ("empty", size):
                self._cached = np.full((H, W, 3), 24, np.uint8)
                self._key = ("empty", size)
            return self._cached
        img = src.frame(now)
        if img is None:
            return self.render(None, bs, size)
        key = (id(src), src.version, size, bs.fit, bs.zoom, bs.move_x, bs.move_y, bs.flip,
               bs.brightness, bs.contrast, bs.hue, bs.saturation, bs.replace_blur)
        if key == self._key and self._cached is not None:
            return self._cached
        out = place_media(img, size, bs.fit, bs.zoom, bs.move_x, bs.move_y, bs.flip)
        out = adjust(out, bs.brightness, bs.contrast, bs.saturation, bs.hue)
        if bs.replace_blur > 0:
            out = blur_background(out, None, bs.replace_blur)
        self._key, self._cached = key, out
        return out
