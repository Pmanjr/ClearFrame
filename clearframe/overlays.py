"""Logo / watermark overlay layers and the name-tag lower third."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .compositor import overlay_bgra
from .media import MediaSource, open_media
from .settings import NameTagSettings, OverlaySettings, parse_color

log = logging.getLogger("clearframe.overlays")


def anchor_pos(anchor: str, W: int, H: int, w: int, h: int, ox: float, oy: float) -> tuple[int, int]:
    """Top-left of a w x h box placed at one of the 9 anchors. Offsets are
    fractions of the frame, pointing inwards from edges (plain shift at centre)."""
    dx, dy = ox * W, oy * H
    if "left" in anchor:
        x = dx
    elif "right" in anchor:
        x = W - w - dx
    else:
        x = (W - w) / 2 + dx
    if "top" in anchor:
        y = dy
    elif "bottom" in anchor:
        y = H - h - dy
    else:
        y = (H - h) / 2 + dy
    return int(round(x)), int(round(y))


class OverlayRenderer:
    """Draws the enabled overlay layers in order (PNG / GIF / MP4...)."""

    def __init__(self):
        self._sources: dict[str, MediaSource | None] = {}
        self._sprites: dict[int, tuple[tuple, np.ndarray]] = {}

    def _source(self, path: str) -> MediaSource | None:
        if path not in self._sources:
            self._sources[path] = open_media(path)
        return self._sources[path]

    def _gc(self, layers: list[OverlaySettings]) -> None:
        live = {l.path for l in layers}
        for path in [p for p in self._sources if p not in live]:
            src = self._sources.pop(path)
            if src is not None:
                src.close()

    def draw(self, bgr: np.ndarray, layers: list[OverlaySettings],
             dst_alpha: np.ndarray | None = None, now: float | None = None) -> None:
        if len(self._sources) > len(layers):
            self._gc(layers)
        H, W = bgr.shape[:2]
        for i, l in enumerate(layers):
            if not l.enabled or not l.path or l.opacity <= 0:
                continue
            src = self._source(l.path)
            if src is None:
                continue
            img = src.frame(now)
            if img is None:
                continue
            tw = max(2, int(round(W * l.size)))
            th = max(2, int(round(img.shape[0] * tw / img.shape[1])))
            key = (l.path, src.version, tw, th, l.flip)
            cached = self._sprites.get(i)
            if cached is None or cached[0] != key:
                sprite = cv2.resize(img, (tw, th),
                                    interpolation=cv2.INTER_AREA if tw < img.shape[1] else cv2.INTER_CUBIC)
                if l.flip:
                    sprite = cv2.flip(sprite, 1)
                self._sprites[i] = cached = (key, np.ascontiguousarray(sprite))
            x, y = anchor_pos(l.anchor, W, H, tw, th, l.offset_x, l.offset_y)
            overlay_bgra(bgr, cached[1], x, y, float(l.opacity), dst_alpha)

    def close(self) -> None:
        for src in self._sources.values():
            if src is not None:
                src.close()
        self._sources.clear()
        self._sprites.clear()


# ---------------------------------------------------------------------------
def _font(bold: bool, size: int):
    from PIL import ImageFont
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_nametag(nt: NameTagSettings, W: int, H: int) -> np.ndarray | None:
    """Lower-third sprite (BGRA) sized for a WxH frame."""
    from PIL import Image, ImageDraw, ImageFilter
    name, title = nt.name.strip(), nt.title.strip()
    if not name and not title:
        return None
    k = H / 1080.0 * max(0.4, nt.scale)
    f_name, f_title = _font(True, max(8, int(46 * k))), _font(False, max(6, int(30 * k)))
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    nb = probe.textbbox((0, 0), name or " ", font=f_name)
    tb = probe.textbbox((0, 0), title or " ", font=f_title)
    name_h = int(56 * k) if name else 0
    title_h = int(40 * k) if title else 0
    pad_x, pad_y, bar = int(26 * k), int(16 * k), int(10 * k)
    text_w = max(nb[2] if name else 0, tb[2] if title else 0)
    w = text_w + pad_x * 2 + (bar if nt.style == "bar" else 0)
    h = name_h + title_h + pad_y * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def rgb(c):
        b, g, r = parse_color(c)
        return r, g, b

    bg = rgb(nt.bg_color) + (int(255 * float(np.clip(nt.bg_opacity, 0, 1))),)
    accent, fg = rgb(nt.accent) + (255,), rgb(nt.text_color) + (255,)
    x_text = pad_x
    if nt.style == "bar":
        d.rectangle((0, 0, w, h), fill=bg)
        d.rectangle((0, 0, bar, h), fill=accent)
        x_text += bar
    elif nt.style == "pill":
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=bg)
        d.rounded_rectangle((pad_x, h - int(9 * k), pad_x + int(text_w * 0.35), h - int(5 * k)),
                            radius=int(2 * k) + 1, fill=accent)
        x_text = pad_x + int(8 * k)
    else:  # minimal: soft shadow under plain text
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        if name:
            sd.text((x_text + 2, pad_y + 2), name, font=f_name, fill=(0, 0, 0, 220))
        if title:
            sd.text((x_text + 2, pad_y + name_h + 2), title, font=f_title, fill=(0, 0, 0, 220))
        img = shadow.filter(ImageFilter.GaussianBlur(max(1.0, 3 * k)))
        d = ImageDraw.Draw(img)
    if name:
        d.text((x_text, pad_y - int(4 * k)), name, font=f_name, fill=fg)
    if title:
        d.text((x_text, pad_y + name_h - int(2 * k)), title, font=f_title,
               fill=fg[:3] + (220,) if nt.style != "minimal" else fg)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)


class NameTagRenderer:
    def __init__(self):
        self._key = None
        self._sprite: np.ndarray | None = None

    def draw(self, bgr: np.ndarray, nt: NameTagSettings, dst_alpha: np.ndarray | None = None) -> None:
        if not nt.enabled:
            return
        H, W = bgr.shape[:2]
        key = (W, H, nt.name, nt.title, nt.style, nt.accent, nt.text_color, nt.bg_color,
               nt.bg_opacity, nt.scale)
        if key != self._key:
            try:
                self._sprite = render_nametag(nt, W, H)
            except Exception as e:  # noqa: BLE001
                log.warning("Name tag render failed: %s", e)
                self._sprite = None
            self._key = key
        if self._sprite is None:
            return
        h, w = self._sprite.shape[:2]
        x, y = anchor_pos(nt.anchor, W, H, w, h, 0.04, 0.07)
        overlay_bgra(bgr, self._sprite, x, y, 1.0, dst_alpha)
