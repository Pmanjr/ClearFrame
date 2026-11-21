"""Media sources for backgrounds and overlays: still images (JPG/PNG/WEBP),
animated GIF/WEBP, and looped muted video (MP4/WEBM) synced to its own FPS."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("clearframe.media")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ANIM_EXT = {".gif"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
MEDIA_EXT = IMAGE_EXT | ANIM_EXT | VIDEO_EXT


def is_media(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXT


def imread_any(path: str | Path) -> np.ndarray | None:
    """Unicode-path-safe imread keeping alpha (BGR or BGRA uint8)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / float(np.iinfo(img.dtype).max))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


class MediaSource:
    """frame(now) returns the BGR/BGRA frame to show at wall-clock `now`.
    `version` changes whenever the returned frame changes (for caching)."""

    animated = False

    def __init__(self, path: str):
        self.path = path
        self.version = 0

    def frame(self, now: float | None = None) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class StillSource(MediaSource):
    def __init__(self, path: str, img: np.ndarray):
        super().__init__(path)
        self._img = img

    def frame(self, now=None):
        return self._img


class AnimatedSource(MediaSource):
    """GIF / animated WEBP decoded up front with per-frame durations."""

    animated = True
    MAX_BYTES = 600 * 1024 * 1024

    def __init__(self, path: str, frames: list[np.ndarray], durations: list[float]):
        super().__init__(path)
        self._frames = frames
        self._ends = np.cumsum(durations)
        self._total = float(self._ends[-1])
        self._t0 = time.perf_counter()

    def frame(self, now=None):
        now = time.perf_counter() if now is None else now
        t = (now - self._t0) % self._total
        i = int(np.searchsorted(self._ends, t, side="right"))
        i = min(i, len(self._frames) - 1)
        self.version = i
        return self._frames[i]


class VideoSource(MediaSource):
    """Looped video, advanced by wall clock at the file's own frame rate."""

    animated = True

    def __init__(self, path: str):
        super().__init__(path)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self._t0: float | None = None
        self._idx = -1           # index of the frame held in _cur
        self._cur: np.ndarray | None = None

    def _rewind(self) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._idx = -1

    def frame(self, now=None):
        now = time.perf_counter() if now is None else now
        if self._t0 is None:
            self._t0 = now
        want = int((now - self._t0) * self.fps)
        if self._cur is None:
            want = 0
        behind = want - self._idx
        if behind <= 0:
            return self._cur
        # drop frames we are late for (cheap grab), decode only the last
        for _ in range(min(behind - 1, 8)):
            if not self._cap.grab():
                break
        ok, f = self._cap.read()
        if not ok or f is None:               # end of file -> loop
            self._rewind()
            self._t0 = now
            ok, f = self._cap.read()
            if not ok or f is None:
                return self._cur
            self._idx = 0
        else:
            self._idx = want
        self._cur = f
        self.version += 1
        return f

    def close(self) -> None:
        self._cap.release()


def _load_animated(path: str) -> MediaSource | None:
    from PIL import Image, ImageSequence
    try:
        im = Image.open(path)
    except Exception:  # noqa: BLE001
        return None
    n = getattr(im, "n_frames", 1)
    if n <= 1:
        return None
    frames, durs, total = [], [], 0
    for fr in ImageSequence.Iterator(im):
        rgba = np.asarray(fr.convert("RGBA"))
        has_alpha = rgba[..., 3].min() < 255
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGBA2BGR)
        frames.append(bgr)
        durs.append(max(0.02, fr.info.get("duration", 100) / 1000.0))
        total += bgr.nbytes
        if total > AnimatedSource.MAX_BYTES:
            log.warning("%s is very large; truncated to %d frames", path, len(frames))
            break
    # keep channel count consistent across frames
    if len({f.shape[2] for f in frames}) > 1:
        frames = [f if f.shape[2] == 4 else cv2.cvtColor(f, cv2.COLOR_BGR2BGRA) for f in frames]
    return AnimatedSource(path, frames, durs)


def open_media(path: str) -> MediaSource | None:
    """Open any supported file. Returns None (and logs) on failure."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("Media file not found: %s", path)
        return None
    ext = p.suffix.lower()
    try:
        if ext in VIDEO_EXT:
            return VideoSource(str(p))
        if ext in ANIM_EXT or ext == ".webp":
            src = _load_animated(str(p))
            if src is not None:
                return src
        img = imread_any(p)
        if img is None:
            log.warning("Could not decode %s", path)
            return None
        return StillSource(str(p), img)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not open %s: %s", path, e)
        return None


def thumbnail(path: str, size=(160, 90)) -> np.ndarray | None:
    """BGR thumbnail (centre-cropped to `size`) for the library grid."""
    p = Path(path)
    img = None
    if p.suffix.lower() in VIDEO_EXT:
        cap = cv2.VideoCapture(str(p))
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if n > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.1))
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None
    elif p.suffix.lower() in ANIM_EXT:
        from PIL import Image
        try:
            img = cv2.cvtColor(np.asarray(Image.open(str(p)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            return None
    else:
        img = imread_any(p)
    if img is None:
        return None
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    w, h = size
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(w, int(round(img.shape[1] * s))), max(h, int(round(img.shape[0] * s)))),
                   interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return np.ascontiguousarray(r[y:y + h, x:x + w])
