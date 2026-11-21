"""Threaded camera capture.

The core quality rule lives here: request MJPG at native 1920x1080 explicitly
(USB 2.0 webcams silently fall back to 720p or lower in YUY2), then log what
the driver actually negotiated and warn loudly on any mismatch.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("clearframe.capture")

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}

# Hardware UVC controls exposed through OpenCV.
UVC_PROPS = {
    "zoom": cv2.CAP_PROP_ZOOM,
    "pan": cv2.CAP_PROP_PAN,
    "tilt": cv2.CAP_PROP_TILT,
    "focus": cv2.CAP_PROP_FOCUS,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "hw_brightness": cv2.CAP_PROP_BRIGHTNESS,
    "hw_contrast": cv2.CAP_PROP_CONTRAST,
    "hw_saturation": cv2.CAP_PROP_SATURATION,
    "hw_sharpness": cv2.CAP_PROP_SHARPNESS,
    "backlight": cv2.CAP_PROP_BACKLIGHT,
}


def fourcc_to_str(v: float) -> str:
    i = int(v)
    s = "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))
    return s if s.strip("\x00").isprintable() and s.strip("\x00") else f"0x{i & 0xFFFFFFFF:08x}"


@dataclass
class Negotiated:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    measured_fps: float = 0.0
    ok: bool = False
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        m = f", measured {self.measured_fps:.1f}" if self.measured_fps else ""
        return f"{self.width}x{self.height} @ {self.fps:.0f} FPS{m} [{self.fourcc}]"


class ThreadedCapture:
    """Grabs frames on its own thread; read() never blocks on the device."""

    def __init__(self, index: int, width=1920, height=1080, fps=30,
                 fourcc="MJPG", backend="dshow", name: str = ""):
        self.index, self.name = index, name or f"Camera {index}"
        self.req = (width, height, fps, fourcc.upper())
        self.backend = backend
        self.negotiated = Negotiated()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap_lock = threading.Lock()   # guards property get/set vs read
        self.failed = False

    # ------------------------------------------------------------------
    def open(self) -> Negotiated:
        w, h, fps, fcc = self.req
        cap = cv2.VideoCapture(self.index, BACKENDS.get(self.backend, cv2.CAP_DSHOW))
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.index} ({self.name}). "
                "Is another app (Zoom, Camera, XSplit VCam...) holding it?")
        # Order matters on DirectShow: FOURCC first, then size, then FPS,
        # then FOURCC again because some drivers reset it on a size change.
        code = cv2.VideoWriter_fourcc(*fcc)
        cap.set(cv2.CAP_PROP_FOURCC, code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, code)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {self.index} ({self.name}) opened but returned no frames.")
        n = Negotiated(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            fourcc=fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        )
        # trust the real frame over the property
        n.height, n.width = frame.shape[:2]
        warns = []
        if (n.width, n.height) != (w, h):
            warns.append(f"resolution is {n.width}x{n.height}, requested {w}x{h}")
        if n.fps and abs(n.fps - fps) > 1.0:
            warns.append(f"FPS is {n.fps:.0f}, requested {fps}")
        if n.fourcc.upper() != fcc:
            warns.append(f"pixel format is {n.fourcc}, requested {fcc}")
        n.warnings = tuple(warns)
        n.ok = not warns
        self.negotiated = n
        self._cap = cap
        self._frame, self._frame_id = frame, 1

        log.info("Camera '%s' (index %d, %s): requested %dx%d @ %d FPS [%s]",
                 self.name, self.index, self.backend, w, h, fps, fcc)
        log.info("Camera negotiated: %s", n.summary())
        if warns:
            bar = "!" * 72
            log.warning(bar)
            log.warning("!! CAPTURE IS NOT WHAT WAS REQUESTED:")
            for wmsg in warns:
                log.warning("!!   - %s", wmsg)
            log.warning("!! The image will be softer than it should be. Check the USB port "
                        "(use USB 3), close other camera apps, or try --cam-backend msmf.")
            log.warning(bar)

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        return n

    def _loop(self) -> None:
        t0, count, fails, windows = time.perf_counter(), 0, 0, 0
        while self._running:
            with self._cap_lock:
                ok = self._cap.grab() if self._cap is not None else False
                frame = None
                if ok:
                    ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    log.error("Camera stopped delivering frames.")
                    self.failed = True
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            count += 1
            dt = time.perf_counter() - t0
            if dt >= 2.0:
                windows += 1
                self.negotiated.measured_fps = count / dt
                if windows == 2:        # first window includes sensor warm-up
                    log.info("Camera measured delivery rate: %.1f FPS", self.negotiated.measured_fps)
                    if self.negotiated.measured_fps < self.req[2] * 0.8:
                        log.warning("!! Camera delivers only %.1f FPS (requested %d). "
                                    "Low light forces long exposures; add light or lower exposure.",
                                    self.negotiated.measured_fps, self.req[2])
                t0, count = time.perf_counter(), 0

    # ------------------------------------------------------------------
    def read(self, last_id: int = 0, timeout: float = 1.0):
        """Return (frame, frame_id). Waits up to `timeout` for a frame newer
        than `last_id`; returns the latest frame either way (never copies)."""
        with self._cond:
            if self._frame_id <= last_id:
                self._cond.wait(timeout)
            return self._frame, self._frame_id

    def set_prop(self, name: str, value: float) -> bool:
        prop = UVC_PROPS.get(name)
        if prop is None or self._cap is None:
            return False
        with self._cap_lock:
            return bool(self._cap.set(prop, float(value)))

    def get_prop(self, name: str) -> float | None:
        prop = UVC_PROPS.get(name)
        if prop is None or self._cap is None:
            return None
        with self._cap_lock:
            v = self._cap.get(prop)
        return None if v == -1 else float(v)

    def open_driver_dialog(self) -> None:
        """DirectShow's native camera property page (all UVC controls)."""
        if self._cap is not None:
            with self._cap_lock:
                self._cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        log.info("Camera '%s' released.", self.name)

    @property
    def is_open(self) -> bool:
        return self._cap is not None and not self.failed


class FileCapture:
    """Image / video file standing in for a camera (testing, demos)."""

    def __init__(self, path: str, fps: float = 30.0):
        self.path, self.name, self.index = path, path, -1
        self.failed = False
        self._img = cv2.imread(path, cv2.IMREAD_COLOR)
        self._cap = None
        self._fps = fps
        if self._img is None:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open input file {path}")
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or fps
        self._id = 0
        self._next = time.perf_counter()
        self.negotiated = Negotiated()

    def open(self) -> Negotiated:
        frame, _ = self.read()
        h, w = frame.shape[:2]
        self.negotiated = Negotiated(w, h, self._fps, "FILE", self._fps, True)
        log.info("Input file %s: %s", self.path, self.negotiated.summary())
        return self.negotiated

    def read(self, last_id: int = 0, timeout: float = 1.0):
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, timeout))
        self._next = max(self._next + 1.0 / self._fps, time.perf_counter())
        if self._img is not None:
            frame = self._img
        else:
            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            self._last = frame
        self._id += 1
        return frame, self._id

    def set_prop(self, name, value): return False
    def get_prop(self, name): return None
    def open_driver_dialog(self): pass

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @property
    def is_open(self) -> bool:
        return True
