"""Virtual camera output through pyvirtualcam.

- OBS Virtual Camera: default (BGR, no alpha).
- Unity Capture: used when RGBA is requested, for true transparency in OBS.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("vcam.output")

INSTALL_HINT = (
    "No virtual camera driver found. Install OBS Studio (ships 'OBS Virtual Camera') "
    "from https://obsproject.com, or Unity Capture from "
    "https://github.com/schellingb/UnityCapture (run Install.bat as admin). See README.")


class VirtualCamError(RuntimeError):
    pass


class VirtualCamOutput:
    def __init__(self, width: int, height: int, fps: int, backend: str = "auto", rgba: bool = False):
        import pyvirtualcam
        from pyvirtualcam import PixelFormat
        self.width, self.height, self.fps = width, height, fps
        self.rgba = False
        self.cam = None
        attempts: list[tuple[str, object]] = []
        if rgba:
            attempts.append(("unitycapture", PixelFormat.RGBA))
            if backend in ("auto", "obs"):
                attempts.append(("obs", PixelFormat.BGR))
        elif backend == "auto":
            attempts += [("obs", PixelFormat.BGR), ("unitycapture", PixelFormat.BGR)]
        else:
            attempts.append((backend, PixelFormat.BGR))
        errors = []
        for be, fmt in attempts:
            try:
                self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=fmt, backend=be)
                self.rgba = fmt == PixelFormat.RGBA
                self.backend = be
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{be}: {e}")
        if self.cam is None:
            raise VirtualCamError(INSTALL_HINT + " (" + " | ".join(errors) + ")")
        if rgba and not self.rgba:
            log.warning("RGBA requested but Unity Capture is not installed; "
                        "sending opaque frames to '%s' (Remove uses the fallback colour).",
                        self.cam.device)
        log.info("Virtual camera: '%s' %dx%d @ %d FPS (%s, %s)", self.cam.device, width, height,
                 fps, self.backend, "RGBA" if self.rgba else "BGR")

    @property
    def device(self) -> str:
        return self.cam.device if self.cam else ""

    def send(self, bgr: np.ndarray, rgba: np.ndarray | None = None) -> None:
        """Send one frame. `rgba` is used when the backend carries alpha."""
        if self.cam is None:
            return
        if self.rgba:
            frame = rgba if rgba is not None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        else:
            frame = bgr
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.cam.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:  # noqa: BLE001
                pass
            self.cam = None
            log.info("Virtual camera closed.")
