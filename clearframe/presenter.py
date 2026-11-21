"""Presenter mode: capture a monitor on a background thread so it can be
shown behind the matted person."""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("clearframe.presenter")


def list_monitors() -> list[tuple[int, str]]:
    try:
        import mss
        with mss.mss() as sct:
            return [(i, f"Monitor {i} ({m['width']}x{m['height']})")
                    for i, m in enumerate(sct.monitors) if i > 0]
    except Exception:  # noqa: BLE001
        return [(1, "Monitor 1")]


class ScreenGrabber:
    FPS = 20

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._running = False
        self._monitor = 1
        self._frame: np.ndarray | None = None

    def start(self, monitor: int = 1) -> None:
        self._monitor = max(1, int(monitor))
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="screen", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            import mss
        except ImportError:
            log.warning("mss is not installed; presenter mode unavailable.")
            return
        period = 1.0 / self.FPS
        with mss.mss() as sct:        # must be created on the thread that uses it
            while self._running:
                t = time.perf_counter()
                try:
                    mons = sct.monitors
                    mon = mons[self._monitor] if self._monitor < len(mons) else mons[1]
                    shot = np.asarray(sct.grab(mon))            # BGRA
                    self._frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                except Exception as e:  # noqa: BLE001
                    log.warning("Screen capture failed: %s", e)
                    time.sleep(1.0)
                time.sleep(max(0.0, period - (time.perf_counter() - t)))

    def frame(self) -> np.ndarray | None:
        return self._frame

    def stop(self) -> None:
        self._running = False
        self._thread = None
        self._frame = None
