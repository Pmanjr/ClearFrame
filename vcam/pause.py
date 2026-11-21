"""Pause Camera (freeze frame) and Pause Loop (ping-pong replay).

The loop buffer keeps the last N seconds of camera frames as JPEGs (encoded
on a worker thread). In loop mode they are replayed forwards then backwards,
so there is no visible seam, and they still go through the whole pipeline.
"""
from __future__ import annotations

import bisect
import collections
import queue
import threading
import time

import cv2
import numpy as np

LIVE, FREEZE, LOOP = "live", "freeze", "loop"


class LoopBuffer:
    QUALITY = 93

    def __init__(self):
        self._items: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self.seconds = 4.0

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="loop-encode", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            t, frame = item
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.QUALITY])
            if not ok:
                continue
            with self._lock:
                self._items.append((t, buf))
                while self._items and t - self._items[0][0] > self.seconds:
                    self._items.popleft()

    def push(self, frame: np.ndarray, seconds: float) -> None:
        self.seconds = max(0.5, float(seconds))
        self._ensure_thread()
        try:
            self._q.put_nowait((time.perf_counter(), frame))
        except queue.Full:
            pass                      # encoder busy: skip this frame

    def take(self) -> tuple[list[float], list[np.ndarray]]:
        with self._lock:
            items = list(self._items)
        if not items:
            return [], []
        t0 = items[0][0]
        return [t - t0 for t, _ in items], [b for _, b in items]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)


class PauseController:
    def __init__(self):
        self.mode = LIVE
        self.buffer = LoopBuffer()
        self._held: np.ndarray | None = None
        self._times: list[float] = []
        self._clip: list[np.ndarray] = []
        self._t0 = 0.0
        self._idx = -1
        self._decoded: np.ndarray | None = None
        self._last_live: np.ndarray | None = None

    def set_mode(self, mode: str) -> str:
        """Switch mode; returns the mode actually entered."""
        if mode == self.mode:
            return mode
        if mode == LOOP:
            self._times, self._clip = self.buffer.take()
            if len(self._clip) < 6:           # nothing recorded yet -> freeze instead
                mode = FREEZE
            else:
                self._t0, self._idx = time.perf_counter(), -1
        if mode == FREEZE:
            self._held = self._last_live
        if mode == LIVE:
            self._clip, self._times, self._decoded = [], [], None
            self.buffer.clear()
        self.mode = mode
        return mode

    def feed(self, frame: np.ndarray, new_frame: bool, loop_seconds: float,
             record: bool = True) -> tuple[np.ndarray, bool]:
        """Camera frame in -> frame the pipeline should use, and whether it is new."""
        if self.mode == LIVE:
            self._last_live = frame
            if new_frame and record:
                self.buffer.push(frame, loop_seconds)
            return frame, new_frame
        if self.mode == FREEZE:
            if self._held is None:
                self._held = frame
            return self._held, False
        # LOOP: ping-pong through the recorded clip on the wall clock
        dur = self._times[-1]
        if dur <= 0:
            return frame, new_frame
        t = (time.perf_counter() - self._t0) % (2 * dur)
        if t > dur:
            t = 2 * dur - t
        i = min(bisect.bisect_right(self._times, t), len(self._clip) - 1)
        if i != self._idx or self._decoded is None:
            img = cv2.imdecode(self._clip[i], cv2.IMREAD_COLOR)
            if img is not None:
                self._decoded, self._idx = img, i
                return img, True
        return (self._decoded if self._decoded is not None else frame), False

    def close(self) -> None:
        self.buffer.close()
