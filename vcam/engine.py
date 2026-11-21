"""The processing engine: capture -> segmentation -> composite -> virtual cam.

Runs on its own thread. The UI/CLI mutates `engine.settings` directly (plain
attribute writes) and calls `engine.request(...)` for changes that need a
device or model to be rebuilt.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import compositor as comp
from .capture import FileCapture, Negotiated, ThreadedCapture
from .devices import resolve_camera
from .media import MediaSource, open_media
from .output import VirtualCamError, VirtualCamOutput
from .segmentation import ModelMissing, Segmenter, create_segmenter
from .settings import Settings, parse_color, pictures_dir

log = logging.getLogger("vcam.engine")


@dataclass
class Stats:
    fps: float = 0.0
    infer_ms: float = 0.0        # model forward pass
    seg_ms: float = 0.0          # model + mask refinement
    frame_ms: float = 0.0        # whole pipeline per frame
    stages: dict = field(default_factory=dict)
    segmenter: str = ""
    provider: str = ""
    camera: str = ""
    capture: Negotiated = field(default_factory=Negotiated)
    output: str = ""
    status: str = "starting"
    warnings: list = field(default_factory=list)


class _Timer:
    """Per-stage timings, exponentially smoothed."""

    def __init__(self):
        self.avg: dict[str, float] = {}
        self._t = 0.0

    def start(self) -> None:
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        ms = (now - self._t) * 1000.0
        self._t = now
        prev = self.avg.get(name)
        self.avg[name] = ms if prev is None else prev * 0.9 + ms * 0.1


class Engine:
    def __init__(self, settings: Settings, input_file: str | None = None,
                 camera_index: int | None = None):
        self.settings = settings
        self.input_file = input_file
        self.camera_index = camera_index
        self.stats = Stats()
        self.cap = None
        self.seg: Segmenter | None = None
        self.out: VirtualCamOutput | None = None
        self.bg_source: MediaSource | None = None
        self.bg_layer = comp.BackgroundLayer()
        self._bg_path = None
        self._requests: set[str] = {"camera", "segmenter", "output"}
        self._req_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._latest_lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._latest_alpha: np.ndarray | None = None
        self._frame_count = 0
        self._alpha: np.ndarray | None = None
        self._timer = _Timer()
        self._warned: set[str] = set()
        self.on_error = None          # callback(str)

    # ------------------------------------------------------------------ API
    def request(self, *what: str) -> None:
        """Ask the engine thread to rebuild: 'camera', 'segmenter', 'output'."""
        with self._req_lock:
            self._requests.update(what)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """(composited BGR, alpha or None, frame counter) of the newest frame."""
        with self._latest_lock:
            return self._latest_bgr, self._latest_alpha, self._frame_count

    def snapshot(self, path: str | Path | None = None) -> Path | None:
        """Save the current composited frame as PNG (with alpha in Remove mode)."""
        bgr, alpha, _ = self.latest()
        if bgr is None:
            return None
        if path is None:
            path = pictures_dir() / time.strftime("vcam-%Y%m%d-%H%M%S.png")
        path = Path(path)
        img = bgr
        if alpha is not None and self.settings.background.mode == "remove" and self._latest_fg is not None:
            img = cv2.cvtColor(self._latest_fg, cv2.COLOR_BGR2BGRA)
            img[..., 3] = alpha
        img = img.copy()

        def _save():
            ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if ok:
                path.parent.mkdir(parents=True, exist_ok=True)
                buf.tofile(str(path))
                log.info("Snapshot saved: %s", path)
        threading.Thread(target=_save, daemon=True).start()
        return path

    _latest_fg: np.ndarray | None = None

    # ------------------------------------------------------------ internals
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)
            self.stats.warnings.append(msg)

    def _error(self, msg: str) -> None:
        log.error(msg)
        self.stats.status = msg
        if self.on_error:
            try:
                self.on_error(msg)
            except Exception:  # noqa: BLE001
                pass

    def _take_requests(self) -> set[str]:
        with self._req_lock:
            r, self._requests = self._requests, set()
            return r

    def _open_camera(self) -> bool:
        self._close_camera()
        cs = self.settings.camera
        try:
            if self.input_file:
                self.cap = FileCapture(self.input_file, cs.fps)
            else:
                dev = resolve_camera(cs.device_name, self.camera_index)
                if dev is None:
                    self._error("No physical camera found.")
                    return False
                self.cap = ThreadedCapture(dev.index, cs.width, cs.height, cs.fps,
                                           cs.fourcc, cs.backend, dev.name)
            neg = self.cap.open()
        except Exception as e:  # noqa: BLE001
            self.cap = None
            self._error(str(e))
            return False
        self.stats.camera = self.cap.name
        self.stats.capture = neg
        self.stats.warnings = [f"Capture: {w}" for w in neg.warnings]
        self._warned.clear()
        for name, value in (cs.uvc or {}).items():
            self.cap.set_prop(name, value)
        if self.seg:
            self.seg.reset()
        self._alpha = None
        self._last_id = 0
        return True

    def _close_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open_segmenter(self) -> None:
        if self.seg is not None:
            self.seg.close()
            self.seg = None
        try:
            self.seg = create_segmenter(self.settings.seg)
            self.stats.segmenter, self.stats.provider = self.seg.name, self.seg.provider
        except ModelMissing as e:
            self._error(str(e))
        except Exception as e:  # noqa: BLE001
            self._error(f"Segmenter failed to load: {e}")
        self._alpha = None

    def _open_output(self) -> None:
        if self.out is not None:
            self.out.close()
            self.out = None
        o = self.settings.output
        self.stats.output = ""
        if not o.enabled:
            return
        W, H = self.settings.output_size()
        try:
            self.out = VirtualCamOutput(W, H, o.fps, o.backend, o.rgba)
            self.stats.output = f"{self.out.device} {W}x{H}@{o.fps}" + (" RGBA" if self.out.rgba else "")
        except VirtualCamError as e:
            self._warn_once("vcam", str(e))
            self.stats.output = "no virtual camera driver"

    def _sync_background_source(self) -> None:
        bs = self.settings.background
        path = bs.replace_path if bs.mode == "replace" else None
        if path != self._bg_path:
            if self.bg_source is not None:
                self.bg_source.close()
            self.bg_source = open_media(path) if path else None
            self._bg_path = path
            self.bg_layer.invalidate()

    # -------------------------------------------------------------- process
    def _fit_to_output(self, frame: np.ndarray, alpha: np.ndarray | None, size: tuple[int, int]):
        """Centre-crop to the output aspect, then scale to output size.
        Downscale only; an upscale means the source is below output res."""
        W, H = size
        h, w = frame.shape[:2]
        if w * H != h * W:                       # aspect mismatch -> centre crop
            if w * H > h * W:
                cw = int(round(h * W / H)) & ~1
                x0 = (w - cw) // 2
                frame = frame[:, x0:x0 + cw]
                alpha = alpha[:, x0:x0 + cw] if alpha is not None else None
            else:
                ch = int(round(w * H / W)) & ~1
                y0 = (h - ch) // 2
                frame = frame[y0:y0 + ch]
                alpha = alpha[y0:y0 + ch] if alpha is not None else None
            h, w = frame.shape[:2]
        if (w, h) != (W, H):
            if w < W:
                self._warn_once("upscale", f"Source {w}x{h} is below output {W}x{H}: upscaling. "
                                           "Lower the output resolution for a sharper picture.")
            interp = cv2.INTER_AREA if w > W else cv2.INTER_CUBIC
            frame = cv2.resize(frame, (W, H), interpolation=interp)
            if alpha is not None:
                alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_AREA if w > W else cv2.INTER_LINEAR)
        return frame, alpha

    def process(self, frame: np.ndarray, new_frame: bool = True):
        """One frame through the pipeline. Returns (bgr, rgba_or_None, alpha_or_None, fg)."""
        s = self.settings
        bs = s.background
        t = self._timer
        t.start()
        mode = bs.mode
        size = s.output_size()

        alpha = None
        if mode != "original" and self.seg is not None:
            if new_frame or self._alpha is None or self._alpha.shape[:2] != frame.shape[:2]:
                self.seg.ema_strength, self.seg.edge_tighten = s.seg.ema, s.seg.edge_tighten
                self._alpha = self.seg.alpha(frame)
            alpha = self._alpha
        t.lap("segment")

        cs = s.camera
        fg = comp.adjust(frame, cs.brightness, cs.contrast, cs.saturation, cs.hue, cs.gamma)
        fg, alpha = self._fit_to_output(fg, alpha, size)
        t.lap("adjust+fit")

        rgba = None
        if alpha is None:
            bgr = fg
        elif mode == "blur":
            bgr = comp.blend(fg, comp.blur_background(fg, alpha, bs.blur_strength, bs.bokeh), alpha)
        elif mode == "replace":
            bgr = comp.blend(fg, self.bg_layer.render(self.bg_source, bs, size), alpha)
        elif mode == "color":
            bgr = comp.blend_color(fg, parse_color(bs.color), alpha)
        else:  # remove
            bgr = comp.blend_color(fg, parse_color(bs.remove_fallback_color), alpha)
            if self.out is not None and self.out.rgba:
                rgba = comp.to_rgba(fg, alpha)
        t.lap("composite")
        return bgr, rgba, alpha, fg

    # ----------------------------------------------------------------- loop
    def _run(self) -> None:
        log.info("Engine started.")
        self._last_id = 0
        next_due = time.perf_counter()
        fps_t0, fps_n = time.perf_counter(), 0
        try:
            while self._running:
                req = self._take_requests()
                if "segmenter" in req:
                    self._open_segmenter()
                if "camera" in req:
                    if not self._open_camera():
                        time.sleep(1.0)
                        self.request("camera")
                        continue
                if "output" in req:
                    self._open_output()
                self._sync_background_source()
                if self.cap is None:
                    time.sleep(0.05)
                    continue
                if getattr(self.cap, "failed", False):
                    self._error("Camera disconnected; retrying...")
                    self.request("camera")
                    continue

                period = 1.0 / max(1, self.settings.output.fps)
                frame, fid = self.cap.read(self._last_id, timeout=period)
                if frame is None:
                    continue
                now = time.perf_counter()
                if now < next_due - 0.35 * period:
                    if fid != self._last_id:
                        continue          # camera is faster than the output: drop
                    time.sleep(max(0.0, next_due - 0.35 * period - now))
                    continue
                next_due = max(next_due + period, now - period)
                new_frame = fid != self._last_id
                self._last_id = fid

                t_frame = time.perf_counter()
                bgr, rgba, alpha, fg = self.process(frame, new_frame)
                if self.out is not None:
                    self.out.send(bgr, rgba)
                    self._timer.lap("vcam send")
                with self._latest_lock:
                    self._latest_bgr, self._latest_alpha, self._latest_fg = bgr, alpha, fg
                    self._frame_count += 1

                st = self.stats
                ms = (time.perf_counter() - t_frame) * 1000.0
                st.frame_ms = ms if st.frame_ms == 0 else st.frame_ms * 0.9 + ms * 0.1
                if self.seg is not None and alpha is not None:
                    st.infer_ms = st.infer_ms * 0.9 + self.seg.infer_ms * 0.1 if st.infer_ms else self.seg.infer_ms
                    st.seg_ms = st.seg_ms * 0.9 + self.seg.total_ms * 0.1 if st.seg_ms else self.seg.total_ms
                st.stages = dict(self._timer.avg)
                st.capture = self.cap.negotiated
                st.status = "running"
                fps_n += 1
                dt = time.perf_counter() - fps_t0
                if dt >= 1.0:
                    st.fps = fps_n / dt
                    fps_t0, fps_n = time.perf_counter(), 0
        except Exception:  # noqa: BLE001
            log.exception("Engine crashed")
            self._error("Engine crashed; see log.")
        finally:
            self._close_camera()
            if self.out is not None:
                self.out.close()
                self.out = None
            if self.seg is not None:
                self.seg.close()
            if self.bg_source is not None:
                self.bg_source.close()
                self.bg_source, self._bg_path = None, None
            log.info("Engine stopped.")
