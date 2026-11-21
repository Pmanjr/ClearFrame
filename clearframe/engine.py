"""The processing engine: capture -> segmentation -> composite -> virtual cam.

Runs on its own thread. The UI/CLI mutates `engine.settings` directly (plain
attribute writes) and calls `engine.request(...)` for changes that need a
device or model to be rebuilt.

Per-frame order:
  camera frame -> pause (freeze / loop) -> matting on the raw frame
  -> auto-framing tracker -> look (camera controls, lighting, LUT)
  -> crop + fit to output size -> background composite
  -> overlays + name tag -> virtual camera / preview
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
from . import pause as pause_mod
from .autoframe import AutoFramer
from .capture import FileCapture, Negotiated, ThreadedCapture
from .denoise import Denoiser, sharpen
from .devices import resolve_camera
from .lighting import LightingEnhancer
from .media import MediaSource, open_media
from .output import VirtualCamError, VirtualCamOutput
from .overlays import NameTagRenderer, OverlayRenderer, anchor_pos
from .pause import PauseController
from .segmentation import ModelMissing, Segmenter, create_segmenter
from .settings import Settings, parse_color, pictures_dir
from .splash import Splash

log = logging.getLogger("clearframe.engine")


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
    pause: str = pause_mod.LIVE
    warnings: list = field(default_factory=list)   # sticky (capture / driver)
    notes: dict = field(default_factory=dict)      # live (framing, upscale...)


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
        self.overlays = OverlayRenderer()
        self.nametag = NameTagRenderer()
        self.framer = AutoFramer()
        self.lighting = LightingEnhancer()
        self.denoiser = Denoiser()
        self.pause = PauseController()
        self.splash = Splash()
        self._lut = None              # (path, Lut3D)
        self._screen = None           # presenter-mode screen grabber
        self._bg_path = None
        self._requests: set[str] = {"camera", "segmenter", "output"}
        self._req_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._latest_lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._latest_alpha: np.ndarray | None = None
        self._latest_fg: np.ndarray | None = None
        self._frame_count = 0
        self._alpha: np.ndarray | None = None
        self._timer = _Timer()
        self._warned: set[str] = set()
        self._calib = None            # (progress_cb, done_cb) when requested
        self._pause_req: str | None = None
        self._last_id = 0
        self.sleeping = False         # camera released, splash on the output
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

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self):
        """(composited BGR, alpha or None, foreground BGR or None, frame counter)."""
        with self._latest_lock:
            return self._latest_bgr, self._latest_alpha, self._latest_fg, self._frame_count

    def set_sleeping(self, sleeping: bool) -> None:
        """Power saving: release the physical camera and show the splash."""
        if sleeping == self.sleeping:
            return
        self.sleeping = sleeping
        log.info("Camera %s.", "sleeping (released)" if sleeping else "waking")
        if not sleeping:
            self.request("camera")

    def set_pause(self, mode: str) -> None:
        """'live', 'freeze' (Pause Camera) or 'loop' (Pause Loop)."""
        self._pause_req = mode

    def toggle_pause(self, mode: str = pause_mod.FREEZE) -> None:
        self.set_pause(pause_mod.LIVE if self.stats.pause != pause_mod.LIVE else mode)

    def calibrate(self, progress=None, done=None) -> None:
        """Benchmark models/providers on the engine thread, apply the winner.
        progress(i, n, label) and done(results, best) are called from that thread."""
        self._calib = (progress, done)
        self.request("calibrate")

    def camera_prop(self, name: str, value: float | None = None):
        """Get (value=None) or set a hardware UVC control on the open camera."""
        cap = self.cap
        if cap is None:
            return None
        if value is None:
            return cap.get_prop(name)
        return cap.set_prop(name, value)

    def snapshot(self, path: str | Path | None = None) -> Path | None:
        """Save the current composited frame as PNG (with alpha in Remove mode)."""
        bgr, alpha, fg, _ = self.latest()
        if bgr is None:
            return None
        if path is None:
            path = pictures_dir() / time.strftime("clearframe-%Y%m%d-%H%M%S.png")
        path = Path(path)
        img = bgr
        if alpha is not None and fg is not None and self.settings.background.mode == "remove" \
                and alpha.shape[:2] == fg.shape[:2]:
            img = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
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

    # ------------------------------------------------------------ internals
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)
            self.stats.warnings.append(msg)

    def _note(self, key: str, msg: str | None) -> None:
        """Live UI note; logged once each time it appears."""
        notes = self.stats.notes
        if msg is None:
            notes.pop(key, None)
        else:
            if key not in notes:
                log.warning(msg)
            notes[key] = msg

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
        self.stats.warnings = [w for w in self.stats.warnings if not w.startswith("Capture")] \
            + [f"Capture is not what was requested: {w}" for w in neg.warnings]
        for name, value in (cs.uvc or {}).items():
            self.cap.set_prop(name, value)
        if self.seg:
            self.seg.reset()
        self.framer.reset()
        self.lighting.reset()
        self.denoiser.reset()
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
        self._warned.discard("driver")
        self.stats.warnings = [w for w in self.stats.warnings if not w.startswith("No virtual camera")]
        if not o.enabled:
            return
        W, H = self.settings.output_size()
        try:
            self.out = VirtualCamOutput(W, H, o.fps, o.backend, o.rgba)
            self.stats.output = f"{self.out.device} {W}x{H}@{o.fps}" + (" RGBA" if self.out.rgba else "")
        except VirtualCamError as e:
            self._warn_once("driver", str(e))
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

    def _run_calibration(self) -> None:
        from .calibrate import apply_choice, run_calibration
        progress, done = self._calib or (None, None)
        self._calib = None
        frame = None
        if self.cap is not None:
            frame, _ = self.cap.read(0, timeout=0.2)
        if self.seg is not None:
            self.seg.close()
            self.seg = None
        self.stats.status = "calibrating"
        target = max(15, self.settings.output.fps)
        overhead = 6.0 + (6.0 if self.settings.camera.denoise > 0 else 0.0) \
            + (5.0 if self.settings.lighting.enabled else 0.0)
        results, best = run_calibration(target, frame, progress, cancel=lambda: not self._running,
                                        downsample_ratio=self.settings.seg.downsample_ratio,
                                        overhead_ms=overhead)
        if not self._running:
            return
        if best is not None:
            apply_choice(self.settings.seg, best, target)
            log.info("Calibration picked: %s (%.1f ms)", best.label, best.ms)
        self._open_segmenter()
        if done:
            try:
                done(results, best)
            except Exception:  # noqa: BLE001
                log.exception("calibration callback failed")

    def _publish(self, bgr, alpha=None, fg=None) -> None:
        with self._latest_lock:
            self._latest_bgr, self._latest_alpha, self._latest_fg = bgr, alpha, fg
            self._frame_count += 1

    def _idle_frame(self, text: str) -> None:
        """Camera is off: keep the virtual camera fed with the splash."""
        img = self.splash.get(self.settings.output_size(), self.settings.pause.splash_path, text)
        if self.out is not None:
            self.out.send(img)
        self._publish(img)
        time.sleep(0.2)

    # ------------------------------------------------------------ look
    def _look(self, frame: np.ndarray, new_frame: bool) -> np.ndarray:
        """Colour pipeline on the full-resolution frame (never resampled)."""
        s = self.settings
        cs = s.camera
        fg = self.denoiser.apply(frame, cs.denoise, new_frame)
        fg = sharpen(fg, cs.sharpen)
        fg = comp.adjust(fg, cs.brightness, cs.contrast, cs.saturation, cs.hue, cs.gamma)
        if s.lighting.enabled:
            fg = self.lighting.apply(fg, s.lighting.strength, update=new_frame)
        ex = s.extras
        if ex.lut_path:
            if self._lut is None or self._lut[0] != ex.lut_path:
                from .lut import Lut3D
                try:
                    self._lut = (ex.lut_path, Lut3D(ex.lut_path))
                except Exception as e:  # noqa: BLE001
                    log.warning("Could not load LUT %s: %s", ex.lut_path, e)
                    self._lut = (ex.lut_path, None)
            if self._lut[1] is not None:
                fg = self._lut[1].apply(fg, ex.lut_strength)
        return fg

    # ------------------------------------------------------------ framing
    def _fit_to_output(self, frame: np.ndarray, alpha: np.ndarray | None, size: tuple[int, int],
                       track: tuple[float, float, float] | None = None):
        """Crop to the output aspect (centred, or the auto-framing window) and
        scale to the output size. The source is only ever scaled *down*; if a
        crop is smaller than the output the user is warned (it must upscale)."""
        W, H = size
        h, w = frame.shape[:2]
        if w * H > h * W:
            bw, bh = h * W / H, float(h)
        else:
            bw, bh = float(w), w * H / W
        cx, cy, z = track if track is not None else (0.5, 0.5, 1.0)
        if track is not None and self.settings.autoframe.limit_native:
            z = min(z, max(1.0, bw / W))
        if z < 1.01:
            z = 1.0                       # settle exactly on the uncropped frame
        cw, ch = bw / z, bh / z
        x0 = float(np.clip(cx * w - cw / 2, 0, w - cw))
        y0 = float(np.clip(cy * h - ch / 2, 0, h - ch))
        scale = W / cw

        if track is not None and cw < W * 0.97:
            self._note("framing", f"Auto-framing zoom {z:.1f}x leaves {int(cw)}x{int(ch)} real pixels, "
                                  f"below the {W}x{H} output (upscaling). Lower the zoom, pick a "
                                  "smaller output resolution, or enable 'Never upscale'.")
        else:
            self._note("framing", None)
        if bw < W - 1:
            self._note("upscale", f"Camera gives {w}x{h}, below the {W}x{H} output (upscaling). "
                                  "Lower the output resolution for a sharper picture.")
        else:
            self._note("upscale", None)

        aligned = abs(x0 - round(x0)) < 0.02 and abs(y0 - round(y0)) < 0.02
        if abs(scale - 1.0) < 1e-4 and aligned:
            xi, yi = int(round(x0)), int(round(y0))
            if (xi, yi, W, H) == (0, 0, w, h):
                return frame, alpha
            frame = frame[yi:yi + H, xi:xi + W]
            alpha = alpha[yi:yi + H, xi:xi + W] if alpha is not None else None
            return np.ascontiguousarray(frame), (np.ascontiguousarray(alpha) if alpha is not None else None)
        if track is not None and scale >= 0.75:
            # sub-pixel window so slow pans are smooth
            M = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
            frame = cv2.warpAffine(frame, M, (W, H), flags=interp, borderMode=cv2.BORDER_REPLICATE)
            if alpha is not None:
                alpha = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return frame, alpha
        xi, yi = int(round(x0)), int(round(y0))
        cwi, chi = min(int(round(cw)), w - xi), min(int(round(ch)), h - yi)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        frame = cv2.resize(frame[yi:yi + chi, xi:xi + cwi], (W, H), interpolation=interp)
        if alpha is not None:
            alpha = cv2.resize(alpha[yi:yi + chi, xi:xi + cwi], (W, H),
                               interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        return frame, alpha

    # ------------------------------------------------------------ presenter
    def _presenter(self, fg: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Screen capture behind a scaled-down matted person."""
        ex = self.settings.extras
        if self._screen is None:
            from .presenter import ScreenGrabber
            self._screen = ScreenGrabber()
        self._screen.start(ex.presenter_monitor)
        W, H = size
        shot = self._screen.frame()
        if shot is None:
            out = np.full((H, W, 3), 16, np.uint8)
        else:
            out = comp.place_media(shot, size, "fit")
            out = out.copy() if out is shot else np.ascontiguousarray(out)
        k = float(np.clip(ex.presenter_scale, 0.1, 1.0))
        pw, ph = max(2, int(W * k)), max(2, int(H * k))
        person = cv2.resize(fg, (pw, ph), interpolation=cv2.INTER_AREA)
        pa = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_AREA)
        x, y = anchor_pos(ex.presenter_anchor, W, H, pw, ph, 0.0, 0.0)
        comp.overlay_bgra(out, np.dstack((person, pa)), x, y)
        return out

    # -------------------------------------------------------------- process
    def process(self, frame: np.ndarray, new_frame: bool = True):
        """One frame through the pipeline. Returns (bgr, rgba_or_None, alpha_or_None, fg)."""
        s = self.settings
        bs = s.background
        t = self._timer
        t.start()
        mode = bs.mode
        size = s.output_size()
        presenter = s.extras.presenter

        alpha = None
        if (mode != "original" or presenter) and self.seg is not None:
            if new_frame or self._alpha is None or self._alpha.shape[:2] != frame.shape[:2]:
                self.seg.ema_strength, self.seg.edge_tighten = s.seg.ema, s.seg.edge_tighten
                self.seg.solidity = s.seg.solidity
                self._alpha = self.seg.alpha(frame)
            alpha = self._alpha
        t.lap("segment")

        track = None
        if s.autoframe.enabled and not presenter:
            track = self.framer.update(frame, s.autoframe, new_frame)
            t.lap("autoframe")

        fg = self._look(frame, new_frame)
        t.lap("look")
        fg, alpha = self._fit_to_output(fg, alpha, size, track)
        t.lap("fit")

        rgba = None
        want_rgba = self.out is not None and self.out.rgba and mode == "remove" and not presenter
        if alpha is None:
            bgr = fg
        elif presenter:
            bgr = self._presenter(fg, alpha, size)
        elif mode == "blur":
            bgr = comp.blend(fg, comp.blur_background(fg, alpha, bs.blur_strength, bs.bokeh), alpha)
        elif mode == "replace":
            bgr = comp.blend(fg, self.bg_layer.render(self.bg_source, bs, size), alpha)
        elif mode == "color":
            bgr = comp.blend_color(fg, parse_color(bs.color), alpha)
        else:  # remove
            bgr = comp.blend_color(fg, parse_color(bs.remove_fallback_color), alpha)
        t.lap("composite")

        layers = [l for l in s.overlays if l.enabled and l.path]
        if layers or s.nametag.enabled:
            if bgr is frame or bgr is fg or bgr.base is not None:
                bgr = bgr.copy()              # never draw on the camera's buffer
            self.overlays.draw(bgr, layers)
            self.nametag.draw(bgr, s.nametag)
            if want_rgba:
                src = comp.fallback_fill(fg, alpha, parse_color(bs.remove_fallback_color))
                a2 = alpha.copy()
                self.overlays.draw(src, layers, a2)
                self.nametag.draw(src, s.nametag, a2)
                rgba = comp.to_rgba(src, a2)
            t.lap("overlays")
        if want_rgba and rgba is None:
            rgba = comp.to_rgba(comp.fallback_fill(fg, alpha, parse_color(bs.remove_fallback_color)), alpha)
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
                if "output" in req:
                    self._open_output()
                if "calibrate" in req:
                    self._run_calibration()
                if self._pause_req is not None:
                    self.stats.pause = self.pause.set_mode(self._pause_req)
                    self._pause_req = None
                    log.info("Pause mode: %s", self.stats.pause)
                if self.sleeping:
                    if self.cap is not None:
                        self._close_camera()
                    self.stats.status = "sleeping"
                    self._idle_frame("Camera sleeping")
                    continue
                if "camera" in req:
                    if not self._open_camera():
                        self._idle_frame("Camera unavailable")
                        time.sleep(0.8)
                        self.request("camera")
                        continue
                self._sync_background_source()
                if self._screen is not None and not self.settings.extras.presenter:
                    self._screen.stop()
                if self.cap is None:
                    self._idle_frame("Camera off")
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
                ps = self.settings.pause
                frame, new_frame = self.pause.feed(frame, new_frame, ps.loop_seconds, ps.loop_buffer)
                if self.pause.mode == pause_mod.FREEZE and ps.splash_on_pause:
                    bgr = self.splash.get(self.settings.output_size(), ps.splash_path, "Camera paused")
                    rgba = alpha = fg = None
                else:
                    try:
                        bgr, rgba, alpha, fg = self.process(frame, new_frame)
                    except Exception as e:  # noqa: BLE001 - one bad frame must not kill the camera
                        if self._warned.isdisjoint({"process"}):
                            self._warned.add("process")
                            log.exception("Frame processing failed; passing the camera through")
                        self._note("process", f"Processing error: {e}")
                        size = self.settings.output_size()
                        bgr, rgba, alpha, fg = cv2.resize(frame, size), None, None, None
                    else:
                        if "process" in self.stats.notes:
                            self._note("process", None)
                            self._warned.discard("process")
                if self.out is not None:
                    self.out.send(bgr, rgba)
                    self._timer.lap("output send")
                self._publish(bgr, alpha, fg)

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
            self.overlays.close()
            self.framer.close()
            self.pause.close()
            if self._screen is not None:
                self._screen.stop()
            log.info("Engine stopped.")
