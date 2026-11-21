"""Command-line front end (Phase 1): run the pipeline headless with flags.

Examples:
  python -m vcam --list-cameras
  python -m vcam cli --mode blur --blur 60
  python -m vcam cli --mode replace --bg C:\\pics\\office.jpg --preview
  python -m vcam cli --mode remove --rgba            (Unity Capture, true alpha)
  python -m vcam cli --benchmark
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .settings import BG_MODES, FPS_CHOICES, RESOLUTIONS, Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vcam cli", description="vcam-clone command-line pipeline")
    g = p.add_argument_group("camera")
    g.add_argument("--list-cameras", action="store_true", help="list physical cameras and exit")
    g.add_argument("--camera", type=int, default=None, help="DirectShow camera index (default: first physical)")
    g.add_argument("--cam-res", default="1920x1080", help="capture resolution WxH (default 1920x1080)")
    g.add_argument("--cam-fps", type=int, default=30, choices=[15, 24, 25, 30, 50, 60])
    g.add_argument("--fourcc", default="MJPG", help="capture pixel format to request (default MJPG)")
    g.add_argument("--cam-backend", default="dshow", choices=["dshow", "msmf"])
    g.add_argument("--denoise", type=float, default=0.5, help="noise reduction 0-1 (0 = off)")
    g.add_argument("--sharpen", type=float, default=0.15, help="sharpen 0-1 (0 = off)")
    g.add_argument("--input", default=None, help="image/video file to use instead of a camera (testing)")

    g = p.add_argument_group("background")
    g.add_argument("--mode", default="blur", choices=BG_MODES)
    g.add_argument("--blur", type=int, default=55, help="blur strength 0-100")
    g.add_argument("--bokeh", action="store_true", help="disc-kernel bokeh instead of Gaussian")
    g.add_argument("--bg", default="", help="replacement background (jpg/png/webp/gif/mp4/webm)")
    g.add_argument("--bg-fit", default="fill", choices=["fit", "fill"])
    g.add_argument("--bg-blur", type=int, default=0, help="blur the replacement background 0-100")
    g.add_argument("--color", default="#00b140", help="colour for --mode color / Remove fallback")

    g = p.add_argument_group("segmentation")
    g.add_argument("--seg", default="quality", choices=["quality", "performance"],
                   help="quality = RVM MobileNetV3, performance = MediaPipe selfie")
    g.add_argument("--provider", default="auto", choices=["auto", "dml", "cpu"])
    g.add_argument("--dml-device", type=int, default=0)
    g.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    g.add_argument("--refiner", default="guided", choices=["guided", "rvm"],
                   help="alpha upsampling: fast guided filter, or RVM's built-in refiner")
    g.add_argument("--downsample", type=float, default=0.375,
                   help="model input scale (0.25 = fastest, 0.375 keeps headphones solid, 0.5 = finest)")
    g.add_argument("--solidity", type=float, default=0.4, help="make half-transparent parts opaque 0-1")
    g.add_argument("--ema", type=float, default=0.35, help="temporal mask smoothing 0-0.95")

    g = p.add_argument_group("output")
    g.add_argument("--out-res", default="1080p", choices=list(RESOLUTIONS))
    g.add_argument("--out-fps", type=int, default=30, choices=list(FPS_CHOICES))
    g.add_argument("--backend", default="auto", choices=["auto", "obs", "unitycapture"])
    g.add_argument("--rgba", action="store_true", help="send RGBA through Unity Capture (true transparency)")
    g.add_argument("--no-vcam", action="store_true", help="do not open a virtual camera")
    g.add_argument("--preview", action="store_true", help="show a preview window (q / Esc quits)")
    g.add_argument("--snapshot", default=None, help="save a PNG of the composited frame on exit")
    g.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")

    g = p.add_argument_group("extras")
    g.add_argument("--overlay", action="append", default=[], help="logo/watermark file (repeatable)")
    g.add_argument("--overlay-anchor", default="top-right")
    g.add_argument("--name", default="", help="name tag: name")
    g.add_argument("--title", default="", help="name tag: title")
    g.add_argument("--autoframe", type=float, default=0, metavar="ZOOM", help="auto-framing zoom (1-3)")
    g.add_argument("--enhance", type=float, default=0, metavar="STRENGTH", help="enhance lighting 0-1")
    g.add_argument("--lut", default="", help=".cube 3D LUT")
    g.add_argument("--presenter", action="store_true", help="screen capture behind the matted person")
    g.add_argument("--pause-loop-after", type=float, default=0, metavar="SEC",
                   help="switch to Pause Loop after SEC seconds (demo/testing)")

    p.add_argument("--benchmark", action="store_true", help="benchmark every model/provider and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def settings_from_args(a: argparse.Namespace) -> Settings:
    s = Settings()
    try:
        w, h = (int(x) for x in a.cam_res.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--cam-res must look like 1920x1080, got {a.cam_res!r}")
    s.camera.width, s.camera.height, s.camera.fps = w, h, a.cam_fps
    s.camera.fourcc, s.camera.backend = a.fourcc, a.cam_backend
    s.camera.denoise, s.camera.sharpen, s.seg.solidity = a.denoise, a.sharpen, a.solidity
    b = s.background
    b.mode, b.blur_strength, b.bokeh = a.mode, a.blur, a.bokeh
    b.replace_path, b.fit, b.replace_blur = a.bg, a.bg_fit, a.bg_blur
    b.color = b.remove_fallback_color = a.color
    g = s.seg
    g.mode, g.provider, g.dml_device = a.seg, a.provider, a.dml_device
    g.rvm_precision, g.rvm_refiner, g.downsample_ratio, g.ema = a.precision, a.refiner, a.downsample, a.ema
    o = s.output
    o.resolution, o.fps, o.backend, o.rgba, o.enabled = a.out_res, a.out_fps, a.backend, a.rgba, not a.no_vcam
    from .settings import OverlaySettings
    s.overlays = [OverlaySettings(path=p, anchor=a.overlay_anchor) for p in a.overlay]
    if a.name or a.title:
        s.nametag.enabled, s.nametag.name, s.nametag.title = True, a.name, a.title
    if a.autoframe:
        s.autoframe.enabled, s.autoframe.zoom = True, a.autoframe
    if a.enhance:
        s.lighting.enabled, s.lighting.strength = True, a.enhance
    s.extras.lut_path, s.extras.presenter = a.lut, a.presenter
    return s


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("vcam.cli")

    if a.list_cameras:
        from .devices import list_cameras
        for d in list_cameras(include_virtual=True):
            print(f"  [{d.index}] {d.name}" + ("   (virtual - excluded)" if d.virtual else ""))
        return 0

    if a.benchmark:
        from .calibrate import print_report, run_calibration
        print_report(run_calibration(target_fps=a.out_fps))
        return 0

    if a.mode == "replace" and not a.bg:
        log.warning("--mode replace without --bg: using a plain dark background.")

    from .engine import Engine
    s = settings_from_args(a)
    eng = Engine(s, input_file=a.input, camera_index=a.camera)
    eng.start()
    t0 = last_log = time.time()
    shown = -1
    try:
        while True:
            if a.preview:
                import cv2
                bgr, _, _, n = eng.latest()
                if bgr is not None and n != shown:
                    shown = n
                    cv2.imshow("vcam-clone preview (q to quit)", bgr)
                if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.05)
            now = time.time()
            if now - last_log >= 2.0:
                last_log = now
                st = eng.stats
                stages = "  ".join(f"{k} {v:.1f}" for k, v in st.stages.items())
                log.info("%.1f FPS | frame %.1f ms | inference %.1f ms (%s/%s) | %s | out: %s",
                         st.fps, st.frame_ms, st.infer_ms, st.segmenter, st.provider.upper(),
                         stages, st.output or "none")
            if a.pause_loop_after and now - t0 >= a.pause_loop_after:
                eng.set_pause("loop")
                a.pause_loop_after = 0
            if a.seconds and now - t0 >= a.seconds:
                break
            if not eng.alive:
                return 1
    except KeyboardInterrupt:
        pass
    finally:
        if a.snapshot:
            eng.snapshot(a.snapshot)
            time.sleep(0.8)
        eng.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
