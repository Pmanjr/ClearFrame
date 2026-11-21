"""Profile every pipeline stage on a fixed 1080p frame (no camera needed).

Usage:  python scripts/profile_pipeline.py [image] [--frames 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcam import compositor as comp  # noqa: E402
from vcam.calibrate import _test_frame  # noqa: E402
from vcam.engine import Engine  # noqa: E402
from vcam.lighting import LightingEnhancer  # noqa: E402
from vcam.segmentation import MediaPipeSegmenter, RVMSegmenter  # noqa: E402
from vcam.settings import Settings  # noqa: E402


def bench(name: str, fn, n: int) -> float:
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ms = float(np.median(ts))
    print(f"  {name:<44} {ms:7.2f} ms   p95 {np.percentile(ts, 95):6.2f}")
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()
    frame = cv2.imread(a.image) if a.image else None
    if frame is None:
        frame = _test_frame()
    frame = cv2.resize(frame, (1920, 1080))
    n = a.frames
    print(f"OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads, frame {frame.shape[1]}x{frame.shape[0]}\n")

    print("Segmentation (model + mask refinement):")
    alpha = None
    for label, make in [
        ("RVM fp32 guided (DML)", lambda: RVMSegmenter("dml", 0, "fp32", "guided")),
        ("RVM fp32 built-in refiner (DML)", lambda: RVMSegmenter("dml", 0, "fp32", "rvm")),
        ("MediaPipe selfie (CPU)", MediaPipeSegmenter),
    ]:
        try:
            seg = make()
            bench(label, lambda: seg.alpha(frame), n)
            print(f"      of which model: {seg.infer_ms:.2f} ms")
            if alpha is None:
                alpha = seg.alpha(frame)
            seg.close()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<44} unavailable: {e}")
    if alpha is None:
        alpha = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(alpha, (960, 600), 400, 255, -1)

    print("\nCompositing (1920x1080):")
    bg = np.ascontiguousarray(frame[::-1])
    bench("blend (replace)", lambda: comp.blend(frame, bg, alpha), n)
    bench("blend_color (colour / remove)", lambda: comp.blend_color(frame, (64, 177, 0), alpha), n)
    bench("blur background 55 + blend", lambda: comp.blend(frame, comp.blur_background(frame, alpha, 55), alpha), n)
    bench("bokeh background 55 + blend", lambda: comp.blend(frame, comp.blur_background(frame, alpha, 55, True), alpha), n)
    bench("to_rgba (Unity Capture)", lambda: comp.to_rgba(frame, alpha), n)

    print("\nLook:")
    bench("camera adjust (brightness/contrast/gamma)", lambda: comp.adjust(frame, 10, 10, 0, 0, 1.1), n)
    bench("camera adjust (+ saturation/hue)", lambda: comp.adjust(frame, 10, 10, 20, 5, 1.1), n)
    le = LightingEnhancer()
    bench("enhance lighting", lambda: le.apply(frame, 0.6), n)

    print("\nWhole pipeline (Engine.process, no camera / virtual cam):")
    for mode in ("original", "blur", "replace", "color", "remove"):
        s = Settings()
        s.background.mode = mode
        eng = Engine(s)
        eng._open_segmenter()
        ms = bench(f"mode={mode}", lambda: eng.process(frame, True), n)
        print(f"      -> {1000 / ms:5.0f} FPS max   stages: " +
              "  ".join(f"{k} {v:.1f}" for k, v in eng._timer.avg.items()))
        eng.seg.close()


if __name__ == "__main__":
    main()
