"""Startup calibration: benchmark each model / execution provider on this
machine and pick the best-quality one that holds the target frame rate."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .segmentation import MediaPipeSegmenter, ModelMissing, RVMSegmenter, available_providers
from .settings import SegmentationSettings

log = logging.getLogger("vcam.calibrate")


@dataclass
class Candidate:
    label: str
    mode: str                  # quality | performance
    provider: str = "cpu"      # dml | cpu
    dml_device: int = 0
    precision: str = "fp32"
    refiner: str = "guided"
    quality: int = 0           # higher = better looking
    ms: float = float("inf")   # median per-frame segmentation cost (model + refine)
    infer_ms: float = float("inf")
    error: str = ""

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms if self.ms > 0 and self.ms != float("inf") else 0.0


def candidates() -> list[Candidate]:
    c: list[Candidate] = []
    if "DmlExecutionProvider" in available_providers():
        for dev in (0, 1):
            tag = f"DML gpu{dev}"
            c += [
                Candidate(f"RVM fp32 + built-in refiner ({tag})", "quality", "dml", dev, "fp32", "rvm", 100),
                Candidate(f"RVM fp32 + guided filter ({tag})", "quality", "dml", dev, "fp32", "guided", 90),
                Candidate(f"RVM fp16 + guided filter ({tag})", "quality", "dml", dev, "fp16", "guided", 85),
            ]
    c += [
        Candidate("RVM fp32 + guided filter (CPU)", "quality", "cpu", 0, "fp32", "guided", 80),
        Candidate("MediaPipe selfie (CPU)", "performance", "cpu", 0, "fp32", "guided", 50),
    ]
    return c


def _test_frame(w=1920, h=1080) -> np.ndarray:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w * 255, y / h * 255, (x + y) / (w + h) * 255], axis=2)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def bench(c: Candidate, frame: np.ndarray, frames: int = 30, warmup: int = 8,
          downsample_ratio: float = 0.375, cancel=None) -> Candidate:
    seg = None
    try:
        if c.mode == "performance":
            seg = MediaPipeSegmenter()
        else:
            seg = RVMSegmenter(c.provider, c.dml_device, c.precision, c.refiner, downsample_ratio)
            if seg.provider != c.provider:
                raise RuntimeError(f"provider fell back to {seg.provider}")
        for _ in range(warmup):
            seg.alpha(frame)
        total, infer = [], []
        t_end = time.perf_counter() + 4.0
        for _ in range(frames):
            seg.alpha(frame)
            total.append(seg.total_ms)
            infer.append(seg.infer_ms)
            if time.perf_counter() > t_end or (cancel and cancel()):
                break
        c.ms, c.infer_ms = float(np.median(total)), float(np.median(infer))
    except ModelMissing as e:
        c.error = str(e)
    except Exception as e:  # noqa: BLE001
        c.error = str(e).splitlines()[0][:160]
    finally:
        if seg is not None:
            seg.close()
    return c


def choose(results: list[Candidate], target_fps: int, overhead_ms: float = 6.0) -> Candidate | None:
    """Best quality whose segmentation cost leaves room for compositing
    (`overhead_ms`) inside the frame budget; else the fastest that works."""
    ok = [r for r in results if not r.error]
    if not ok:
        return None
    budget = 1000.0 / target_fps - overhead_ms
    fits = [r for r in ok if r.ms <= budget]
    if fits:
        return max(fits, key=lambda r: (r.quality, -r.ms))
    return min(ok, key=lambda r: r.ms)


def run_calibration(target_fps: int = 30, frame: np.ndarray | None = None,
                    progress=None, cancel=None,
                    downsample_ratio: float = 0.375) -> tuple[list[Candidate], Candidate | None]:
    """progress(i, n, label) is called before each candidate. cancel() -> True
    aborts (app is quitting); nothing is selected in that case."""
    frame = _test_frame() if frame is None else frame
    cands = candidates()
    results = []
    dml_dead: set[int] = set()
    for i, c in enumerate(cands):
        if cancel and cancel():
            return results, None
        if progress:
            progress(i, len(cands), c.label)
        if c.provider == "dml" and c.dml_device in dml_dead:
            c.error = "adapter not available"
        else:
            bench(c, frame, downsample_ratio=downsample_ratio, cancel=cancel)
            if c.error and c.provider == "dml" and c.refiner == "rvm":
                # first candidate of this adapter failed -> adapter likely absent
                if "device" in c.error.lower() or "adapter" in c.error.lower() or c.dml_device > 0:
                    dml_dead.add(c.dml_device)
        log.info("  %-46s %s", c.label, c.error or f"{c.ms:6.1f} ms  ({c.fps:5.1f} FPS max)")
        results.append(c)
    if progress:
        progress(len(cands), len(cands), "done")
    return results, choose(results, target_fps)


def apply_choice(s: SegmentationSettings, c: Candidate, target_fps: int) -> None:
    s.mode, s.provider, s.dml_device = c.mode, c.provider, c.dml_device
    s.rvm_precision, s.rvm_refiner = c.precision, c.refiner
    s.calibrated, s.target_fps = True, target_fps


def print_report(result: tuple[list[Candidate], Candidate | None]) -> None:
    results, best = result
    print("\nCalibration (1920x1080 input, segmentation + mask refinement per frame):")
    for r in results:
        mark = "->" if r is best else "  "
        if r.error:
            print(f" {mark} {r.label:<46} unavailable: {r.error}")
        else:
            print(f" {mark} {r.label:<46} {r.ms:6.1f} ms  (model {r.infer_ms:5.1f} ms)  {r.fps:6.1f} FPS max")
    if best:
        print(f"\nSelected: {best.label}")
