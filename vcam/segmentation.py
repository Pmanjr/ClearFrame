"""Person segmentation / matting.

Segmentation always runs on a *downscaled copy* of the frame. Only the alpha
mask is brought back to full resolution (fast guided filter, or RVM's own
refiner) - the camera pixels themselves are never resampled here.
"""
from __future__ import annotations

import functools
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .settings import SegmentationSettings, models_dir

log = logging.getLogger("vcam.seg")

RVM_MODELS = {"fp32": "rvm_mobilenetv3_fp32.onnx", "fp16": "rvm_mobilenetv3_fp16.onnx"}
MP_SELFIE = "selfie_segmenter_landscape.tflite"


class ModelMissing(RuntimeError):
    pass


def _model(name: str) -> Path:
    p = models_dir() / name
    if not p.exists():
        raise ModelMissing(f"Model {name} not found in {models_dir()}. "
                           "Run: python scripts/download_models.py")
    return p


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------
FIT_W = 360


def downscale(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Anti-aliased downscale for the model input. Exact 2x area steps are
    SIMD fast paths; the remainder (< 2x) is bilinear. Several times faster
    than one generic INTER_AREA pass, with no visible aliasing."""
    w, h = size
    H, W = frame.shape[:2]
    while W >= w * 2 and H >= h * 2 and W % 2 == 0 and H % 2 == 0:
        W, H = W // 2, H // 2
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    if (W, H) != (w, h):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


def guided_upsample(alpha_small: np.ndarray, guide_small_bgr: np.ndarray,
                    guide_full_bgr: np.ndarray, radius: int = 3, eps: float = 2e-4,
                    ws: dict | None = None) -> np.ndarray:
    """Fast guided filter (He & Sun 2015): fit alpha ~= a*I + b on the small
    image, upsample the coefficients, apply them to the full-res luma. Edges
    of the result follow the full-resolution image. Returns uint8 HxW.

    `ws` is a per-caller workspace dict: the full-resolution float scratch
    buffers live there so they are not reallocated (and page-faulted) on
    every frame, which is where most of the time used to go."""
    # The linear fit is smooth, so it never needs more than FIT_W pixels of
    # width: fitting a larger mask on a reduced copy gives the same result
    # (mean alpha difference ~0.1/255) for a third of the cost.
    sw = guide_small_bgr.shape[1]
    if sw > FIT_W:
        fw, fh = FIT_W, max(2, round(guide_small_bgr.shape[0] * FIT_W / sw))
        guide_small_bgr = cv2.resize(guide_small_bgr, (fw, fh), interpolation=cv2.INTER_AREA)
        alpha_small = cv2.resize(alpha_small, (fw, fh), interpolation=cv2.INTER_AREA)
        radius = max(2, round(radius * FIT_W / sw))
    I = cv2.cvtColor(guide_small_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
    p = alpha_small
    k = (2 * radius + 1, 2 * radius + 1)
    # one 4-channel box filter instead of four separate ones
    m = cv2.blur(cv2.merge((I, p, I * p, I * I)), k)
    mean_I, mean_p = m[..., 0], m[..., 1]
    a = (m[..., 2] - mean_I * mean_p) / (m[..., 3] - mean_I * mean_I + eps)
    b = mean_p - a * mean_I
    ab = cv2.blur(cv2.merge((a, b)), k)
    a, b = np.ascontiguousarray(ab[..., 0]), np.ascontiguousarray(ab[..., 1])
    H, W = guide_full_bgr.shape[:2]
    if ws is None:
        ws = {}
    if ws.get("shape") != (H, W):
        ws["shape"] = (H, W)
        ws["a"] = np.empty((H, W), np.float32)
        ws["b"] = np.empty((H, W), np.float32)
        ws["q"] = np.empty((H, W), np.float32)
        ws["gray"] = np.empty((H, W), np.uint8)
    a_full = cv2.resize(a, (W, H), dst=ws["a"], interpolation=cv2.INTER_LINEAR)
    b_full = cv2.resize(b, (W, H), dst=ws["b"], interpolation=cv2.INTER_LINEAR)
    gray_full = cv2.cvtColor(guide_full_bgr, cv2.COLOR_BGR2GRAY, dst=ws["gray"])
    # q = a * (gray/255) + b, scaled to 0..255 and saturated to uint8 in one pass
    q = cv2.multiply(a_full, gray_full, dst=ws["q"], scale=1.0, dtype=cv2.CV_32F)
    # the result is a fresh array: it is published to the UI / kept across frames
    return cv2.addWeighted(q, 1.0, b_full, 255.0, 0.0, dtype=cv2.CV_8U)


@functools.lru_cache(maxsize=64)
def _edge_lut(tighten: float, solidity: float = 0.0) -> np.ndarray:
    """Alpha curve applied to the finished mask.

    tighten:  cuts the faintest alpha, removing the thin halo of old
              background around hair and shoulders.
    solidity: lifts mid alpha towards opaque. The matting model is unsure
              about things attached to a person (headphones, a chair back,
              a microphone) and returns them half-transparent; this makes
              them solid without hardening the outline much.
    """
    lo = 0.45 * float(np.clip(tighten, 0.0, 1.0))
    x = np.arange(256, dtype=np.float32) / 255.0
    x = np.clip((x - lo) / max(1.0 - lo - 0.02, 1e-3), 0.0, 1.0)
    power = 1.0 + 3.0 * float(np.clip(solidity, 0.0, 1.0))
    x = 1.0 - np.power(1.0 - x, power)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


class TemporalEMA:
    """Motion-adaptive exponential smoothing of the (small) mask: static areas
    are smoothed hard to kill flicker, moving edges track the new mask so
    there is no ghost trail."""

    def __init__(self):
        self.prev: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, mask: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0.0 or self.prev is None or self.prev.shape != mask.shape:
            self.prev = mask.copy()
            return mask
        keep = float(np.clip(strength, 0.0, 0.95))
        # weight of the new sample: (1-keep) when static, -> 1 when it moves
        w = cv2.absdiff(mask, self.prev)
        cv2.addWeighted(w, 3.0, w, 0.0, 1.0 - keep, dst=w)
        cv2.min(w, 1.0, dst=w)
        delta = cv2.subtract(mask, self.prev)
        cv2.multiply(delta, w, dst=delta)
        out = cv2.add(self.prev, delta)
        self.prev = out
        return out


# ---------------------------------------------------------------------------
# ONNX Runtime session helper
# ---------------------------------------------------------------------------
def available_providers() -> list[str]:
    import onnxruntime as ort
    return ort.get_available_providers()


def make_session(path: Path, provider: str = "dml", dml_device: int = 0):
    """provider: 'dml' or 'cpu'. Returns (session, provider_actually_used)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if provider == "dml" and "DmlExecutionProvider" in ort.get_available_providers():
        so.enable_mem_pattern = False            # required by DirectML
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            sess = ort.InferenceSession(
                str(path), so,
                providers=[("DmlExecutionProvider", {"device_id": int(dml_device)}),
                           "CPUExecutionProvider"])
            if "DmlExecutionProvider" in sess.get_providers():
                return sess, "dml"
            log.warning("DirectML session fell back to CPU.")
            return sess, "cpu"
        except Exception as e:  # noqa: BLE001
            log.warning("DirectML unavailable (%s); falling back to CPU.", e)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]), "cpu"


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------
class Segmenter:
    name = "none"
    provider = "cpu"

    def __init__(self):
        self.ema = TemporalEMA()
        self.ema_strength = 0.35
        self.edge_tighten = 0.3
        self.solidity = 0.4
        self._ws: dict = {}      # scratch buffers for guided_upsample
        self.infer_ms = 0.0      # model only
        self.total_ms = 0.0      # model + resize + EMA + upsample

    def reset(self) -> None:
        self.ema.reset()

    def _finish(self, a8: np.ndarray) -> np.ndarray:
        if self.edge_tighten <= 0 and self.solidity <= 0:
            return a8
        return cv2.LUT(a8, _edge_lut(round(self.edge_tighten, 3), round(self.solidity, 3)))

    def alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Full-resolution uint8 alpha (255 = person)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class RVMSegmenter(Segmenter):
    """Robust Video Matting (MobileNetV3), recurrent state kept across frames.

    refiner='guided': the network sees a copy downscaled by `downsample_ratio`
        (480x270 for 1080p at 0.25) and the alpha is upsampled with a fast
        guided filter against the full-res frame. Cheapest; default.
    refiner='rvm': the network gets the full frame with downsample_ratio set,
        so RVM's built-in deep guided filter produces the full-res alpha.
    """

    def __init__(self, provider="dml", dml_device=0, precision="fp32",
                 refiner="guided", downsample_ratio=0.25):
        super().__init__()
        self.sess, self.provider = make_session(_model(RVM_MODELS[precision]), provider, dml_device)
        self.dtype = np.float16 if precision == "fp16" else np.float32
        self.refiner = refiner
        self.ratio = float(downsample_ratio)
        self.name = f"rvm-{precision}-{refiner}"
        self._rec = None
        self._shape = None
        self._out_names = ["pha", "r1o", "r2o", "r3o", "r4o"]

    def reset(self) -> None:
        super().reset()
        self._rec = None

    def _run(self, src_bgr: np.ndarray, ratio: float) -> np.ndarray:
        if self._rec is None or self._shape != src_bgr.shape:
            self._rec = [np.zeros((1, 1, 1, 1), self.dtype) for _ in range(4)]
            self._shape = src_bgr.shape
        blob = cv2.dnn.blobFromImage(src_bgr, 1.0 / 255.0, swapRB=True)
        if self.dtype is not np.float32:
            blob = blob.astype(self.dtype)
        t = time.perf_counter()
        pha, *rec = self.sess.run(self._out_names, {
            "src": blob,
            "r1i": self._rec[0], "r2i": self._rec[1], "r3i": self._rec[2], "r4i": self._rec[3],
            "downsample_ratio": np.array([ratio], np.float32),
        })
        self.infer_ms = (time.perf_counter() - t) * 1000.0
        self._rec = rec
        pha = pha[0, 0]
        return pha if pha.dtype == np.float32 else pha.astype(np.float32)

    def alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        if self.refiner == "rvm":
            # RVM is temporally stable by itself (recurrent state); an extra
            # full-resolution EMA would cost ~6 ms for no visible gain.
            out = cv2.convertScaleAbs(self._run(frame_bgr, self.ratio), alpha=255.0)
        else:
            # keep dimensions even so the recurrent pyramid is stable
            sw = max(64, int(round(W * self.ratio / 2)) * 2)
            sh = max(64, int(round(H * self.ratio / 2)) * 2)
            small = downscale(frame_bgr, (sw, sh))
            pha = self.ema(self._run(small, 1.0), self.ema_strength)
            out = guided_upsample(pha, small, frame_bgr, radius=max(2, round(sw / 160)), ws=self._ws)
        out = self._finish(out)
        self.total_ms = (time.perf_counter() - t0) * 1000.0
        return out


class MediaPipeSegmenter(Segmenter):
    """MediaPipe selfie segmenter (landscape, 256x144) on CPU/XNNPACK."""

    name = "mediapipe-selfie"
    SIZE = (256, 144)

    def __init__(self):
        super().__init__()
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_buffer=_model(MP_SELFIE).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False)
        self._seg = vision.ImageSegmenter.create_from_options(opts)
        self._ts = 0
        self.provider = "cpu"

    def alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        small = cv2.resize(frame_bgr, self.SIZE, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33
        t = time.perf_counter()
        res = self._seg.segment_for_video(img, self._ts)
        self.infer_ms = (time.perf_counter() - t) * 1000.0
        masks = res.confidence_masks
        m = np.array(masks[-1].numpy_view(), dtype=np.float32, copy=True)
        if m.ndim == 3:
            m = m[..., 0]
        # sharpen the soft sigmoid a little so edges are decisive before refinement
        m = np.clip((m - 0.5) * 2.2 + 0.5, 0.0, 1.0)
        m = self.ema(m, self.ema_strength)
        out = self._finish(guided_upsample(m, small, frame_bgr, radius=2, eps=1e-3, ws=self._ws))
        self.total_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def close(self) -> None:
        try:
            self._seg.close()
        except Exception:  # noqa: BLE001
            pass


def create_segmenter(s: SegmentationSettings) -> Segmenter:
    """Build the segmenter described by settings, falling back gracefully."""
    provider = "dml" if s.provider in ("auto", "dml") else "cpu"
    seg: Segmenter
    if s.mode == "performance":
        try:
            seg = MediaPipeSegmenter()
        except ModelMissing:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("MediaPipe segmenter failed (%s); using RVM instead.", e)
            seg = RVMSegmenter(provider, s.dml_device, s.rvm_precision, s.rvm_refiner, s.downsample_ratio)
    else:
        seg = RVMSegmenter(provider, s.dml_device, s.rvm_precision, s.rvm_refiner, s.downsample_ratio)
    seg.ema_strength, seg.edge_tighten, seg.solidity = s.ema, s.edge_tighten, s.solidity
    log.info("Segmenter: %s on %s", seg.name, seg.provider.upper())
    return seg
