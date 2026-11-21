"""Download the segmentation / face-detection models into models/ (gitignored).

Usage:  python scripts/download_models.py [--force]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# name -> (url, minimum plausible size in bytes)
MODELS = {
    "rvm_mobilenetv3_fp32.onnx": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        10_000_000,
    ),
    "rvm_mobilenetv3_fp16.onnx": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx",
        5_000_000,
    ),
    "selfie_segmenter_landscape.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
        100_000,
    ),
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        100_000,
    ),
}


def download(name: str, url: str, min_size: int, force: bool) -> bool:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size >= min_size and not force:
        print(f"  ok       {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {name} ...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vcam-clone-model-downloader"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size < min_size:
            raise RuntimeError(f"download too small ({tmp.stat().st_size} bytes)")
        tmp.replace(dest)
        print(f" done ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f" FAILED: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")
    ok = all([download(n, u, s, args.force) for n, (u, s) in MODELS.items()])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
