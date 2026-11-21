"""3D colour LUTs (.cube) applied at full resolution.

The cube is resampled once to a 64^3 table of *deltas* (LUT output minus
input). Per frame the pixel is quantised to 6 bits per channel only to pick
the delta, which is then added to the original 8-bit pixel - so smooth LUTs
introduce no banding and the image keeps its full precision.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_cube(path: str | Path) -> np.ndarray:
    """Parse a .cube file -> float32 array [b, g, r, 3] of RGB values 0..1."""
    size = 0
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    rows: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif up.startswith("LUT_1D_SIZE"):
                raise ValueError("1D LUTs are not supported (need LUT_3D_SIZE)")
            elif up.startswith("DOMAIN_MIN"):
                lo = np.array(line.split()[1:4], np.float32)
            elif up.startswith("DOMAIN_MAX"):
                hi = np.array(line.split()[1:4], np.float32)
            elif line[0].isdigit() or line[0] in "-.":
                rows.append(line)
    if size < 2:
        raise ValueError("missing LUT_3D_SIZE")
    data = np.loadtxt(rows, dtype=np.float32, ndmin=2)[:, :3]
    if data.shape[0] != size ** 3:
        raise ValueError(f"expected {size ** 3} entries, found {data.shape[0]}")
    data = (data - lo) / np.maximum(hi - lo, 1e-6)
    return data.reshape(size, size, size, 3)        # red varies fastest -> index [b][g][r]


def _trilinear(cube: np.ndarray, n: int = 64) -> np.ndarray:
    """Resample cube[b, g, r] to n^3 samples at the centres of the 6-bit bins."""
    size = cube.shape[0]
    c = (np.arange(n, dtype=np.float32) * (256 // n) + (128 // n)) / 255.0 * (size - 1)
    i0 = np.clip(np.floor(c).astype(np.int32), 0, size - 2)
    f = (c - i0).astype(np.float32)
    out = cube
    for axis in range(3):                           # separable linear interpolation
        a = np.take(out, i0, axis=axis)
        b = np.take(out, i0 + 1, axis=axis)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        out = a + (b - a) * f.reshape(shape)
    return out                                      # [b6, g6, r6, RGB]


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=Lut3D.STRIPS, thread_name_prefix="lut")
    return _POOL


class Lut3D:
    N = 64
    STRIPS = 8

    def __init__(self, path: str | Path):
        cube = load_cube(path)
        n = self.N
        res = _trilinear(cube, n)                   # RGB out, indexed [b, g, r]
        centre = (np.arange(n, dtype=np.float32) * (256 // n) + (128 // n))
        b, g, r = np.meshgrid(centre, centre, centre, indexing="ij")
        inp_bgr = np.stack([b, g, r], axis=-1)
        out_bgr = res[..., ::-1] * 255.0
        self._delta = np.round(out_bgr - inp_bgr).astype(np.int16).reshape(-1, 3)
        x = np.arange(256, dtype=np.int32) >> 2
        self._index_lut = np.empty((256, 1, 3), np.int32)
        self._index_lut[:, 0, 0] = x << 12          # B
        self._index_lut[:, 0, 1] = x << 6           # G
        self._index_lut[:, 0, 2] = x                # R
        self.path = str(path)

    def apply(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return frame_bgr
        H = frame_bgr.shape[0]
        out = np.empty_like(frame_bgr)
        n = min(self.STRIPS, max(1, H // 64))
        bounds = [(H * i // n, H * (i + 1) // n) for i in range(n)]

        def work(rows):
            y0, y1 = rows
            src = frame_bgr[y0:y1]
            idx3 = cv2.LUT(src, self._index_lut)                    # int32, one table lookup
            idx = idx3[..., 0] + idx3[..., 1] + idx3[..., 2]
            delta = self._delta.take(idx, axis=0)                   # int16 strip x 3
            cv2.addWeighted(src, 1.0, delta, strength, 0.0, dst=out[y0:y1], dtype=cv2.CV_8U)

        # numpy and OpenCV release the GIL here, so strips run in parallel
        list(_pool().map(work, bounds))
        return out


def write_identity_cube(path: str | Path, size: int = 17, tint=(1.0, 1.0, 1.0), gamma: float = 1.0) -> None:
    """Small helper for tests / a starting point: identity (or tinted) cube."""
    g = np.linspace(0, 1, size, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"ClearFrame\"\nLUT_3D_SIZE {size}\n")
        for b in g:
            for gr in g:
                for r in g:
                    v = [min(1.0, (r ** gamma) * tint[0]), min(1.0, (gr ** gamma) * tint[1]),
                         min(1.0, (b ** gamma) * tint[2])]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
