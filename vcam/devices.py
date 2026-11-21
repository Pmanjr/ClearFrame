"""Camera enumeration (DirectShow), with virtual cameras filtered out."""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("vcam.devices")
logging.getLogger("comtypes").setLevel(logging.WARNING)

# Substrings (lower-case) that mark a device as a virtual camera.
VIRTUAL_MARKERS = (
    "obs virtual", "obs-camera", "unity video capture", "vcam", "xsplit",
    "virtual cam", "virtual camera", "snap camera", "ndi ", "newtek",
    "manycam", "droidcam", "iriun", "epoccam", "camo", "mmhmm", "streamlabs",
    "nvidia broadcast", "splitcam", "e2esoft", "chromacam", "avatarify",
)


@dataclass(frozen=True)
class CameraDevice:
    index: int       # DirectShow index, as used by cv2.VideoCapture(index, CAP_DSHOW)
    name: str
    virtual: bool


def is_virtual(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in VIRTUAL_MARKERS)


def list_cameras(include_virtual: bool = False) -> list[CameraDevice]:
    """DirectShow video inputs in cv2 index order."""
    names: list[str] = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception as e:  # noqa: BLE001
        log.warning("DirectShow enumeration failed (%s); probing indices", e)
        import cv2
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                names.append(f"Camera {i}")
            else:
                names.append("")
            cap.release()
    devs = [CameraDevice(i, n, is_virtual(n)) for i, n in enumerate(names) if n]
    return devs if include_virtual else [d for d in devs if not d.virtual]


def resolve_camera(name: str = "", index: int | None = None) -> CameraDevice | None:
    """Pick by explicit index, else by saved name, else first physical camera."""
    if index is not None and index >= 0:
        for d in list_cameras(include_virtual=True):
            if d.index == index:
                return d
        return CameraDevice(index, f"Camera {index}", False)
    cams = list_cameras()
    if name:
        for d in cams:
            if d.name == name:
                return d
    return cams[0] if cams else None
