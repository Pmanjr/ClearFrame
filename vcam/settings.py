"""Settings model, persisted as JSON in %APPDATA%\\vcam-clone\\settings.json."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("vcam.settings")

APP_NAME = "vcam-clone"

RESOLUTIONS = {"1080p": (1920, 1080), "720p": (1280, 720), "480p": (854, 480), "360p": (640, 360)}
FPS_CHOICES = (15, 30, 60)
BG_MODES = ("original", "blur", "remove", "replace", "color")
ANCHORS = ("top-left", "top", "top-right", "left", "center", "right",
           "bottom-left", "bottom", "bottom-right")


def app_dir() -> Path:
    p = Path(os.environ.get("APPDATA") or Path.home()) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def pictures_dir() -> Path:
    p = Path.home() / "Pictures" / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def root_dir() -> Path:
    """Folder holding models/ — next to the exe when frozen, repo root otherwise."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return root_dir() / "models"


def default_library_dir() -> Path:
    return app_dir() / "backgrounds"


@dataclass
class CameraSettings:
    device_name: str = ""          # "" = first physical camera
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"
    backend: str = "dshow"         # dshow | msmf
    # software image controls
    brightness: float = 0.0        # -100..100
    contrast: float = 0.0          # -100..100
    saturation: float = 0.0        # -100..100
    hue: float = 0.0               # -180..180 degrees
    gamma: float = 1.0             # 0.3..3.0
    # hardware UVC controls: name -> value (only those the user touched)
    uvc: dict = field(default_factory=dict)


@dataclass
class SegmentationSettings:
    mode: str = "quality"          # quality (RVM) | performance (MediaPipe)
    provider: str = "auto"         # auto | dml | cpu
    dml_device: int = 0
    rvm_precision: str = "fp32"    # fp32 | fp16
    rvm_refiner: str = "guided"    # guided (fast guided filter) | rvm (built-in refiner)
    downsample_ratio: float = 0.25
    ema: float = 0.35              # 0 = off, higher = smoother / laggier
    edge_tighten: float = 0.3      # 0..1 contract the soft edge (kills halo)
    calibrated: bool = False
    target_fps: int = 30


@dataclass
class BackgroundSettings:
    mode: str = "blur"
    blur_strength: int = 55        # 0..100
    bokeh: bool = False
    color: str = "#00b140"         # colour / chroma mode
    remove_fallback_color: str = "#00b140"   # shown by apps that ignore alpha
    replace_path: str = ""
    fit: str = "fill"              # fit | fill
    zoom: float = 1.0
    move_x: float = 0.0            # -1..1 (fraction of frame)
    move_y: float = 0.0
    flip: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    hue: float = 0.0
    saturation: float = 0.0
    replace_blur: int = 0          # 0..100 blur on the replacement background


@dataclass
class OutputSettings:
    resolution: str = "1080p"
    fps: int = 30
    backend: str = "auto"          # auto | obs | unitycapture
    rgba: bool = False             # true transparency (Unity Capture only)
    enabled: bool = True


@dataclass
class OverlaySettings:
    path: str = ""
    enabled: bool = True
    anchor: str = "top-right"
    size: float = 0.15             # fraction of frame width
    offset_x: float = 0.02         # fraction of frame, inwards from anchor
    offset_y: float = 0.02
    opacity: float = 1.0
    flip: bool = False


@dataclass
class NameTagSettings:
    enabled: bool = False
    name: str = "Your Name"
    title: str = ""
    anchor: str = "bottom-left"
    style: str = "bar"             # bar | pill | minimal
    accent: str = "#2d7ff9"
    text_color: str = "#ffffff"
    bg_color: str = "#101418"
    bg_opacity: float = 0.75
    scale: float = 1.0


@dataclass
class AutoFrameSettings:
    enabled: bool = False
    zoom: float = 1.4              # 1..3 (how tight around the face)
    speed: float = 0.5             # 0..1
    limit_native: bool = False     # cap zoom so the crop never drops below output pixels


@dataclass
class LightingSettings:
    enabled: bool = False
    strength: float = 0.5          # 0..1


@dataclass
class PauseSettings:
    loop_seconds: float = 4.0
    loop_buffer: bool = True       # keep the last N seconds ready for Pause Loop
    splash_path: str = ""          # custom idle image ("" = built-in)
    splash_on_pause: bool = False  # show splash instead of the frozen frame


@dataclass
class PowerSettings:
    release_when_tray: bool = True
    sleep_minutes: int = 0         # 0 = never


@dataclass
class HotkeySettings:
    enabled: bool = True
    toggle_blur: str = "Ctrl+Alt+B"
    toggle_remove: str = "Ctrl+Alt+R"
    toggle_original: str = "Ctrl+Alt+O"
    cycle_background: str = "Ctrl+Alt+N"
    pause: str = "Ctrl+Alt+P"
    snapshot: str = "Ctrl+Alt+S"


@dataclass
class ExtrasSettings:
    lut_path: str = ""
    lut_strength: float = 1.0
    presenter: bool = False        # screen capture behind the matted person
    presenter_monitor: int = 1
    presenter_scale: float = 0.35  # person height as fraction of frame
    presenter_anchor: str = "bottom-right"


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    seg: SegmentationSettings = field(default_factory=SegmentationSettings)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    overlays: list[OverlaySettings] = field(default_factory=list)
    nametag: NameTagSettings = field(default_factory=NameTagSettings)
    autoframe: AutoFrameSettings = field(default_factory=AutoFrameSettings)
    lighting: LightingSettings = field(default_factory=LightingSettings)
    pause: PauseSettings = field(default_factory=PauseSettings)
    power: PowerSettings = field(default_factory=PowerSettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    extras: ExtrasSettings = field(default_factory=ExtrasSettings)
    library_dir: str = ""
    pexels_api_key: str = ""
    unsplash_api_key: str = ""
    mirror_preview: bool = True
    start_minimized: bool = False
    close_to_tray: bool = False     # X button: False = quit, True = keep running in the tray
    minimize_to_tray: bool = True   # minimise button hides the window in the tray

    # ---- helpers -------------------------------------------------------
    def output_size(self) -> tuple[int, int]:
        return RESOLUTIONS.get(self.output.resolution, RESOLUTIONS["1080p"])

    def library_path(self) -> Path:
        p = Path(self.library_dir) if self.library_dir else default_library_dir()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # ---- persistence ---------------------------------------------------
    @staticmethod
    def path() -> Path:
        return app_dir() / "settings.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or cls.path()
        if not path.exists():
            return cls()
        try:
            return _from_dict(cls, json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read %s (%s); using defaults", path, e)
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or self.path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)


def _from_dict(cls, data):
    """Tolerant dataclass loader: unknown keys ignored, missing keys default."""
    if not isinstance(data, dict):
        return cls()
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        t = hints[f.name]
        if dataclasses.is_dataclass(t):
            kwargs[f.name] = _from_dict(t, v)
        elif typing.get_origin(t) is list and dataclasses.is_dataclass(typing.get_args(t)[0]):
            item = typing.get_args(t)[0]
            kwargs[f.name] = [_from_dict(item, x) for x in (v or [])]
        else:
            default = f.default if f.default is not dataclasses.MISSING else None
            if default is not None and not isinstance(default, bool) and isinstance(default, (int, float)):
                try:
                    v = type(default)(v)
                except (TypeError, ValueError):
                    continue
            kwargs[f.name] = v
    return cls(**kwargs)


def parse_color(s: str, default=(0, 177, 64)) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r)."""
    try:
        s = s.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (b, g, r)
    except Exception:  # noqa: BLE001
        return (default[2], default[1], default[0])
