"""Settings tabs: Background, Camera, Output, Settings."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..devices import list_cameras
from ..settings import FPS_CHOICES, RESOLUTIONS
from .library import LibraryWidget
from .widgets import Form

MODE_LABELS = [("original", "Original"), ("blur", "Blur"), ("remove", "Remove"),
               ("replace", "Replace"), ("color", "Colour")]


# ---------------------------------------------------------------------------
class BackgroundTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        s = win.settings
        bs = s.background
        ch = win.on_changed

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.buttons = {}
        for mode, label in MODE_LABELS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, m=mode: win.set_mode(m))
            self.group.addButton(b)
            self.buttons[mode] = b
            row.addWidget(b)

        self.stack = QStackedWidget()
        self.pages = {}
        self.forms: list[Form] = []

        f = Form(ch)
        f.hint("Camera passes through untouched. Camera adjustments, overlays and the name tag still apply.")
        self._add_page("original", f)

        f = Form(ch)
        f.section("Blur")
        f.slider("Strength", bs, "blur_strength", 0, 100)
        f.check("Bokeh (disc-shaped lens blur)", bs, "bokeh")
        f.hint("The blur ignores your own pixels, so there is no coloured halo around you.")
        self._add_page("blur", f)

        f = Form(ch)
        f.section("Remove")
        f.hint("True transparency needs the Unity Capture driver with 'Send alpha (RGBA)' enabled on the "
               "Output tab (use it in OBS). Apps that ignore alpha see the fallback colour instead.")
        f.color("Fallback colour", bs, "remove_fallback_color")
        self._add_page("remove", f)

        f = Form(ch)
        f.section("Library")
        self.library = LibraryWidget(s)
        self.library.picked.connect(win.set_background_file)
        f.widget(self.library)
        f.section("Fit and position")
        f.combo("Scale", bs, "fit", [("Fill (crop)", "fill"), ("Fit (letterbox)", "fit")])
        f.slider("Zoom", bs, "zoom", 0.5, 3.0, decimals=2, suffix="x")
        f.slider("Move X", bs, "move_x", -1.0, 1.0, decimals=2)
        f.slider("Move Y", bs, "move_y", -1.0, 1.0, decimals=2)
        f.check("Flip horizontally", bs, "flip")
        f.section("Adjust background")
        f.slider("Blur", bs, "replace_blur", 0, 100)
        f.slider("Brightness", bs, "brightness", -100, 100)
        f.slider("Contrast", bs, "contrast", -100, 100)
        f.slider("Hue", bs, "hue", -180, 180, suffix="°")
        f.slider("Saturation", bs, "saturation", -100, 100)
        f.button("Reset background edits", self._reset_edits)
        self._add_page("replace", f)

        f = Form(ch)
        f.section("Solid colour")
        f.color("Colour", bs, "color")
        f.buttons(("Chroma green", lambda: self._set_color("#00b140")),
                  ("Chroma blue", lambda: self._set_color("#0047bb")),
                  ("Black", lambda: self._set_color("#000000")),
                  ("White", lambda: self._set_color("#ffffff")))
        f.hint("Chroma green/blue give a clean key in OBS or any editor.")
        self._add_page("color", f)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 8, 0, 0)
        top = QWidget()
        top.setLayout(row)
        row.setContentsMargins(14, 0, 14, 0)
        v.addWidget(top)
        v.addWidget(self.stack, 1)
        self.refresh()

    def _add_page(self, mode: str, form: Form) -> None:
        self.pages[mode] = self.stack.addWidget(form.scrolled())
        self.forms.append(form)

    def _set_color(self, c: str) -> None:
        self.win.settings.background.color = c
        self.refresh()
        self.win.on_changed(None)

    def _reset_edits(self) -> None:
        bs = self.win.settings.background
        bs.fit, bs.zoom, bs.move_x, bs.move_y, bs.flip = "fill", 1.0, 0.0, 0.0, False
        bs.brightness = bs.contrast = bs.hue = bs.saturation = 0.0
        bs.replace_blur = 0
        self.refresh()
        self.win.on_changed(None)

    def refresh(self) -> None:
        mode = self.win.settings.background.mode
        b = self.buttons.get(mode)
        if b:
            b.setChecked(True)
        self.stack.setCurrentIndex(self.pages.get(mode, 0))
        for f in self.forms:
            f.refresh()
        self.library.select(self.win.settings.background.replace_path)


# ---------------------------------------------------------------------------
CAM_RES = [("1920 x 1080 (native 1080p)", (1920, 1080)), ("1280 x 720", (1280, 720)),
           ("2560 x 1440", (2560, 1440)), ("3840 x 2160", (3840, 2160)), ("640 x 480", (640, 480))]

UVC_SLIDERS = [("Zoom", "zoom", 0, 500), ("Pan", "pan", -180, 180), ("Tilt", "tilt", -180, 180),
               ("Focus", "focus", 0, 255), ("Exposure", "exposure", -13, 0)]


class _UVC:
    """Tiny adapter so Form rows can bind to the uvc dict + live camera."""

    def __init__(self, win):
        object.__setattr__(self, "_win", win)

    def __getattr__(self, name):
        win = object.__getattribute__(self, "_win")
        v = win.settings.camera.uvc.get(name)
        if v is None:
            v = win.engine.camera_prop(name)
        return 0 if v is None else v

    def __setattr__(self, name, value):
        win = object.__getattribute__(self, "_win")
        win.settings.camera.uvc[name] = value
        win.engine.camera_prop(name, value)


class CameraTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        cs = win.settings.camera
        f = self.form = Form(win.on_changed)

        f.section("Source")
        self.device = QComboBox()
        self.device.currentIndexChanged.connect(self._device_changed)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload_devices)
        roww = QWidget()
        h = QHBoxLayout(roww)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.device, 1)
        h.addWidget(refresh)
        f.widget(roww, "Camera")
        self.res = QComboBox()
        for text, wh in CAM_RES:
            self.res.addItem(text, wh)
        i = self.res.findData((cs.width, cs.height))
        self.res.setCurrentIndex(max(i, 0))
        self.res.currentIndexChanged.connect(self._res_changed)
        f.widget(self.res, "Capture size")
        f.combo("Capture FPS", cs, "fps", [("30", 30), ("60", 60), ("25", 25), ("24", 24), ("15", 15)], request="camera")
        f.combo("Pixel format", cs, "fourcc", [("MJPG (needed for 1080p on USB 2)", "MJPG"),
                                               ("YUY2 (uncompressed)", "YUY2"), ("NV12", "NV12"), ("H264", "H264")],
                request="camera")
        f.combo("Capture API", cs, "backend", [("DirectShow", "dshow"), ("Media Foundation", "msmf")], request="camera")
        self.negotiated = f.hint("")

        f.section("Image (software)")
        f.slider("Brightness", cs, "brightness", -100, 100)
        f.slider("Contrast", cs, "contrast", -100, 100)
        f.slider("Saturation", cs, "saturation", -100, 100)
        f.slider("Hue", cs, "hue", -180, 180, suffix="°")
        f.slider("Gamma", cs, "gamma", 0.3, 3.0, decimals=2)
        f.button("Reset image", self._reset_image)

        f.section("Hardware (UVC)")
        f.hint("Driver-level controls. Ranges differ per camera; controls a camera does not "
               "support are ignored. The driver dialog shows the exact ranges.")
        uvc = _UVC(win)
        for label, name, lo, hi in UVC_SLIDERS:
            f.slider(label, uvc, name, lo, hi)
        f.buttons(("Auto focus", lambda: self._auto("autofocus", 1)),
                  ("Auto exposure", lambda: self._auto("auto_exposure", 1)),
                  ("Driver dialog...", self._driver_dialog))
        f.button("Forget hardware overrides", self._reset_uvc)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(f.scrolled())
        self.reload_devices()

    def reload_devices(self) -> None:
        cs = self.win.settings.camera
        self.device.blockSignals(True)
        self.device.clear()
        cams = list_cameras()
        for d in cams:
            self.device.addItem(d.name, d.name)
        if not cams:
            self.device.addItem("(no physical camera found)", "")
        i = self.device.findData(cs.device_name)
        self.device.setCurrentIndex(max(i, 0))
        self.device.blockSignals(False)

    def _device_changed(self, i: int) -> None:
        name = self.device.itemData(i) or ""
        if name != self.win.settings.camera.device_name:
            self.win.settings.camera.device_name = name
            self.win.settings.camera.uvc = {}
            self.win.on_changed("camera")

    def _res_changed(self, i: int) -> None:
        cs = self.win.settings.camera
        cs.width, cs.height = self.res.itemData(i)
        self.win.on_changed("camera")

    def _reset_image(self) -> None:
        cs = self.win.settings.camera
        cs.brightness = cs.contrast = cs.saturation = cs.hue = 0.0
        cs.gamma = 1.0
        self.form.refresh()
        self.win.on_changed(None)

    def _auto(self, name: str, value: float) -> None:
        cs = self.win.settings.camera
        cs.uvc.pop("focus" if name == "autofocus" else "exposure", None)
        cs.uvc[name] = value
        self.win.engine.camera_prop(name, value)
        self.win.on_changed(None)

    def _reset_uvc(self) -> None:
        self.win.settings.camera.uvc = {}
        self.win.on_changed("camera")

    def _driver_dialog(self) -> None:
        cap = self.win.engine.cap
        if cap is not None:
            cap.open_driver_dialog()

    def set_negotiated(self, text: str) -> None:
        self.negotiated.setText(text)


# ---------------------------------------------------------------------------
class OutputTab(QWidget):
    def __init__(self, win):
        super().__init__()
        o = win.settings.output
        f = self.form = Form(win.on_changed)
        f.section("Virtual camera")
        f.check("Send to virtual camera", o, "enabled", request="output")
        f.combo("Resolution", o, "resolution", [(k, k) for k in RESOLUTIONS], request="output")
        f.combo("Frame rate", o, "fps", [(f"{x} FPS", x) for x in FPS_CHOICES], request="output")
        f.combo("Driver", o, "backend", [("Auto (OBS, then Unity Capture)", "auto"),
                                         ("OBS Virtual Camera", "obs"),
                                         ("Unity Capture", "unitycapture")], request="output")
        f.check("Send alpha (RGBA, Unity Capture only)", o, "rgba", request="output",
                tooltip="True transparency in OBS when the background mode is Remove.")
        self.status = f.hint("")
        f.section("In your apps")
        f.hint("Pick 'OBS Virtual Camera' (or 'Unity Video Capture') as the camera in Zoom, Meet, "
               "Teams, Discord or OBS. The output is never mirrored; the preview mirror is only for you.")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(f.scrolled())

    def set_status(self, text: str) -> None:
        self.status.setText(text)


# ---------------------------------------------------------------------------
class SettingsTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        s = win.settings
        g = s.seg
        f = self.form = Form(win.on_changed)

        f.section("Performance")
        f.combo("Segmentation", g, "mode", [("Quality - Robust Video Matting", "quality"),
                                            ("Performance - MediaPipe selfie", "performance")], request="segmenter")
        f.combo("Run on", g, "provider", [("Auto (GPU via DirectML)", "auto"), ("GPU (DirectML)", "dml"),
                                          ("CPU", "cpu")], request="segmenter")
        f.combo("GPU", g, "dml_device", [("Adapter 0", 0), ("Adapter 1", 1)], request="segmenter")
        f.combo("RVM precision", g, "rvm_precision", [("fp32", "fp32"), ("fp16", "fp16")], request="segmenter")
        f.combo("Mask upsampling", g, "rvm_refiner", [("Fast guided filter", "guided"),
                                                      ("RVM built-in refiner", "rvm")], request="segmenter")
        f.slider("Model scale", g, "downsample_ratio", 0.15, 0.5, decimals=2, request="segmenter",
                 tooltip="Size of the downscaled copy the model sees. 0.25 is right for 1080p.")
        f.slider("Flicker smoothing", g, "ema", 0.0, 0.9, decimals=2)
        f.slider("Edge tighten", g, "edge_tighten", 0.0, 1.0, decimals=2)
        f.button("Run calibration...", win.run_calibration)
        f.hint("Calibration benchmarks every model and processor on this PC and picks the "
               "best-looking one that holds the output frame rate.")

        f.section("Preview")
        f.check("Mirror preview (output is never mirrored)", s, "mirror_preview")

        f.section("Power saving")
        f.check("Release webcam when in the tray and output is off", s.power, "release_when_tray")
        f.slider("Sleep timer", s.power, "sleep_minutes", 0, 120, suffix=" min",
                 tooltip="Release the webcam after this long in the tray, even while the virtual "
                         "camera is on (it shows the splash image). 0 = never.")
        f.check("Start minimised to the tray", s, "start_minimized")

        f.section("Splash image")
        f.file("Custom image", s.pause, "splash_path", "Choose splash image",
               "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        f.hint("Shown on the virtual camera while the webcam is asleep or off.")

        f.section("Background library")
        f.file("Folder", s, "library_dir", "Choose library folder", folder=True, after=win.library_changed)
        f.text("Pexels API key", s, "pexels_api_key", "optional", password=True)
        f.text("Unsplash access key", s, "unsplash_api_key", "optional", password=True)
        f.hint("Keys are stored only in %APPDATA%\\vcam-clone\\settings.json.")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(f.scrolled())


class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.resize(640, 360)
        self.label = QLabel("Benchmarking models on this PC...")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model / processor", "Per frame", "Model only", "Max FPS"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        v = QVBoxLayout(self)
        v.addWidget(self.label)
        v.addWidget(self.table, 1)
        v.addWidget(self.close_btn, 0, Qt.AlignRight)

    def progress(self, i: int, n: int, label: str) -> None:
        self.label.setText(f"Benchmarking {i + 1}/{n}: {label}" if i < n else "Finishing...")

    def done_(self, results, best) -> None:
        self.table.setRowCount(len(results))
        for r, c in enumerate(results):
            cells = [c.label + ("   <- selected" if c is best else ""),
                     "unavailable" if c.error else f"{c.ms:.1f} ms",
                     "" if c.error else f"{c.infer_ms:.1f} ms",
                     "" if c.error else f"{c.fps:.0f}"]
            for col, text in enumerate(cells):
                self.table.setItem(r, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.label.setText(f"Selected: {best.label}" if best else "No working model found.")
        self.close_btn.setEnabled(True)
