"""Main window: live preview, settings tabs, system tray."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
                               QSplitter, QSystemTrayIcon, QTabWidget, QVBoxLayout, QWidget)

from ..engine import Engine
from ..settings import Settings
from .panels import MODE_LABELS, BackgroundTab, CalibrationDialog, CameraTab, OutputTab, SettingsTab
from .widgets import PreviewWidget, app_icon

log = logging.getLogger("vcam.ui")


class MainWindow(QMainWindow):
    calib_progress = Signal(int, int, str)
    calib_done = Signal(object, object)
    engine_error = Signal(str)

    def __init__(self, settings: Settings, engine: Engine):
        super().__init__()
        self.settings = settings
        self.engine = engine
        self.setWindowTitle("vcam-clone")
        self.setWindowIcon(app_icon())
        self.resize(1360, 760)
        self._quitting = False
        self._shown_frame = -1
        self._hidden_since: float | None = None
        self._tray_tip_shown = False
        self._calib_dialog: CalibrationDialog | None = None

        # ---- left: preview + action bar + stats
        self.preview = PreviewWidget()
        self.warn = QLabel("")
        self.warn.setObjectName("warn")
        self.warn.setWordWrap(True)
        self.warn.hide()
        self.stats = QLabel("Starting...")
        self.stats.setObjectName("stats")
        self.stats.setWordWrap(True)

        self.actions_row = QHBoxLayout()
        self.snap_btn = QPushButton("Snapshot")
        self.snap_btn.clicked.connect(self.snapshot)
        self.mirror_btn = QPushButton("Mirror preview")
        self.mirror_btn.setCheckable(True)
        self.mirror_btn.setChecked(settings.mirror_preview)
        self.mirror_btn.toggled.connect(self._mirror_toggled)
        self.actions_row.addWidget(self.snap_btn)
        self.actions_row.addStretch(1)
        self.actions_row.addWidget(self.mirror_btn)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 6, 10)
        lv.addWidget(self.warn)
        lv.addWidget(self.preview, 1)
        lv.addLayout(self.actions_row)
        lv.addWidget(self.stats)

        # ---- right: tabs
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(430)
        self.bg_tab = BackgroundTab(self)
        self.cam_tab = CameraTab(self)
        self.out_tab = OutputTab(self)
        self.settings_tab = SettingsTab(self)
        self.tabs.addTab(self.bg_tab, "Background")
        self.tabs.addTab(self.cam_tab, "Camera")
        self.extra_tabs = []
        self._add_extra_tabs()
        self.tabs.addTab(self.out_tab, "Output")
        self.tabs.addTab(self.settings_tab, "Settings")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        # ---- timers
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save)
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._tick_frame)
        self._frame_timer.start()
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._tick_stats)
        self._stats_timer.start()

        self.calib_progress.connect(self._on_calib_progress)
        self.calib_done.connect(self._on_calib_done)
        self.engine_error.connect(lambda msg: self.stats.setText(msg))
        engine.on_error = self.engine_error.emit

        self._build_tray()
        self._init_extras()

    # ------------------------------------------------------------ extension
    def _add_extra_tabs(self) -> None:
        """Phase 3 tabs (overlays, name tag, framing, lighting, hotkeys)."""
        try:
            from .panels_extra import build_extra_tabs
        except ImportError:
            return
        for title, widget in build_extra_tabs(self):
            self.tabs.addTab(widget, title)
            self.extra_tabs.append(widget)

    def _init_extras(self) -> None:
        try:
            from .panels_extra import init_extras
        except ImportError:
            return
        init_extras(self)

    # --------------------------------------------------------------- tray
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("vcam-clone")
        menu = QMenu()
        self.act_show = QAction("Show window", self)
        self.act_show.triggered.connect(self.show_window)
        menu.addAction(self.act_show)
        menu.addSeparator()
        self.mode_actions = {}
        for mode, label in MODE_LABELS:
            a = QAction(label, self)
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
            menu.addAction(a)
            self.mode_actions[mode] = a
        menu.addSeparator()
        self.tray_menu = menu
        self.act_next_bg = QAction("Next background", self)
        self.act_next_bg.triggered.connect(self.cycle_background)
        menu.addAction(self.act_next_bg)
        self.act_snap = QAction("Snapshot", self)
        self.act_snap.triggered.connect(self.snapshot)
        menu.addAction(self.act_snap)
        self.act_sleep = QAction("Sleep camera", self)
        self.act_sleep.setCheckable(True)
        self.act_sleep.triggered.connect(lambda on: self.engine.set_sleeping(on))
        menu.addAction(self.act_sleep)
        menu.addSeparator()
        quit_a = QAction("Quit", self)
        quit_a.triggered.connect(self.quit)
        menu.addAction(quit_a)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()
        self._sync_mode_ui()

    def _tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            if self.isVisible() and not self.isMinimized():
                self.hide_to_tray()
            else:
                self.show_window()

    def show_window(self) -> None:
        self._hidden_since = None
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if self.engine.sleeping and not self.act_sleep.isChecked():
            self.engine.set_sleeping(False)

    def hide_to_tray(self) -> None:
        self.hide()
        self._hidden_since = time.time()
        if not self._tray_tip_shown:
            self._tray_tip_shown = True
            self.tray.showMessage("vcam-clone is still running",
                                  "The virtual camera keeps working. Right-click the tray icon to quit.",
                                  QSystemTrayIcon.Information, 4000)

    def closeEvent(self, e: QCloseEvent) -> None:
        if self._quitting or not QSystemTrayIcon.isSystemTrayAvailable():
            e.accept()
            return
        e.ignore()
        self.hide_to_tray()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == e.Type.WindowStateChange and self.isMinimized() and not self._quitting:
            QTimer.singleShot(0, self.hide_to_tray)

    def quit(self) -> None:
        self._quitting = True
        self._save()
        self.tray.hide()
        QApplication.instance().quit()

    # ------------------------------------------------------------ settings
    def on_changed(self, request: str | None = None) -> None:
        if request:
            self.engine.request(request)
        self._save_timer.start()

    def _save(self) -> None:
        try:
            self.settings.save()
        except OSError as e:
            log.warning("Could not save settings: %s", e)

    def _mirror_toggled(self, on: bool) -> None:
        self.settings.mirror_preview = on
        self.settings_tab.form.refresh()
        self.on_changed()

    def library_changed(self) -> None:
        self.bg_tab.library.reload()

    # -------------------------------------------------------------- actions
    def _sync_mode_ui(self) -> None:
        mode = self.settings.background.mode
        for m, a in self.mode_actions.items():
            a.setChecked(m == mode)

    def set_mode(self, mode: str) -> None:
        self.settings.background.mode = mode
        self.bg_tab.refresh()
        self._sync_mode_ui()
        self.on_changed()

    def toggle_mode(self, mode: str) -> None:
        """Hotkey behaviour: pressing the active mode's key returns to Original."""
        cur = self.settings.background.mode
        self.set_mode("original" if cur == mode and mode != "original" else mode)

    def set_background_file(self, path: str) -> None:
        self.settings.background.replace_path = path
        self.set_mode("replace")

    def cycle_background(self) -> None:
        files = self.bg_tab.library.files()
        if not files:
            return
        cur = self.settings.background.replace_path
        i = (files.index(cur) + 1) % len(files) if cur in files else 0
        self.set_background_file(files[i])

    def snapshot(self) -> None:
        path = self.engine.snapshot()
        if path:
            self.stats.setText(f"Snapshot saved to {path}")
            self.tray.showMessage("Snapshot saved", str(path), QSystemTrayIcon.Information, 2500)

    def run_calibration(self) -> None:
        if self._calib_dialog is not None:
            return
        self._calib_dialog = CalibrationDialog(self)
        self._calib_dialog.finished.connect(lambda _: setattr(self, "_calib_dialog", None))
        self._calib_dialog.show()
        self.engine.calibrate(self.calib_progress.emit, self.calib_done.emit)

    def _on_calib_progress(self, i: int, n: int, label: str) -> None:
        if self._calib_dialog:
            self._calib_dialog.progress(i, n, label)

    def _on_calib_done(self, results, best) -> None:
        if self._calib_dialog:
            self._calib_dialog.done_(results, best)
        self.settings_tab.form.refresh()
        self.on_changed()

    # ---------------------------------------------------------------- ticks
    def _tick_frame(self) -> None:
        if not self.isVisible():
            return
        bgr, alpha, fg, n = self.engine.latest()
        if bgr is None or n == self._shown_frame:
            return
        self._shown_frame = n
        remove = self.settings.background.mode == "remove"
        self.preview.set_frame(bgr, self.settings.mirror_preview,
                               alpha if remove else None, fg if remove else None)

    def _tick_stats(self) -> None:
        st = self.engine.stats
        self._power_tick()
        if not self.isVisible():
            return
        cap = st.capture
        if st.status == "running":
            parts = [f"{st.fps:.1f} FPS", f"inference {st.infer_ms:.1f} ms", f"frame {st.frame_ms:.1f} ms",
                     f"{st.segmenter} / {st.provider.upper()}"]
            line2 = f"Camera: {st.camera}  {cap.summary()}    Output: {st.output or 'off'}"
            self.stats.setText("   |   ".join(parts) + "\n" + line2)
        elif st.status in ("sleeping", "calibrating", "starting"):
            self.stats.setText(st.status.capitalize() + "...")
        else:
            self.stats.setText(st.status)
        warns = [w.split(" (obs:")[0] for w in st.warnings]
        want = self.settings.camera.fps
        if st.status == "running" and cap.measured_fps and cap.measured_fps < want * 0.8:
            warns.append(f"Camera delivers only {cap.measured_fps:.0f} FPS (requested {want}). "
                         "In low light the sensor uses long exposures: add light or lower exposure.")
        if warns:
            self.warn.setText("  ⚠  " + "\n  ⚠  ".join(warns))
            self.warn.show()
        else:
            self.warn.hide()
        self.cam_tab.set_negotiated(f"Negotiated: {cap.summary()}" if cap.width else "")
        self.out_tab.set_status(f"Active: {st.output}" if st.output else "Virtual camera is off.")
        self.act_sleep.setChecked(self.engine.sleeping)

    def _power_tick(self) -> None:
        """Release the webcam when hidden in the tray and the output is idle,
        or when the sleep timer runs out."""
        if self._hidden_since is None or self.engine.sleeping:
            return
        p = self.settings.power
        idle_output = self.engine.out is None
        hidden_for = time.time() - self._hidden_since
        if (p.release_when_tray and idle_output and hidden_for > 3.0) or \
                (p.sleep_minutes > 0 and hidden_for > p.sleep_minutes * 60):
            self.engine.set_sleeping(True)
