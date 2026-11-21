"""Phase 3 UI: overlays + name tag, enhance (auto-framing, lighting, pause),
global hotkeys. Loaded by MainWindow if present."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QKeySequenceEdit, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from .. import pause as pause_mod
from ..hotkeys import ACTIONS, HotkeyManager
from ..media import MEDIA_EXT
from ..settings import ANCHORS, OverlaySettings
from .widgets import Form

ANCHOR_ITEMS = [(a.replace("-", " ").title(), a) for a in ANCHORS]


# ---------------------------------------------------------------------------
class OverlaysTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        s = win.settings

        head = QLabel("Logos and watermarks")
        head.setObjectName("section")
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._item_toggled)
        add = QPushButton("Add...")
        add.clicked.connect(self._add)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove)
        up = QPushButton("Up")
        up.clicked.connect(lambda: self._move(-1))
        down = QPushButton("Down")
        down.clicked.connect(lambda: self._move(1))
        row = QHBoxLayout()
        for b in (add, rem, up, down):
            row.addWidget(b)

        self.layer_holder = QVBoxLayout()
        self.layer_holder.setContentsMargins(0, 0, 0, 0)
        self.layer_form: Form | None = None

        nt = s.nametag
        f = self.tag_form = Form(win.on_changed)
        f.section("Name tag")
        f.check("Show name tag", nt, "enabled")
        f.text("Name", nt, "name")
        f.text("Title", nt, "title", "optional")
        f.combo("Position", nt, "anchor", ANCHOR_ITEMS)
        f.combo("Style", nt, "style", [("Bar", "bar"), ("Pill", "pill"), ("Minimal", "minimal")])
        f.slider("Size", nt, "scale", 0.5, 2.0, decimals=2, suffix="x")
        f.color("Accent", nt, "accent")
        f.color("Text", nt, "text_color")
        f.color("Background", nt, "bg_color")
        f.slider("Background opacity", nt, "bg_opacity", 0.0, 1.0, decimals=2)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(14, 8, 22, 0)
        tv.addWidget(head)
        tv.addWidget(self.list)
        tv.addLayout(row)
        tv.addLayout(self.layer_holder)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.addWidget(top)
        iv.addWidget(f)
        iv.addStretch(1)
        from PySide6.QtWidgets import QScrollArea
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(inner)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(sa)
        self._reload(0)

    # -- layer list
    def _layers(self) -> list[OverlaySettings]:
        return self.win.settings.overlays

    def _reload(self, select: int = -1) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for l in self._layers():
            it = QListWidgetItem(Path(l.path).name or "(empty)")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if l.enabled else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self._layers():
            self.list.setCurrentRow(max(0, min(select, len(self._layers()) - 1)))
        self._select(self.list.currentRow())

    def _item_toggled(self, it: QListWidgetItem) -> None:
        i = self.list.row(it)
        if 0 <= i < len(self._layers()):
            self._layers()[i].enabled = it.checkState() == Qt.Checked
            self.win.on_changed()

    def _select(self, i: int) -> None:
        if self.layer_form is not None:
            self.layer_holder.removeWidget(self.layer_form)
            self.layer_form.deleteLater()
            self.layer_form = None
        if not (0 <= i < len(self._layers())):
            return
        l = self._layers()[i]
        f = self.layer_form = Form(self.win.on_changed)
        f.lay.setContentsMargins(0, 6, 0, 0)
        f.combo("Anchor", l, "anchor", ANCHOR_ITEMS)
        f.slider("Size", l, "size", 0.02, 1.0, decimals=2)
        f.slider("Offset X", l, "offset_x", -0.5, 0.5, decimals=2)
        f.slider("Offset Y", l, "offset_y", -0.5, 0.5, decimals=2)
        f.slider("Opacity", l, "opacity", 0.0, 1.0, decimals=2)
        f.check("Flip horizontally", l, "flip")
        self.layer_holder.addWidget(f)

    def _add(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(MEDIA_EXT))
        paths, _ = QFileDialog.getOpenFileNames(self, "Add overlay (PNG / GIF / MP4...)", "", f"Media ({exts})")
        for p in paths:
            self._layers().append(OverlaySettings(path=p))
        if paths:
            self._reload(len(self._layers()) - 1)
            self.win.on_changed()

    def _remove(self) -> None:
        i = self.list.currentRow()
        if 0 <= i < len(self._layers()):
            del self._layers()[i]
            self._reload(i)
            self.win.on_changed()

    def _move(self, d: int) -> None:
        i = self.list.currentRow()
        j = i + d
        L = self._layers()
        if 0 <= i < len(L) and 0 <= j < len(L):
            L[i], L[j] = L[j], L[i]
            self._reload(j)
            self.win.on_changed()


# ---------------------------------------------------------------------------
class EnhanceTab(QWidget):
    def __init__(self, win):
        super().__init__()
        s = win.settings
        f = self.form = Form(win.on_changed)

        f.section("Auto-framing")
        f.check("Follow my face", s.autoframe, "enabled")
        f.slider("Zoom", s.autoframe, "zoom", 1.0, 3.0, decimals=2, suffix="x")
        f.slider("Speed", s.autoframe, "speed", 0.0, 1.0, decimals=2)
        f.check("Never upscale (cap zoom at native pixels)", s.autoframe, "limit_native",
                tooltip="Limits the zoom so the crop never has fewer pixels than the output. "
                        "With a 1080p camera this needs a 720p or smaller output to zoom at all.")
        f.hint("The crop is taken from the full-resolution camera frame. A warning appears above "
               "the preview when the zoom leaves fewer real pixels than the output resolution.")

        f.section("Enhance lighting")
        f.check("Auto gamma, local contrast and white balance", s.lighting, "enabled")
        f.slider("Strength", s.lighting, "strength", 0.0, 1.0, decimals=2)

        f.section("Pause")
        f.slider("Loop length", s.pause, "loop_seconds", 1.0, 10.0, decimals=1, suffix=" s")
        f.check("Keep the last seconds buffered for Pause Loop", s.pause, "loop_buffer",
                tooltip="Off saves a little CPU; Pause Loop then falls back to a frozen frame.")
        f.check("Show the splash image while paused", s.pause, "splash_on_pause")
        f.hint("Pause Camera freezes the frame. Pause Loop replays the last seconds forwards then "
               "backwards, so there is no jump. Both buttons are under the preview.")
        self._extras(f, win)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(f.scrolled())

    def _extras(self, f: Form, win) -> None:
        """Extras: .cube LUT and presenter mode."""
        try:
            from .panels_bonus import add_bonus_rows
        except ImportError:
            return
        add_bonus_rows(f, win)


# ---------------------------------------------------------------------------
class _HotkeyBridge(QObject):
    fired = Signal(str)


def _add_hotkey_rows(win) -> None:
    hk = win.settings.hotkeys
    f: Form = win.settings_tab.form
    f.section("Global hotkeys")
    f.check("Enable global hotkeys", hk, "enabled", after=win.apply_hotkeys)
    win._hotkey_edits = {}
    for name, label in ACTIONS:
        edit = QKeySequenceEdit(QKeySequence(getattr(hk, name)))
        edit.setMaximumSequenceLength(1)
        edit.setClearButtonEnabled(True)

        def changed(seq, n=name):
            setattr(hk, n, seq.toString(QKeySequence.PortableText))
            win.apply_hotkeys()
            win.on_changed()

        edit.keySequenceChanged.connect(changed)
        f.widget(edit, label)
        win._hotkey_edits[name] = edit
    win.hotkey_status = f.hint("Work while any app is focused. Click a field and press the keys.")


def build_extra_tabs(win):
    win.overlays_tab = OverlaysTab(win)
    win.enhance_tab = EnhanceTab(win)
    return [("Overlays", win.overlays_tab), ("Enhance", win.enhance_tab)]


def init_extras(win) -> None:
    eng = win.engine

    # ---- pause buttons under the preview
    win.pause_btn = QPushButton("Pause camera")
    win.pause_btn.setCheckable(True)
    win.loop_btn = QPushButton("Pause loop")
    win.loop_btn.setCheckable(True)

    def set_pause(mode: str) -> None:
        eng.set_pause(mode)
        win.pause_btn.setChecked(mode == pause_mod.FREEZE)
        win.loop_btn.setChecked(mode == pause_mod.LOOP)
        win.act_pause.setChecked(mode != pause_mod.LIVE)

    win.set_pause = set_pause
    win.pause_btn.clicked.connect(lambda on: set_pause(pause_mod.FREEZE if on else pause_mod.LIVE))
    win.loop_btn.clicked.connect(lambda on: set_pause(pause_mod.LOOP if on else pause_mod.LIVE))
    win.actions_row.insertWidget(0, win.pause_btn)
    win.actions_row.insertWidget(1, win.loop_btn)

    win.act_pause = QAction("Pause camera", win)
    win.act_pause.setCheckable(True)
    win.act_pause.triggered.connect(lambda on: set_pause(pause_mod.FREEZE if on else pause_mod.LIVE))
    win.tray_menu.insertAction(win.act_snap, win.act_pause)

    # ---- hotkeys
    bridge = win._hotkey_bridge = _HotkeyBridge(win)

    def on_hotkey(name: str) -> None:
        if name == "toggle_blur":
            win.toggle_mode("blur")
        elif name == "toggle_remove":
            win.toggle_mode("remove")
        elif name == "toggle_original":
            win.set_mode("original")
        elif name == "cycle_background":
            win.cycle_background()
        elif name == "pause":
            set_pause(pause_mod.LIVE if eng.stats.pause != pause_mod.LIVE else pause_mod.FREEZE)
        elif name == "snapshot":
            win.snapshot()

    bridge.fired.connect(on_hotkey)          # queued onto the UI thread
    win.hotkeys = HotkeyManager(bridge.fired.emit)

    def apply_hotkeys() -> None:
        problems = win.hotkeys.apply(win.settings.hotkeys)
        if hasattr(win, "hotkey_status"):
            win.hotkey_status.setText("Problem: " + "; ".join(problems) if problems else
                                      "Work while any app is focused. Click a field and press the keys.")

    win.apply_hotkeys = apply_hotkeys
    _add_hotkey_rows(win)
    apply_hotkeys()
