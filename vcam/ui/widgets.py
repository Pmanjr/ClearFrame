"""Reusable widgets: live preview, settings-bound form rows, theme."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

ACCENT = "#3d8bfd"

STYLE = f"""
* {{ font-family: 'Segoe UI'; font-size: 10pt; }}
QWidget {{ background: #16191d; color: #e6e8eb; }}
QToolTip {{ background: #2a2f36; color: #e6e8eb; border: 1px solid #3a4048; }}
QTabWidget::pane {{ border: 0; }}
QTabBar::tab {{ background: transparent; padding: 8px 12px; color: #9aa3ad; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: #ffffff; }}
QPushButton {{ background: #262b32; border: 1px solid #343a42; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: #2f353d; }}
QPushButton:pressed {{ background: #22262c; }}
QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton:disabled {{ color: #6b7480; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: #20242a; border: 1px solid #343a42; border-radius: 6px; padding: 5px 8px; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: #20242a; selection-background-color: {ACCENT}; border: 1px solid #343a42; }}
QSlider::groove:horizontal {{ height: 4px; background: #343a42; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4a525c; background: #20242a; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QScrollArea {{ border: 0; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3a4048; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QListWidget {{ background: #1b1f24; border: 1px solid #2a2f36; border-radius: 8px; }}
QListWidget::item {{ border-radius: 6px; padding: 2px; color: #b8c0c9; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; }}
QLabel#section {{ color: #ffffff; font-weight: 600; font-size: 10.5pt; padding-top: 10px; }}
QLabel#hint {{ color: #8b95a1; font-size: 9pt; }}
QLabel#warn {{ background: #4a3413; color: #ffd591; border-radius: 6px; padding: 6px 10px; }}
QLabel#stats {{ color: #9aa3ad; font-size: 9pt; }}
QMenu {{ background: #20242a; border: 1px solid #343a42; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: #343a42; margin: 4px 8px; }}
"""


def app_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(ACCENT))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setBrush(QColor("white"))
    p.drawEllipse(22, 14, 20, 20)
    p.drawChord(QRect(12, 36, 40, 40), 0, 180 * 16)
    p.end()
    return QIcon(pm)


def bgr_to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, w * 3, QImage.Format_BGR888).copy()


class PreviewWidget(QWidget):
    """Shows the composited frame, letterboxed. Mirroring is preview-only."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._checker: np.ndarray | None = None
        self.message = ""

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def _target(self, w: int, h: int) -> tuple[int, int]:
        dpr = self.devicePixelRatioF()
        W, H = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        s = min(W / w, H / h)
        return max(1, int(w * s)), max(1, int(h * s))

    def _checkerboard(self, w: int, h: int) -> np.ndarray:
        if self._checker is None or self._checker.shape[:2] != (h, w):
            yy, xx = np.indices((h, w))
            c = (((xx // 12) + (yy // 12)) % 2 * 40 + 70).astype(np.uint8)
            self._checker = cv2.merge((c, c, c))
        return self._checker

    def set_frame(self, bgr: np.ndarray, mirror: bool = False, alpha: np.ndarray | None = None,
                  fg: np.ndarray | None = None) -> None:
        """If alpha+fg are given (Remove mode) the person is shown over a
        checkerboard so transparency is visible."""
        from ..compositor import blend
        h, w = bgr.shape[:2]
        tw, th = self._target(w, h)
        interp = cv2.INTER_AREA if tw < w else cv2.INTER_LINEAR
        if alpha is not None and fg is not None and alpha.shape[:2] == fg.shape[:2]:
            f = cv2.resize(fg, (tw, th), interpolation=interp)
            a = cv2.resize(alpha, (tw, th), interpolation=interp)
            small = blend(f, self._checkerboard(tw, th), a)
        else:
            small = cv2.resize(bgr, (tw, th), interpolation=interp)
        if mirror:
            small = cv2.flip(small, 1)
        self._img = bgr_to_qimage(small)
        self._img.setDevicePixelRatio(self.devicePixelRatioF())
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0c0e10"))
        if self._img is not None:
            dpr = self._img.devicePixelRatio()
            iw, ih = self._img.width() / dpr, self._img.height() / dpr
            p.drawImage(int((self.width() - iw) / 2), int((self.height() - ih) / 2), self._img)
        if self.message:
            p.setPen(QPen(QColor("#e6e8eb")))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
        p.end()


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, color: str):
        super().__init__()
        self.setFixedSize(54, 28)
        self._color = color
        self._apply()
        self.clicked.connect(self._pick)

    def _apply(self) -> None:
        self.setStyleSheet(f"background: {self._color}; border: 1px solid #4a525c; border-radius: 6px;")

    def set_color(self, c: str) -> None:
        self._color = c
        self._apply()

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if c.isValid():
            self.set_color(c.name())
            self.changed.emit(c.name())


class Form(QWidget):
    """A column of rows bound to attributes on settings dataclasses.

    Every setter writes the attribute, then calls `on_change(request)` where
    request names an engine rebuild ('camera', 'segmenter', 'output') or None.
    """

    def __init__(self, on_change):
        super().__init__()
        self._on_change = on_change
        self._refreshers = []
        self.lay = QFormLayout(self)
        self.lay.setContentsMargins(14, 8, 22, 14)
        self.lay.setHorizontalSpacing(12)
        self.lay.setVerticalSpacing(9)
        self.lay.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    # -- plumbing
    def _changed(self, request=None, after=None) -> None:
        if after:
            after()
        self._on_change(request)

    def refresh(self) -> None:
        """Re-read every bound widget from settings (after external changes)."""
        for fn in self._refreshers:
            fn()

    def scrolled(self) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self)
        v.addStretch(1)
        sa.setWidget(holder)
        return sa

    # -- rows
    def section(self, title: str) -> QLabel:
        lab = QLabel(title)
        lab.setObjectName("section")
        self.lay.addRow(lab)
        return lab

    def hint(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("hint")
        lab.setWordWrap(True)
        self.lay.addRow(lab)
        return lab

    def widget(self, w: QWidget, label: str | None = None) -> QWidget:
        if label is None:
            self.lay.addRow(w)
        else:
            self.lay.addRow(label, w)
        return w

    def separator(self) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2a2f36;")
        self.lay.addRow(line)

    def slider(self, label, obj, attr, lo, hi, decimals=0, request=None, suffix="", after=None,
               tooltip=""):
        k = 10 ** decimals
        sl = QSlider(Qt.Horizontal)
        sl.setRange(int(round(lo * k)), int(round(hi * k)))
        val = QLabel()
        val.setFixedWidth(52)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cast = float if decimals else int

        def show(v):
            val.setText((f"{v / k:.{decimals}f}" if decimals else str(v)) + suffix)

        def on(v):
            setattr(obj, attr, cast(v / k))
            show(v)
            self._changed(request, after)

        def refresh():
            sl.blockSignals(True)
            sl.setValue(int(round(getattr(obj, attr) * k)))
            sl.blockSignals(False)
            show(sl.value())

        refresh()
        sl.valueChanged.connect(on)
        self._refreshers.append(refresh)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(sl, 1)
        h.addWidget(val)
        if tooltip:
            row.setToolTip(tooltip)
        self.lay.addRow(label, row)
        return sl

    def check(self, label, obj, attr, request=None, after=None, tooltip=""):
        cb = QCheckBox(label)

        def on(state):
            setattr(obj, attr, bool(state))
            self._changed(request, after)

        def refresh():
            cb.blockSignals(True)
            cb.setChecked(bool(getattr(obj, attr)))
            cb.blockSignals(False)

        refresh()
        cb.toggled.connect(on)
        self._refreshers.append(refresh)
        if tooltip:
            cb.setToolTip(tooltip)
        self.lay.addRow(cb)
        return cb

    def combo(self, label, obj, attr, items, request=None, after=None, tooltip=""):
        """items: list of (text, value)."""
        cb = QComboBox()
        for text, value in items:
            cb.addItem(text, value)

        def on(i):
            if i >= 0:
                setattr(obj, attr, cb.itemData(i))
                self._changed(request, after)

        def refresh():
            cb.blockSignals(True)
            i = cb.findData(getattr(obj, attr))
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)

        refresh()
        cb.currentIndexChanged.connect(on)
        self._refreshers.append(refresh)
        if tooltip:
            cb.setToolTip(tooltip)
        self.lay.addRow(label, cb)
        return cb

    def text(self, label, obj, attr, placeholder="", password=False, request=None, after=None):
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        if password:
            le.setEchoMode(QLineEdit.PasswordEchoOnEdit)

        def on():
            setattr(obj, attr, le.text().strip() if password else le.text())
            self._changed(request, after)

        def refresh():
            le.blockSignals(True)
            le.setText(str(getattr(obj, attr)))
            le.blockSignals(False)

        refresh()
        le.textEdited.connect(lambda _: on())
        self._refreshers.append(refresh)
        self.lay.addRow(label, le)
        return le

    def color(self, label, obj, attr, request=None, after=None):
        btn = ColorButton(getattr(obj, attr))

        def on(c):
            setattr(obj, attr, c)
            self._changed(request, after)

        btn.changed.connect(on)
        self._refreshers.append(lambda: btn.set_color(getattr(obj, attr)))
        self.lay.addRow(label, btn)
        return btn

    def file(self, label, obj, attr, caption="Choose file", filt="All files (*)", folder=False,
             request=None, after=None):
        le = QLineEdit()
        le.setReadOnly(True)
        le.setPlaceholderText("(none)")
        browse = QPushButton("...")
        browse.setFixedWidth(34)
        clear = QPushButton("x")
        clear.setFixedWidth(30)

        def set_value(v):
            setattr(obj, attr, v)
            le.setText(v)
            self._changed(request, after)

        def pick():
            if folder:
                v = QFileDialog.getExistingDirectory(self, caption, getattr(obj, attr) or "")
            else:
                v, _ = QFileDialog.getOpenFileName(self, caption, getattr(obj, attr) or "", filt)
            if v:
                set_value(v)

        browse.clicked.connect(pick)
        clear.clicked.connect(lambda: set_value(""))
        self._refreshers.append(lambda: le.setText(str(getattr(obj, attr))))
        le.setText(str(getattr(obj, attr)))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(le, 1)
        h.addWidget(browse)
        h.addWidget(clear)
        self.lay.addRow(label, row)
        return le

    def button(self, text, fn, label: str | None = None) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(lambda: fn())
        if label is None:
            self.lay.addRow(b)
        else:
            self.lay.addRow(label, b)
        return b

    def buttons(self, *pairs) -> list[QPushButton]:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        out = []
        for text, fn in pairs:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            h.addWidget(b)
            out.append(b)
        self.lay.addRow(row)
        return out
