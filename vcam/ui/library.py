"""Local background library grid + optional online search dialog."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from .. import stock
from ..media import MEDIA_EXT, VIDEO_EXT, is_media, thumbnail
from .widgets import bgr_to_qimage

THUMB = (160, 90)


class _Signals(QObject):
    done = Signal(str, object)      # key, payload
    failed = Signal(str, str)


class _Task(QRunnable):
    def __init__(self, key: str, fn):
        super().__init__()
        self.key, self.fn, self.signals = key, fn, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.key, self.fn())
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(self.key, str(e))


def _thumb_qimage(path: str):
    t = thumbnail(path, THUMB)
    return bgr_to_qimage(t) if t is not None else None


class LibraryWidget(QWidget):
    """Thumbnails of every media file in the library folder."""

    picked = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self._tasks = []
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(*THUMB))
        self.list.setGridSize(QSize(THUMB[0] + 14, THUMB[1] + 30))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setWordWrap(False)
        self.list.setMinimumHeight(260)
        self.list.itemClicked.connect(lambda it: self.picked.emit(it.data(Qt.UserRole)))
        self.setAcceptDrops(True)

        add = QPushButton("Add files...")
        add.clicked.connect(self.add_files)
        folder = QPushButton("Open folder")
        folder.clicked.connect(lambda: os.startfile(str(self.settings.library_path())))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        self.online = QPushButton("Search online...")
        self.online.clicked.connect(self.search_online)
        row = QHBoxLayout()
        for b in (add, folder, refresh, self.online):
            row.addWidget(b)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(row)
        v.addWidget(self.list)
        self.empty = QLabel("Drop JPG / PNG / WEBP / GIF / MP4 / WEBM files here, or use Add files.")
        self.empty.setObjectName("hint")
        self.empty.setWordWrap(True)
        v.addWidget(self.empty)
        self.reload()

    def files(self) -> list[str]:
        d = self.settings.library_path()
        return [str(p) for p in sorted(d.iterdir(), key=lambda p: p.name.lower())
                if p.is_file() and p.suffix.lower() in MEDIA_EXT]

    def reload(self) -> None:
        self.list.clear()
        self._tasks.clear()
        placeholder = QPixmap(*THUMB)
        placeholder.fill(Qt.darkGray)
        files = self.files()
        self.empty.setVisible(not files)
        for path in files:
            name = Path(path).name
            it = QListWidgetItem(QIcon(placeholder), ("> " if Path(path).suffix.lower() in VIDEO_EXT else "") + name)
            it.setData(Qt.UserRole, path)
            it.setToolTip(name)
            it.setSizeHint(QSize(THUMB[0] + 10, THUMB[1] + 26))
            self.list.addItem(it)
            task = _Task(path, lambda p=path: _thumb_qimage(p))
            task.signals.done.connect(self._thumb_ready)
            self._tasks.append(task)
            self.pool.start(task)
        self.select(self.settings.background.replace_path)

    def _thumb_ready(self, path: str, img) -> None:
        if img is None:
            return
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == path:
                it.setIcon(QIcon(QPixmap.fromImage(img)))
                break

    def select(self, path: str) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == path:
                self.list.setCurrentItem(it)
                return
        self.list.clearSelection()

    def _import(self, paths: list[str]) -> str | None:
        d = self.settings.library_path()
        last = None
        for src in paths:
            if not is_media(src):
                continue
            dest = d / Path(src).name
            if Path(src).resolve() != dest.resolve():
                shutil.copy2(src, dest)
            last = str(dest)
        self.reload()
        return last

    def add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(MEDIA_EXT))
        paths, _ = QFileDialog.getOpenFileNames(self, "Add backgrounds", "", f"Media ({exts})")
        if paths:
            last = self._import(paths)
            if last:
                self.picked.emit(last)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        last = self._import([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        if last:
            self.picked.emit(last)

    def search_online(self) -> None:
        dlg = StockDialog(self.settings, self)
        if dlg.exec() and dlg.result_path:
            self.reload()
            self.picked.emit(dlg.result_path)


class StockDialog(QDialog):
    """Pexels / Unsplash search. Needs an API key in Settings."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.result_path = ""
        self.setWindowTitle("Search backgrounds online")
        self.resize(760, 560)
        self.pool = QThreadPool(self)
        self._tasks = []
        self._items: dict[str, stock.StockItem] = {}

        self.provider = QComboBox()
        if settings.pexels_api_key:
            self.provider.addItem("Pexels", "pexels")
        if settings.unsplash_api_key:
            self.provider.addItem("Unsplash", "unsplash")
        self.query = QLineEdit()
        self.query.setPlaceholderText("office, bookshelf, beach...")
        self.query.returnPressed.connect(self.run_search)
        go = QPushButton("Search")
        go.clicked.connect(self.run_search)
        top = QHBoxLayout()
        top.addWidget(self.provider)
        top.addWidget(self.query, 1)
        top.addWidget(go)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(*THUMB))
        self.list.setGridSize(QSize(THUMB[0] + 14, THUMB[1] + 30))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.itemDoubleClicked.connect(self.use_item)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        use = QPushButton("Download and use")
        use.clicked.connect(lambda: self.use_item(self.list.currentItem()))
        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(self.list, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(use)
        v.addLayout(bottom)
        if self.provider.count() == 0:
            self.status.setText("Add a Pexels or Unsplash API key in the Settings tab to search online.")
            go.setEnabled(False)
            use.setEnabled(False)

    def _key(self, provider: str) -> str:
        return self.settings.pexels_api_key if provider == "pexels" else self.settings.unsplash_api_key

    def _start(self, key, fn, done):
        t = _Task(key, fn)
        t.signals.done.connect(done)
        t.signals.failed.connect(lambda k, msg: self.status.setText(f"Error: {msg}"))
        self._tasks.append(t)
        self.pool.start(t)

    def run_search(self) -> None:
        q = self.query.text().strip()
        if not q or self.provider.count() == 0:
            return
        prov = self.provider.currentData()
        self.status.setText("Searching...")
        self.list.clear()
        self._items.clear()
        self._start("search", lambda: stock.search(prov, self._key(prov), q), self._results)

    def _results(self, _key, items) -> None:
        self.status.setText(f"{len(items)} results. Double-click to download and use.")
        ph = QPixmap(*THUMB)
        ph.fill(Qt.darkGray)
        for it in items:
            uid = f"{it.provider}:{it.id}"
            self._items[uid] = it
            li = QListWidgetItem(QIcon(ph), it.author[:22])
            li.setData(Qt.UserRole, uid)
            self.list.addItem(li)
            self._start(uid, lambda u=it.thumb_url: stock.fetch_bytes(u), self._thumb)

    def _thumb(self, uid, data) -> None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA)
        for i in range(self.list.count()):
            li = self.list.item(i)
            if li.data(Qt.UserRole) == uid:
                li.setIcon(QIcon(QPixmap.fromImage(bgr_to_qimage(img))))
                break

    def use_item(self, li) -> None:
        if li is None:
            return
        it = self._items.get(li.data(Qt.UserRole))
        if it is None:
            return
        self.status.setText("Downloading...")
        dest = self.settings.library_path()
        self._start("dl", lambda: str(stock.download(it, dest, self._key(it.provider))), self._downloaded)

    def _downloaded(self, _key, path) -> None:
        self.result_path = path
        self.accept()
