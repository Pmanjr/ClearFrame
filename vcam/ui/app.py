"""Desktop app entry point."""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys

from ..settings import Settings, app_dir


INSTANCE_KEY = "vcam-clone-single-instance"


def _setup_logging(verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(app_dir() / "vcam.log", maxBytes=1_000_000, backupCount=1,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vcam")
    ap.add_argument("--minimized", action="store_true", help="start hidden in the tray")
    ap.add_argument("--input", default=None, help="image/video file instead of a camera (testing)")
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)   # dev: save a UI grab and exit
    ap.add_argument("--tab", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    _setup_logging(a.verbose)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from ..engine import Engine
    from .main_window import MainWindow
    from .widgets import STYLE, app_icon

    app = QApplication(sys.argv[:1])
    app.setApplicationName("vcam-clone")
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())

    # ---- single instance: a second launch just brings the first one forward
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(400):
        sock.write(b"show")
        sock.waitForBytesWritten(400)
        sock.disconnectFromServer()
        logging.getLogger("vcam.ui").info("Already running; activated the existing window.")
        return 0
    QLocalServer.removeServer(INSTANCE_KEY)          # stale name from a crashed run
    server = QLocalServer()
    server.listen(INSTANCE_KEY)

    settings = Settings.load()
    engine = Engine(settings, input_file=a.input)
    win = MainWindow(settings, engine)
    engine.start()

    def _activated() -> None:
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.close()
        win.show_window()
    server.newConnection.connect(_activated)

    if a.minimized or settings.start_minimized:
        win.hide_to_tray()
    else:
        win.show()

    if not settings.seg.calibrated and not a.screenshot:
        QTimer.singleShot(1500, win.run_calibration)     # first run: pick the best model

    if a.screenshot:
        def grab():
            win.tabs.setCurrentIndex(a.tab)
            QTimer.singleShot(700, lambda: (win.grab().save(a.screenshot), win.quit()))
        QTimer.singleShot(6000, grab)

    code = app.exec()
    ulog = logging.getLogger("vcam.ui")
    ulog.info("Event loop ended; stopping engine.")
    server.close()
    engine.stop()                 # releases the webcam and the virtual camera
    settings.save()
    ulog.info("Exit.")
    logging.shutdown()
    _terminate(code)


def _terminate(code: int) -> None:
    """End the process right now. Everything that matters is already done
    (settings saved, webcam released, virtual camera closed). A normal exit
    then spends several seconds unloading GPU / Qt / DirectShow DLLs and can
    be held up by helper threads (hotkey hook, thread pools) - during which
    the app looks closed but is still running."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.TerminateProcess(k32.GetCurrentProcess(), int(code) & 0xFFFFFFFF)
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
