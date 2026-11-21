"""Global hotkeys (pynput). Shortcuts are stored in Qt's portable text form
('Ctrl+Alt+B') and translated to pynput's ('<ctrl>+<alt>+b')."""
from __future__ import annotations

import logging

log = logging.getLogger("clearframe.hotkeys")

ACTIONS = [
    ("toggle_blur", "Toggle blur"),
    ("toggle_remove", "Toggle remove"),
    ("toggle_original", "Original (no effect)"),
    ("cycle_background", "Next background"),
    ("pause", "Pause / resume camera"),
    ("snapshot", "Snapshot"),
]

_MODS = {"ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>", "meta": "<cmd>", "win": "<cmd>"}
_KEYS = {"space": "<space>", "return": "<enter>", "enter": "<enter>", "tab": "<tab>", "esc": "<esc>",
         "escape": "<esc>", "backspace": "<backspace>", "del": "<delete>", "delete": "<delete>",
         "ins": "<insert>", "insert": "<insert>", "home": "<home>", "end": "<end>",
         "pgup": "<page_up>", "pgdown": "<page_down>", "up": "<up>", "down": "<down>",
         "left": "<left>", "right": "<right>", "print": "<print_screen>", "pause": "<pause>"}


def to_pynput(shortcut: str) -> str | None:
    """'Ctrl+Alt+B' -> '<ctrl>+<alt>+b'. None if empty/unsupported."""
    s = shortcut.strip()
    if not s:
        return None
    if s.startswith("<"):                    # already pynput syntax
        return s
    parts = [p.strip() for p in s.replace("++", "+plus").split("+") if p.strip()]
    out = []
    for p in parts:
        low = p.lower()
        if low in _MODS:
            out.append(_MODS[low])
        elif low in _KEYS:
            out.append(_KEYS[low])
        elif low == "plus":
            out.append("+")
        elif len(low) > 1 and low[0] == "f" and low[1:].isdigit():
            out.append(f"<{low}>")
        elif len(p) == 1:
            out.append(low)
        else:
            return None
    if not out or all(o in _MODS.values() for o in out):
        return None
    return "+".join(out)


class HotkeyManager:
    """callback(action_name) is invoked on pynput's thread."""

    def __init__(self, callback):
        self._callback = callback
        self._listener = None

    def apply(self, hk) -> list[str]:
        """(Re)register from a HotkeySettings. Returns the problems found."""
        self.stop()
        problems: list[str] = []
        if not hk.enabled:
            return problems
        try:
            from pynput import keyboard
        except Exception as e:  # noqa: BLE001
            return [f"pynput unavailable: {e}"]
        mapping = {}
        for name, label in ACTIONS:
            raw = getattr(hk, name, "")
            if not raw:
                continue
            combo = to_pynput(raw)
            if combo is None:
                problems.append(f"{label}: unsupported shortcut '{raw}'")
                continue
            try:
                keyboard.HotKey.parse(combo)
            except ValueError:
                problems.append(f"{label}: unsupported shortcut '{raw}'")
                continue
            if combo in mapping:
                problems.append(f"{label}: '{raw}' is already used")
                continue
            mapping[combo] = (lambda n=name: self._callback(n))
        if mapping:
            try:
                self._listener = keyboard.GlobalHotKeys(mapping)
                self._listener.daemon = True
                self._listener.start()
            except Exception as e:  # noqa: BLE001
                problems.append(f"could not register hotkeys: {e}")
        for p in problems:
            log.warning("Hotkey: %s", p)
        return problems

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass
            self._listener = None
