"""Extras VCam does not have: .cube LUT and presenter mode (Enhance tab)."""
from __future__ import annotations

from ..presenter import list_monitors
from ..settings import ANCHORS
from .widgets import Form


def add_bonus_rows(f: Form, win) -> None:
    ex = win.settings.extras
    f.section("Colour LUT")
    f.file("LUT file", ex, "lut_path", "Choose a .cube LUT", "3D LUT (*.cube)")
    f.slider("Strength", ex, "lut_strength", 0.0, 1.0, decimals=2)

    f.section("Presenter mode")
    f.check("Show my screen behind me", ex, "presenter")
    f.combo("Screen", ex, "presenter_monitor", list_monitors())
    f.slider("My size", ex, "presenter_scale", 0.15, 1.0, decimals=2)
    f.combo("My position", ex, "presenter_anchor", [(a.replace("-", " ").title(), a) for a in ANCHORS])
    f.hint("Replaces the background with a live capture of the chosen screen and places you, "
           "cut out, in a corner.")
