"""Entry point.

  python -m vcam                 -> desktop app (UI)
  python -m vcam cli [flags]     -> headless pipeline
  python -m vcam --list-cameras  -> list cameras
"""
from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "cli":
        from .cli import main as cli_main
        return cli_main(argv[1:])
    if argv and argv[0] in ("--list-cameras", "--benchmark"):
        from .cli import main as cli_main
        return cli_main(argv)
    try:
        from .ui.app import main as ui_main
    except ImportError as e:
        print(f"UI not available ({e}). Use: python -m vcam cli --help")
        return 2
    return ui_main(argv)


if __name__ == "__main__":
    sys.exit(main())
