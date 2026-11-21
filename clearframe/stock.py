"""Optional Pexels / Unsplash background search.

API keys come from the user's settings file in %APPDATA% (never the repo).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests

TIMEOUT = 15
UA = {"User-Agent": "clearframe"}


@dataclass
class StockItem:
    provider: str
    id: str
    thumb_url: str
    full_url: str
    author: str = ""
    ping_url: str = ""     # Unsplash asks apps to hit this when a photo is used


def search(provider: str, key: str, query: str, per_page: int = 24) -> list[StockItem]:
    if not key:
        raise ValueError(f"No {provider} API key set (Settings tab).")
    if provider == "pexels":
        r = requests.get("https://api.pexels.com/v1/search",
                         params={"query": query, "per_page": per_page, "orientation": "landscape"},
                         headers={"Authorization": key, **UA}, timeout=TIMEOUT)
        r.raise_for_status()
        return [StockItem("pexels", str(p["id"]), p["src"]["tiny"],
                          p["src"].get("large2x") or p["src"]["original"], p.get("photographer", ""))
                for p in r.json().get("photos", [])]
    if provider == "unsplash":
        r = requests.get("https://api.unsplash.com/search/photos",
                         params={"query": query, "per_page": per_page, "orientation": "landscape"},
                         headers={"Authorization": f"Client-ID {key}", "Accept-Version": "v1", **UA},
                         timeout=TIMEOUT)
        r.raise_for_status()
        out = []
        for p in r.json().get("results", []):
            u = p["urls"]
            out.append(StockItem("unsplash", str(p["id"]), u["thumb"],
                                 u["raw"] + "&w=1920&h=1080&fit=crop&fm=jpg&q=90",
                                 (p.get("user") or {}).get("name", ""),
                                 (p.get("links") or {}).get("download_location", "")))
        return out
    raise ValueError(f"Unknown provider {provider}")


def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def download(item: StockItem, dest_dir: Path, key: str = "") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "", item.id)
    dest = dest_dir / f"{item.provider}-{safe}.jpg"
    if not dest.exists():
        dest.write_bytes(fetch_bytes(item.full_url))
        if item.ping_url and key:
            try:
                requests.get(item.ping_url, headers={"Authorization": f"Client-ID {key}", **UA}, timeout=TIMEOUT)
            except requests.RequestException:
                pass
    return dest
