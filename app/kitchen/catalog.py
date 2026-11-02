import csv
import re

from . import config


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load() -> list[dict]:
    with open(config.MENU_CSV, newline="", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    for it in items:
        it["slug"] = slug(it["item_name"])
        it.setdefault("availability", "available")
    return items


def active() -> list[dict]:
    """Items that have been cooked/photographed (availability == 'available')."""
    return [it for it in load() if it["availability"] == "available"]
