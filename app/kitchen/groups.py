"""Photo groups: ranges of consecutive photos (by contact-sheet number) that show the same dish.

work/photo_groups.csv is the single source of truth. It is first filled with visual best guesses and
is corrected through the Excel tracker (`python -m kitchen apply-excel`). A corrected group has
confidence 'C' (confirmed) and is trusted fully."""
import csv

from . import catalog, config

GROUPS_CSV = config.ROOT / "labels" / "photo_groups.csv"      # versioned in git (hand-made labels)
INDEX_CSV = config.WORK / "label_sheets" / "index.csv"
FIELDS = ["group_id", "n_from", "n_to", "items", "confidence", "note", "confirmed"]
WEIGHT = {"H": 1.0, "M": 0.8, "L": 0.5, "C": 1.0}


def load() -> list[dict]:
    with open(GROUPS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_from"], r["n_to"] = int(r["n_from"]), int(r["n_to"])
        r["items"] = r["items"].split()
        if r["confirmed"].strip().upper() == "Y":
            r["confidence"] = "C"
    return rows


def save(rows: list[dict]) -> None:
    with open(GROUPS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "items": " ".join(r["items"])})


def photo_index() -> dict:
    """contact-sheet number -> photo_id"""
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        return {int(r["n"]): r["photo_id"] for r in csv.DictReader(f)}


def original_names() -> dict:
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        return {int(r["n"]): r["original_file"] for r in csv.DictReader(f)}


def expand(item_ids: list[str]) -> list[str]:
    """Items a group stands for: the ones named + any cooked item that looks identical (same visual prompt)."""
    active = {it["item_id"]: it for it in catalog.active()}
    prompts = {active[i]["clip_prompt"] for i in item_ids if i in active}
    return [i for i, it in active.items() if i in item_ids or it["clip_prompt"] in prompts]
