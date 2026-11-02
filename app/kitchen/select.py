"""Stage 4: pick the best TOP_N photos per menu item from the labelled photo groups.

For every cooked item: candidate photos = photos in groups that name the item (or an identical-looking one).
score = group confidence weight * (0.5 + 0.5 * technical quality), quality = sharpness + exposure.
The best photo of each group is taken first, so the 3 picks show different shots/angles of the dish."""
import csv
import json
import os
import shutil
from contextlib import closing
from typing import Optional

import imagehash
import numpy as np

from . import catalog, config, db, groups, slotstore

DEST_PLAIN = config.ROOT / "ready_for_crop"


def _quality(rows) -> dict:
    sharp = np.array([r["sharp"] or 0 for r in rows])
    ref = np.log1p(np.percentile(sharp, 90)) or 1.0
    q = {}
    for r in rows:
        s = min(1.0, np.log1p(r["sharp"] or 0) / ref)
        e = 1 - min(1.0, abs((r["lum"] or 0.5) - 0.5) * 2)
        q[r["id"]] = 0.6 * s + 0.4 * e
    return q


def _pins() -> dict:
    pins = {}
    if config.OVERRIDES.exists():
        with open(config.OVERRIDES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pins.setdefault(r["item_id"], []).append(r["photo_id"])
    return pins


def run(con, scope: Optional[set] = None) -> dict:
    """`scope` (optional set of photo ids) limits the candidate photos, e.g. to the ones still in your list."""
    index, grps = groups.photo_index(), groups.load()
    rows = {r["id"]: r for r in con.execute("SELECT * FROM files WHERE status='downscaled'")}
    quality = _quality(list(rows.values()))
    manual = {r["photo_id"]: r["item_id"] for r in con.execute("SELECT photo_id,item_id FROM photo_items")}
    cand = {}                                           # item_id -> {group_id -> [(score, photo_id, conf)]}
    for g in grps:
        for item in groups.expand(g["items"], g["mat"]):
            for n in range(g["n_from"], g["n_to"] + 1):
                pid = index.get(n)
                if pid in rows and pid not in manual and (scope is None or pid in scope):
                    score = groups.WEIGHT[g["confidence"]] * (0.5 + 0.5 * quality[pid])
                    cand.setdefault(item, {}).setdefault(g["group_id"], []).append((score, pid, g["confidence"]))
    active = {it["item_id"] for it in catalog.active()}
    for pid, item in manual.items():                    # your brand tag wins over the group labels: ranked first
        if item in active and pid in rows and (scope is None or pid in scope):
            cand.setdefault(item, {}).setdefault("tag", []).append((10 + quality[pid], pid, "C"))
    pins, items = _pins(), catalog.load()
    excluded = {(r["item_id"], r["photo_id"]) for r in con.execute("SELECT item_id,photo_id FROM excluded")}
    deleted_ranks = {}
    for r in con.execute("SELECT item_id,rank FROM slots WHERE status='deleted'"):
        deleted_ranks.setdefault(r["item_id"], set()).add(r["rank"])
    selection, coverage = {}, []
    for it in items:
        iid = it["item_id"]
        picks, hashes = [], []

        def take(pid, score, conf, gid):
            h = imagehash.hex_to_hash(rows[pid]["phash"])
            if (iid, pid) in excluded or any(h - o < config.MIN_PHASH_DISTANCE for o in hashes) \
                    or any(p["photo_id"] == pid for p in picks):
                return
            picks.append({"photo_id": pid, "score": round(score, 4), "file": rows[pid]["down_path"],
                          "group": gid, "confidence": conf})
            hashes.append(h)

        if it["availability"] == "available":
            for pid in pins.get(iid, []):
                if pid in rows:
                    take(pid, 1.0, "C", "pinned")
            per_group = cand.get(iid, {})
            best = sorted(((max(v), gid) for gid, v in per_group.items()), reverse=True)
            for _, gid in best:                          # round 1: the best shot of each group
                s, pid, conf = max(per_group[gid])
                take(pid, s, conf, gid)
            for _, gid in best:                          # round 2: more shots from the same groups
                for s, pid, conf in sorted(per_group[gid], reverse=True):
                    take(pid, s, conf, gid)
            picks = picks[:config.TOP_N + config.ALTERNATES]
        # ranks stay stable when a slot was deleted: rank 2 deleted -> the picks fill ranks 1 and 3
        free_ranks = [r for r in range(1, config.TOP_N + 1) if r not in deleted_ranks.get(iid, set())]
        for k, p in enumerate(picks):
            p["rank"] = free_ranks[k] if k < len(free_ranks) else None
        selection[iid] = picks
        ranked = [p for p in picks if p["rank"]]
        confs = [p["confidence"] for p in ranked]
        coverage.append({"item_id": iid, "brand": it["brand"], "item": it["item_name"],
                         "cooked": it["availability"] == "available", "photos": len(ranked),
                         "best_confidence": (confs[0] if confs else ""),
                         "flag": ("NOT COOKED YET" if it["availability"] != "available" else
                                  "MISSING" if not picks else
                                  "CHECK" if confs[0] == "L" else "ok")})
    config.SELECTION.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    _export(items, selection)
    export_plain()
    rep = config.OUTPUT / "_reports"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, coverage[0].keys())
        w.writeheader()
        w.writerows(coverage)
    flags = [c["flag"] for c in coverage]
    return {"items": len(items), "ok": flags.count("ok"), "check": flags.count("CHECK"),
            "missing": flags.count("MISSING"), "not_cooked": flags.count("NOT COOKED YET")}


def delete_slot(key: str) -> dict:
    """Delete one photo slot for good (the review GUI's Delete button).

    * database: slot status 'deleted' + the photo is excluded for that dish (never suggested again)
    * files: the copies in output/, ready_for_crop/ and final_output/ are removed
    * the list: the pick is dropped from selection.json and ready_for_crop/_index.csv
    The original photo in assets/ and its 1 MB copy in work/downscaled/ are never touched."""
    item_id, rank = key.split(":")
    rank = int(rank)
    final = slotstore.mark_deleted(key)
    it = next(i for i in catalog.load() if i["item_id"] == item_id)
    name = f"{item_id}_{it['slug']}_{rank}.jpg"
    paths = [config.OUTPUT / it["brand"] / it["category"] / name, DEST_PLAIN / it["brand"] / name]
    if final:
        paths.append(final)
    removed = []
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
            removed.append(str(p))
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    sel[item_id] = [p for p in sel.get(item_id, []) if p.get("rank") != rank]
    config.SELECTION.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    index = DEST_PLAIN / "_index.csv"
    if index.exists():
        with open(index, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["file"] != name]
        if rows:
            with open(index, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    return {"slot": key, "removed": removed}


def _clean(d, item_id):
    for old in d.glob(f"{item_id}_*"):
        old.unlink()


def _export(items, selection):
    shutil.rmtree(config.OUTPUT / "_unmatched", ignore_errors=True)
    for it in items:
        d = config.OUTPUT / it["brand"] / it["category"]
        d.mkdir(parents=True, exist_ok=True)
        _clean(d, it["item_id"])
        for pick in (p for p in selection[it["item_id"]] if p["rank"]):
            shutil.copy2(pick["file"], d / f"{it['item_id']}_{it['slug']}_{pick['rank']}.jpg")


def export_plain(dest=None) -> dict:
    """Uncropped, unenhanced, upright, <=1 MB copies of every pick, grouped by brand only.
    Also writes _index.csv mapping each new file back to its original camera file."""
    from . import db
    dest = dest or DEST_PLAIN
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    con = db.connect()
    src = {r["id"]: r["path"] for r in con.execute("SELECT id,path FROM files")}
    rows, n = [], 0
    for it in catalog.load():
        d = dest / it["brand"]
        d.mkdir(parents=True, exist_ok=True)
        _clean(d, it["item_id"])
        for pick in (p for p in sel.get(it["item_id"], []) if p.get("rank")):
            name = f"{it['item_id']}_{it['slug']}_{pick['rank']}.jpg"
            shutil.copy2(pick["file"], d / name)
            rows.append({"brand": it["brand"], "file": name, "item_id": it["item_id"],
                         "item_name": it["item_name"], "category": it["category"], "rank": pick["rank"],
                         "photo_id": pick["photo_id"], "group": pick.get("group", ""),
                         "confidence": pick.get("confidence", ""), "original_file": src[pick["photo_id"]]})
            n += 1
    with open(dest / "_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return {"files": n, "folder": str(dest)}
