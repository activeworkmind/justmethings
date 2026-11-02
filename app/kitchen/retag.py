"""Re-tag photos by the brand of the mat they were shot on (see groups.py: green / royal / plain).

    python -m kitchen retag            preview only - changes nothing
    python -m kitchen retag --apply    back up, re-select, move your saved decisions, fix file-name initials

Only the photos still in your list are used (anything you deleted stays deleted). Your saved decisions
(approve / reject, crop, rotation, tilt, enhance flag, custom name) follow their photo to the corrected item;
the brand initials of the file name are corrected (BR- / SB- / DM-), a custom dish name is kept."""
import json
import os
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from contextlib import closing

from . import catalog, config, db, groups, select, slotstore

ID_TOKEN = re.compile(r"^[A-Za-z][A-Za-z-]*?-?\d{2,3}(?=_|$)")      # BR-001 / DM-045 / "Sunday-biryani-006"


def remaining_photos(con) -> set:
    """Photos still in the list: current ranked picks + saved slots, minus everything you deleted."""
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    rem = {p["photo_id"] for ps in sel.values() for p in ps if p.get("rank")}
    rem |= {r["photo_id"] for r in con.execute("SELECT photo_id FROM slots WHERE status!='deleted'")}
    rem -= {r["photo_id"] for r in con.execute("SELECT photo_id FROM excluded")}
    return rem


def new_name(old_name: str, old_item: str, old_rank: int, new_item: str, new_rank: int, items: dict) -> str:
    """Correct the initials (and rank) of a file name; a default name is regenerated, a custom dish name is kept."""
    old_default = f"{old_item}_{items[old_item]['slug']}_{old_rank}"
    if not old_name or old_name == old_default:
        return f"{new_item}_{items[new_item]['slug']}_{new_rank}"
    name = ID_TOKEN.sub(new_item, old_name, count=1) if ID_TOKEN.match(old_name) else f"{new_item}_{old_name}"
    return re.sub(r"_\d$", f"_{new_rank}", name)


def plan(con) -> list[dict]:
    """What would change: per photo group, the items it feeds now vs. after applying the mat rule."""
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    now = defaultdict(set)
    for item, ps in sel.items():
        for p in ps:
            if p.get("rank"):
                now[p["photo_id"]].add(item)
    rem, idx = remaining_photos(con), groups.photo_index()
    out = []
    for g in groups.load():
        pids = [(n, idx[n]) for n in range(g["n_from"], g["n_to"] + 1) if idx.get(n) in rem]
        if not pids:
            continue
        before = sorted({i for _, p in pids for i in now.get(p, ())})
        after = sorted(groups.expand(g["items"], g["mat"]))
        if not g["mat"]:
            status = "UNDECIDED (mat conflict - not changed)"
        elif not after:
            status = "UNASSIGNED (dish not on this brand's menu)"
        elif set(before) == set(after):
            status = "unchanged"
        else:
            status = "RE-TAGGED"
        out.append({"group": g["group_id"], "mat": g["mat"], "photos": [n for n, _ in pids], "before": before,
                    "after": after if g["mat"] else before, "status": status, "note": g["note"]})
    return out


def apply() -> dict:
    with closing(db.connect()) as con:
        rem = remaining_photos(con)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bk = config.WORK / "backups" / f"retag_{stamp}"
        bk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.SELECTION, bk / "selection.json")
        shutil.copy2(groups.GROUPS_CSV, bk / "photo_groups.csv")
        with closing(sqlite3.connect(bk / "state.db")) as dst:
            con.backup(dst)                                           # consistent copy even while WAL is active
        old = slotstore.load_all()
        select.run(con, scope=rem)
    res = migrate_slots(old)
    res["backup"] = str(bk)
    return res


def migrate_slots(old: dict) -> dict:
    """After a re-selection: move every saved decision (approve / reject, crop, look, custom name) to the slot where
    its photo now lives, correct the initials of its name, and keep the saved file of every slot that did not change."""
    new_sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    where = defaultdict(list)
    for item, ps in new_sel.items():
        for p in ps:
            if p.get("rank"):
                where[p["photo_id"]].append((item, p["rank"]))
    items = {i["item_id"]: i for i in catalog.load()}
    moved, dropped, rows = [], [], {}
    for key, s in old.items():
        if s["status"] == "deleted":
            continue
        old_item, old_rank = key.split(":")
        cands = where.get(s["photo_id"], [])
        if not cands:
            dropped.append((key, s["status"], s["name"]))
            continue
        pick = next((c for c in cands if c[0] == old_item), None) or \
            next((c for c in cands if items[c[0]]["clip_prompt"] == items[old_item]["clip_prompt"]), cands[0])
        new_key = f"{pick[0]}:{pick[1]}"
        if new_key in rows:
            dropped.append((key, s["status"], s["name"]))
            continue
        rows[new_key] = {**s, "name": new_name(s["name"], old_item, int(old_rank), pick[0], pick[1], items)}
        moved.append((key, new_key, s["status"], s["name"], rows[new_key]["name"]))
    with closing(db.connect()) as c:
        c.execute("DELETE FROM slots WHERE status!='deleted'")        # deleted ranks / exclusions are kept as they are
        c.commit()
    keep = {o for o, n, _, a, b in moved if o == n and a == b}       # same slot, same name: its saved file is still right
    for k, s in rows.items():
        slotstore.save(k, s)
        if k in keep:
            slotstore.restore_export(k, old[k])
    for key, s in old.items():                                        # files saved under the old slot / name are stale now
        if s.get("final_path") and s["status"] != "deleted" and key not in keep:
            try:
                os.remove(s["final_path"])
            except OSError:
                pass                                                  # already gone
    return {"moved": moved, "dropped": dropped, "kept_as_is": sum(1 for m in moved if m[0] == m[1])}
