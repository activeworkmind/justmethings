"""Review decisions live in SQLite (table `slots`). Slots are plain dicts in the GUI/finalize code."""
import json
import time
from contextlib import closing

from . import db


def _row_to_slot(r) -> dict:
    return {"photo_id": r["photo_id"], "status": r["status"], "name": r["name"], "rot": r["rot"],
            "tilt": r["tilt"], "crop": json.loads(r["crop"]) if r["crop"] else None,
            "aspect": r["aspect"], "enhance": bool(r["enhance"]), "exported": bool(r["exported"]),
            "exported_enhanced": bool(r["exported_enhanced"]), "final_path": r["final_path"]}


def load_all() -> dict:
    with closing(db.connect()) as con:
        return {r["slot_key"]: _row_to_slot(r) for r in con.execute("SELECT * FROM slots")}


def is_exported(key: str) -> bool:
    with closing(db.connect()) as con:
        row = con.execute("SELECT exported FROM slots WHERE slot_key=?", (key,)).fetchone()
    return bool(row and row["exported"])


def save(key: str, s: dict) -> None:
    """Upsert one slot. A changed decision invalidates the previous export (exported -> 0)."""
    item_id, rank = key.split(":")
    crop = json.dumps([round(v, 5) for v in s["crop"]]) if s.get("crop") else None
    with closing(db.connect()) as con:
        old = con.execute("SELECT * FROM slots WHERE slot_key=?", (key,)).fetchone()
        new_decision = (s["photo_id"], s["status"], s["name"], s["rot"], s["tilt"], crop, bool(s["enhance"]))
        changed = old is None or (old["photo_id"], old["status"], old["name"], old["rot"], old["tilt"],
                                  old["crop"], bool(old["enhance"])) != new_decision
        exported = 0 if changed else (old["exported"] if old else 0)
        con.execute(
            "INSERT INTO slots(slot_key,item_id,rank,photo_id,status,name,rot,tilt,crop,aspect,enhance,reviewed_at,"
            "exported,exported_enhanced,final_path,exported_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(slot_key) DO UPDATE SET photo_id=excluded.photo_id,status=excluded.status,"
            "name=excluded.name,rot=excluded.rot,tilt=excluded.tilt,crop=excluded.crop,aspect=excluded.aspect,"
            "enhance=excluded.enhance,reviewed_at=excluded.reviewed_at,exported=excluded.exported",
            (key, item_id, int(rank), s["photo_id"], s["status"], s["name"], s["rot"], s["tilt"], crop,
             s["aspect"], int(bool(s["enhance"])), time.time(), exported,
             old["exported_enhanced"] if old else 0, old["final_path"] if old else None,
             old["exported_at"] if old else None))
        con.commit()


def mark_exported(key: str, path: str, enhanced: bool) -> None:
    with closing(db.connect()) as con:
        con.execute("UPDATE slots SET exported=1,exported_enhanced=?,final_path=?,exported_at=? WHERE slot_key=?",
                    (int(enhanced), path, time.time(), key))
        con.commit()


def clear_export(key: str) -> None:
    with closing(db.connect()) as con:
        con.execute("UPDATE slots SET exported=0,exported_enhanced=0,final_path=NULL,exported_at=NULL "
                    "WHERE slot_key=?", (key,))
        con.commit()
