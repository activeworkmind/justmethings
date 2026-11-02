"""Render the hand-reviewed slots (from the GUI) into final_output/<brand>/<category>/<name>.jpg.

Each approved slot is re-developed from the ORIGINAL file, rotated/tilted, cropped exactly as
chosen in the GUI (crop is stored as fractions, so it is resolution-independent), optionally
given the daylight touch, and encoded <= 1 MB. Every export is recorded in the SQLite `slots`
table (exported / exported_enhanced / final_path), which the Excel tracker reads."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import catalog, config, db, enhance, slotstore
from .downscale import encode_under, load_rgb


def apply_geometry(img: Image.Image, rot: int, tilt: float) -> Image.Image:
    if rot % 4:
        img = img.rotate(-90 * (rot % 4), expand=True)       # +1 = 90 deg clockwise
    if abs(tilt) > 0.01:
        a = np.asarray(img)
        h, w = a.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
        img = Image.fromarray(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REFLECT))
    return img


def auto_crop(img: Image.Image) -> list:
    """Initial FREE-FORM crop guess (fractions x0,y0,x1,y1): the dishes plus a margin, not forced square.
    Falls back to the full frame with a small inset."""
    w, h = img.size
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    box = enhance._dish_cluster(bgr)
    if not box:
        return [0.03, 0.03, 0.97, 0.97]
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.12
    return [max(x0 - mx, 0) / w, max(y0 - my, 0) / h, min(x1 + mx, w) / w, min(y1 + my, h) / h]


def render_image(img: Image.Image, slot: dict, final_edge: int = config.LONG_EDGE) -> Image.Image:
    img = apply_geometry(img, slot.get("rot", 0), slot.get("tilt", 0.0))
    W, H = img.size
    x0, y0, x1, y1 = slot["crop"]
    img = img.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    s = final_edge / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    if slot.get("look"):                                      # brand look (studio): background, shadows, light, ...
        from studio import image_engine as studio
        return studio.apply_look(img, slot["_brand"], slot["look"], min(final_edge, max(img.size)))   # never upscale
    if slot.get("enhance", True):                             # plain daylight touch
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        bgr = enhance._sharpen(enhance._daylight(bgr))
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return img


def brand_key(slug: str) -> str:
    """'biryani-rules' -> 'biryani_rules' (the studio's brand key)."""
    from studio.brands import BRANDS
    return next(k for k, b in BRANDS.items() if b.slug == slug)


def _export_one(job):
    src, slot, out = job
    try:
        img = load_rgb(src)
        s = enhance.WORK_EDGE / max(img.size)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        data = encode_under(render_image(img, slot))
        tmp = Path(out).with_suffix(".tmp")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, out)
        return out, None
    except Exception as e:
        return out, f"{type(e).__name__}: {e}"


def _decision(s: dict) -> tuple:
    """Everything that changes how the final file looks or is named."""
    return (s["photo_id"], s["status"], s["name"], s["rot"], s["tilt"], tuple(s["crop"] or ()), bool(s["enhance"]),
            json.dumps(s.get("look") or []))


def _remove(path) -> bool:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except OSError:
        pass
    return False


def export_slot(key: str) -> dict:
    """Bring the file in final_output/ in line with ONE slot's current decision. Called in the background as soon as
    you approve / change / reject a photo:  approved -> file written (<= 1 MB) under its name;  renamed -> old file
    removed;  rejected / deleted -> file removed.  Safe to call repeatedly (an up-to-date file is left alone)."""
    s = slotstore.load_all().get(key)
    if s is None:
        return {"status": "missing"}
    it = next((i for i in catalog.load() if i["item_id"] == key.split(":")[0]), None)
    if it is None or it["availability"] != "available" or s["status"] != "approved":
        removed = _remove(s["final_path"])
        if s["exported"] or s["final_path"]:
            slotstore.clear_export(key)
        return {"status": "removed" if removed else "nothing to save"}
    out = str(config.FINAL / it["brand"] / it["category"] / f"{s['name']}.jpg")
    if s["exported"] and s["final_path"] == out and os.path.exists(out):
        return {"status": "up to date", "path": out}
    if s["final_path"] and s["final_path"] != out:
        _remove(s["final_path"])                                         # renamed: drop the old file
    with closing(db.connect()) as con:
        src = con.execute("SELECT path FROM files WHERE id=?", (s["photo_id"],)).fetchone()["path"]
    path, err = _export_one((src, {**s, "_brand": brand_key(it["brand"]) if s["look"] else None}, out))
    if err:
        raise RuntimeError(err)
    now = slotstore.load_all().get(key)                                  # did you change it again while we rendered?
    slotstore.mark_exported(key, path, bool(s["enhance"] or s["look"]), valid=bool(now) and _decision(now) == _decision(s))
    return {"status": "saved", "path": path, "bytes": os.path.getsize(path)}


def run(workers: int = 3) -> dict:
    state = slotstore.load_all()
    items = {it["item_id"]: it for it in catalog.load()}
    con = db.connect()
    jobs, keys, removed, skipped = [], [], 0, 0
    for key, s in state.items():
        it = items.get(key.split(":")[0])
        wanted = it is not None and it["availability"] == "available" and s["status"] == "approved"
        if not wanted:
            if s["final_path"] and os.path.exists(s["final_path"]):    # un-approved / rejected after export
                os.remove(s["final_path"])
                removed += 1
            if s["exported"] or s["final_path"]:
                slotstore.clear_export(key)
            continue
        out = str(config.FINAL / it["brand"] / it["category"] / f"{s['name']}.jpg")
        if s["exported"] and s["final_path"] == out and os.path.exists(out):
            skipped += 1                                               # unchanged since last export
            continue
        if s["final_path"] and s["final_path"] != out and os.path.exists(s["final_path"]):
            os.remove(s["final_path"])                                 # renamed: drop the stale file
        src = con.execute("SELECT path FROM files WHERE id=?", (s["photo_id"],)).fetchone()["path"]
        jobs.append((src, {**s, "_brand": brand_key(it["brand"]) if s["look"] else None}, out))
        keys.append(key)
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, job, (out, err) in tqdm(zip(keys, jobs, ex.map(_export_one, jobs)), total=len(jobs), unit="img"):
            if err:
                errors.append((out, err))
            else:
                slotstore.mark_exported(key, out, bool(job[1]["enhance"] or job[1]["look"]))
    rejected = sum(1 for s in state.values() if s["status"] == "rejected")
    return {"exported_now": len(jobs) - len(errors), "unchanged": skipped, "removed": removed,
            "rejected": rejected, "errors": errors}
