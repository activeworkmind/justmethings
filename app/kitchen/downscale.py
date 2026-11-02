"""Stage 2: decode every image (DNG included) and downscale it to a JPEG of <= 1 MB.

Workers only compute; the parent process writes the DB so there is a single writer.
Output files are written to a temp name then atomically renamed."""
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import imagehash
import numpy as np
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm

from . import config

Image.MAX_IMAGE_PIXELS = None


def load_rgb(path: str) -> Image.Image:
    ext = Path(path).suffix.lower()
    if ext in config.RAW_EXT:
        with rawpy.imread(path) as raw:
            try:
                arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                return Image.fromarray(arr)
            except Exception:
                # Some DNG variants don't demosaic in LibRaw; the embedded preview is fine for our needs.
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return ImageOps.exif_transpose(Image.open(io.BytesIO(thumb.data))).convert("RGB")
                return Image.fromarray(thumb.data)
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def encode_under(img: Image.Image, max_bytes=config.MAX_BYTES, long_edge=config.LONG_EDGE) -> bytes:
    """Smallest-loss JPEG <= max_bytes: lower quality first, then shrink dimensions."""
    img = _fit(img, long_edge)
    while True:
        for q in (92, 88, 84, 80, 76, 72):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue()
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)


def _fit(img: Image.Image, long_edge: int) -> Image.Image:
    scale = long_edge / max(img.size)
    if scale < 1:
        return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def metrics(img: Image.Image) -> dict:
    small = _fit(img, 512)
    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
    return {
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "lum": float(gray.mean() / 255.0),
        "phash": str(imagehash.phash(small)),
    }


def process(fid: str, src: str) -> dict:
    try:
        img = load_rgb(src)
        data = encode_under(img)
        out = config.DOWNSCALED / f"{Path(src).stem}__{fid}.jpg"
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, out)
        final = Image.open(io.BytesIO(data))
        return {"id": fid, "ok": True, "down_path": str(out), "bytes": len(data),
                "w": final.width, "h": final.height, **metrics(final)}
    except Exception as e:  # one bad file must never stop the batch
        return {"id": fid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def run(con, limit: int | None = None, workers: int = config.WORKERS) -> dict:
    config.DOWNSCALED.mkdir(parents=True, exist_ok=True)
    rows = con.execute(
        "SELECT id,path FROM files WHERE kind='image' AND status IN ('scanned','error') ORDER BY path"
    ).fetchall()
    if limit:
        rows = rows[:limit]
    done = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(total=len(rows), unit="img") as bar:
        futs = [ex.submit(process, r["id"], r["path"]) for r in rows]
        pending = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r["ok"]:
                con.execute(
                    "UPDATE files SET status='downscaled',down_path=?,down_bytes=?,width=?,height=?,"
                    "sharp=?,lum=?,phash=?,error=NULL,updated=strftime('%s','now') WHERE id=?",
                    (r["down_path"], r["bytes"], r["w"], r["h"], r["sharp"], r["lum"], r["phash"], r["id"]))
                done += 1
            else:
                con.execute("UPDATE files SET status='error',error=?,updated=strftime('%s','now') WHERE id=?",
                            (r["error"], r["id"]))
                failed += 1
            pending += 1
            if pending >= config.BATCH_SIZE:
                con.commit()
                pending = 0
            bar.update()
        con.commit()
    return {"downscaled": done, "failed": failed}
