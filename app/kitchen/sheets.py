"""Numbered contact sheets of all photos (sorted by original file name, so shooting bursts sit
together) for visual labelling. index -> photo_id map is written to work/label_sheets/index.csv."""
import csv
import os

from PIL import Image, ImageDraw, ImageFont

from . import config, db

COLS, ROWS, T = 6, 4, 330


def run() -> dict:
    out = config.WORK / "label_sheets"
    out.mkdir(parents=True, exist_ok=True)
    con = db.connect()
    rows = sorted(con.execute("SELECT id,path,down_path FROM files WHERE status='downscaled'"),
                  key=lambda r: os.path.basename(r["path"]).lower())
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "photo_id", "original_file"])
        for n, r in enumerate(rows, 1):
            w.writerow([n, r["id"], os.path.basename(r["path"])])
    try:
        font = ImageFont.truetype("arialbd.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    per = COLS * ROWS
    for s in range(0, len(rows), per):
        sheet = Image.new("RGB", (COLS * T, ROWS * T), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows[s:s + per]):
            im = Image.open(r["down_path"]).convert("RGB")
            im.thumbnail((T - 6, T - 6))
            x, y = (i % COLS) * T, (i // COLS) * T
            sheet.paste(im, (x + (T - im.width) // 2, y + (T - im.height) // 2))
            d.rectangle((x + 2, y + 2, x + 78, y + 42), fill=(0, 0, 0))
            d.text((x + 6, y + 2), str(s + i + 1), fill=(255, 235, 59), font=font)
        sheet.save(out / f"sheet_{s // per + 1:02d}.jpg", quality=88)
    return {"photos": len(rows), "sheets": -(-len(rows) // per), "folder": str(out)}
