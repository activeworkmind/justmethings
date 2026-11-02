"""Batch runner: apply one recorded EnhancementRecipe to a whole folder of photos and write upload-ready files.

Output:  <out_root>/<brand-slug>/<name>.jpg|png   (default out_root = <project>/upload_ready)
The default profile 'swiggy_zomato' = JPEG, 1:1, long edge 2048 px, <= 1 MB, no badge / border overlays.
Every exported file is recorded in the SQLite table `uploads`, which the Excel tracker reads.

CLI:  python -m studio.batch --input <folder> --recipe recipe.json  |  --brand day_meal --preset"""
import argparse
import csv
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from PIL import Image

from . import image_engine as E
from .brands import BRANDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "upload_ready"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".dng", ".cr2", ".nef", ".arw"}
ITEM_ID = re.compile(r"^([A-Z]{2}-\d{3})_")


def list_images(folder: Path, recursive: bool = False):
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMG_EXT and not p.name.startswith("_"))


def load_photo(path: Path) -> Image.Image:
    """Open any supported photo (DNG / RAW via the pipeline's decoder when available)."""
    if path.suffix.lower() in {".dng", ".cr2", ".nef", ".arw"}:
        from kitchen.downscale import load_rgb
        return load_rgb(str(path))
    img = Image.open(path)
    from PIL import ImageOps
    return ImageOps.exif_transpose(img).convert("RGB")


def record_upload(info: dict, item_id, brand_key, recipe_json):
    try:
        from kitchen import db
        with closing(db.connect()) as con:
            con.execute("INSERT INTO uploads(path,item_id,brand,bytes,width,height,recipe,exported_at) VALUES(?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET item_id=excluded.item_id,brand=excluded.brand,bytes=excluded.bytes,"
                        "width=excluded.width,height=excluded.height,recipe=excluded.recipe,exported_at=excluded.exported_at",
                        (info["path"], item_id, brand_key, info["bytes"], info["width"], info["height"], recipe_json, time.time()))
            con.commit()
    except (ImportError, sqlite3.Error):
        pass                                                           # tracking is best-effort; the file is already written


def refresh_tracker():
    """Best-effort: rebuild Image_Tracker.xlsx so 'Upload Ready' / 'Ready to upload' reflect the new exports."""
    try:
        from kitchen import excel
        return excel.run()
    except Exception:
        return None


def run_recipe_on_folder(recipe: E.EnhancementRecipe, in_dir, out_root=None, profile: str = "swiggy_zomato",
                         brand: str = None, recursive: bool = False, skip_existing: bool = False,
                         progress=None, files=None) -> dict:
    """Apply `recipe` to every photo; returns a report {ok, failed, skipped, files, errors}."""
    prof = E.PROFILES[profile]
    brand_key = brand or recipe.brand
    rec = recipe.with_ratio(prof.ratio)
    if not prof.allow_overlays:
        rec = rec.without("badge")                                    # logos / borders are stripped for platform uploads
    out_dir = Path(out_root or DEFAULT_OUT) / BRANDS[brand_key].slug
    ext = ".png" if prof.fmt == "PNG" else ".jpg"
    paths = [Path(f) for f in files] if files else list_images(Path(in_dir), recursive)
    report = {"ok": 0, "failed": 0, "skipped": 0, "files": [], "errors": [], "out_dir": str(out_dir)}
    for i, p in enumerate(paths, 1):
        target = out_dir / (p.stem + ext)
        try:
            if skip_existing and target.exists():
                report["skipped"] += 1
            else:
                scene = rec.apply_to(load_photo(p), brand=brand_key)
                img = E.render(scene, prof.long_edge)
                info = E.save_image(img, target, prof)
                m = ITEM_ID.match(p.stem)
                record_upload(info, m.group(1) if m else None, brand_key, rec.to_json())
                report["ok"] += 1
                report["files"].append(info)
        except Exception as e:                                          # one bad photo never stops the batch
            report["failed"] += 1
            report["errors"].append((p.name, f"{type(e).__name__}: {e}"))
        if progress:
            progress(i, len(paths), p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "_batch_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "bytes", "width", "height"])
        for info in report["files"]:
            w.writerow([Path(info["path"]).name, info["bytes"], info["width"], info["height"]])
    if report["ok"]:
        refresh_tracker()
    return report


def main():
    ap = argparse.ArgumentParser(description="Run an enhancement recipe on a folder of photos")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--recipe", help="recipe .json recorded in the app")
    ap.add_argument("--brand", choices=list(BRANDS), help="brand (with --preset, or to override the recipe's brand)")
    ap.add_argument("--preset", action="store_true", help="use the brand's 1-click master preset")
    ap.add_argument("--profile", default="swiggy_zomato", choices=list(E.PROFILES))
    ap.add_argument("--model", default=None, choices=list(E.ISOLATE_MODELS))
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if a.recipe:
        recipe = E.EnhancementRecipe.from_json(Path(a.recipe).read_text(encoding="utf-8"))
    elif a.preset and a.brand:
        recipe = E.master_preset(a.brand, "1:1", model=a.model)
    else:
        ap.error("give --recipe FILE, or --preset together with --brand")
    if a.model:
        recipe.isolate_model = a.model
    rep = run_recipe_on_folder(recipe, a.input, a.out, a.profile, a.brand, a.recursive, a.skip_existing,
                               progress=lambda i, n, name: print(f"[{i}/{n}] {name}"))
    print({k: rep[k] for k in ("ok", "failed", "skipped", "out_dir")})
    for name, err in rep["errors"]:
        print("  FAILED", name, err)


if __name__ == "__main__":
    main()
