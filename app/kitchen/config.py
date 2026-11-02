"""Paths and tunables. Everything derives from ROOT so the project can be moved."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ASSETS = ROOT / "assets"          # originals: read-only, never modified
MENU_CSV = ROOT / "menu" / "catalog.csv"
WORK = ROOT / "work"              # derived data, safe to delete and rebuild
OUTPUT = ROOT / "output"

DB_PATH = WORK / "state.db"
DOWNSCALED = WORK / "downscaled"  # every photo, <=1 MB JPEG
REVIEW = WORK / "review"
BACKUPS = WORK / "backups"
LOGS = WORK / "logs"
EMBEDDINGS = WORK / "scores.npz"
SELECTION = WORK / "selection.json"
OVERRIDES = WORK / "overrides.csv"  # optional: item_id,photo_id  (pin a photo)
REVIEW_STATE = WORK / "review_state.json"   # GUI decisions (crop/rotate/enhance/name/status)
FINAL = ROOT / "final_output"                      # hand-reviewed deliverables

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts", ".3gp", ".wmv"}

MAX_BYTES = 1_000_000   # hard cap per image (1 MB, decimal, to be safe)
LONG_EDGE = 2048        # starting long edge in px; shrinks only if needed for the cap
BATCH_SIZE = 25         # DB commit + progress flush every N photos
WORKERS = 4
TOP_N = 3               # photos exported per menu item
ALTERNATES = 5          # extra candidates shown in the review sheet
MIN_PHASH_DISTANCE = 8  # top picks for one item must differ by at least this

CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
REJECT_PROMPTS = {
    "X-person": "a photo of a person or people",
    "X-empty": "a photo of an empty table, empty plate or kitchen counter with no food",
    "X-packaging": "a photo of packaging, a paper bag or a delivery box",
    "X-blur": "a very blurry out of focus photo",
}
