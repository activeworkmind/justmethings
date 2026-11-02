# Cloud Kitchen - Menu Image Pipeline

Turns a folder of raw phone photos (DNG / JPEG, ~25 GB) into one tidy set of dish photos per menu item,
ready for the Swiggy / Zomato restaurant-partner apps: **upright, cropped by hand, optionally given a natural
daylight touch, <= 1 MB each, named by menu item**. Videos are ignored.

Three brands, 113 menu items (see `menu/catalog.csv`): *Biryani Rules*, *The Day Meal*, *The Sunday Biryani*.

> Photos and generated data are **not** in this repository (they live on Google Drive). This repo holds the
> code, the menu catalogue, the hand-made photo labels and a text export of the database.

## How it works

```
assets/ (originals, read-only)
   |  scan        register images, list + skip videos
   |  downscale   DNG/JPEG -> upright JPEG <= 1 MB            -> work/downscaled/
   |  sheets      numbered contact sheets for labelling        -> work/label_sheets/
   v
labels/photo_groups.csv   ranges of photos = one dish (hand-labelled, correctable in Excel)
   |  select      top 3 photos per cooked item                 -> output/, ready_for_crop/
   v
review GUI        crop (free-form) / rotate / tilt / enhance? / rename / approve   -> SQLite (work/state.db)
   |  finalize    render from the original RAW, <= 1 MB       -> final/<brand>/<category>/
   v
Image_Tracker.xlsx   kept in sync with the database automatically (both directions)
```

* **Labelling** - dish identity comes from `labels/photo_groups.csv` (ranges of consecutive photos with a
  confidence level), not from an AI guess: CLIP could not tell biryani variants apart. Correct a group in the
  Excel *Photo Groups* sheet and save - the app applies it automatically.
* **Availability** - dishes not cooked yet are flagged `not_cooked` in `menu/catalog.csv` and are excluded from
  matching until they are switched to `available`.
* **Enhancement** is photographic only (exposure, daylight warmth, shadow lift, local contrast, sharpen). Food is
  never added, removed or repainted.

### Review GUI (`start_review.bat`)
Free-form crop (drag / corners / wheel; `A` cycles 1:1, 4:3, 3:4), 90 deg rotation, **tilt slider** (-15..+15,
0.1 steps, level grid, auto-level), enhance on/off, rename, other-photo cycling, preview, approve / reject.
Every action is saved to SQLite immediately. Shortcuts are listed at the bottom of the window.

**Reject vs Delete** - *Reject* (`X`) keeps the photo in the list marked "Rejected" (it is not exported to
`final/`). *Delete* (`Del`, red button, asks to confirm) removes it from the list **and** deletes its copies in
`output/`, `ready_for_crop/` and `final/`; that photo is never suggested again for that dish. Ranks stay stable
(delete rank 2 and ranks 1 and 3 keep their files). Originals in `assets/` are never touched.

### Excel <-> database sync
`Image_Tracker.xlsx` (Summary, Menu Items, Photo List, Photo Groups) is rebuilt automatically a few seconds after
every change; corrections typed in *Photo Groups* are applied automatically after you save the workbook. If the
workbook is open in Excel the sync waits and retries. Implemented in `kitchen/sync.py`.

## Food Photo Studio - enhancement + upload-ready export (`start_studio.bat`)

The last step before uploading to Swiggy / Zomato. A Streamlit app (`app/studio/`) that takes the reviewed photos
(`final/`), cuts the dish out, puts it on a brand-themed table and writes **upload-ready files** to
`upload_ready/<brand>/` (JPEG, 1:1, long edge 2048 px, **<= 1 MB**, no logo / badge / border overlays).

* **Brand themes** (`studio/brands.py`): Biryani Rules (royal, dark, amber rim light), The Sunday Biryani (festive
  teak, banana leaf, brass katori), The Day Meal (airy wood, jute runner, window light) - prompt, colour grade,
  vignette and dietary-icon rules per brand.
* **10 enhancements**, stackable in any order, each adjustable: dish isolation (rembg), AI theme background
  (free HuggingFace API, with a local-template and an offline procedural fallback), grounded dual shadows,
  natural sunlight (+ optional IC-Light), shahi rim & candlelight glow, angle / tilt / framing fix (dish at
  74-78 % of the width), LAB "masala pop", steam, DSLR bokeh, brand badge & dietary icon - plus a brand colour grade.
* **1-click brand master preset**, **Undo / Redo / Reset / before-after slider** per photo; the cut-out is cached
  so lighting / framing / background tweaks are instant.
* **Recipes**: every click is recorded; "Run recipe on folder" (Batch tab or `python -m studio.batch`) applies it to
  a whole folder. Every export is logged in the database and shows in the Excel tracker (*Upload Ready* sheet,
  *Ready to upload* column).
* Engine: `studio/image_engine.py` (pure OpenCV / numpy / PIL), UI: `studio/app.py`, batch: `studio/batch.py`.

The upload profile strips badges and borders because item photos are normally checked for logos / watermarks -
confirm the current Swiggy / Zomato image rules in your partner app (ratio, minimum size) before the first upload.

## Setup (Windows, Python 3.12)

```powershell
cd app
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Restore the data from Google Drive into the same folder layout (`assets/`, `work/`, ...), or rebuild it:

```powershell
.\.venv\Scripts\python -m kitchen scan          # needs assets/
.\.venv\Scripts\python -m kitchen downscale
.\.venv\Scripts\python -m kitchen sheets
.\.venv\Scripts\python -m kitchen db-restore    # review decisions from db/state.sql (optional)
.\.venv\Scripts\python -m kitchen select
```

## Commands (`python -m kitchen <command>`)

| Command | Purpose |
|---|---|
| `scan` / `downscale` / `sheets` | register, downscale (<= 1 MB) and contact-sheet the photos |
| `select` | pick the top-3 photos per cooked item from the labelled groups |
| `gui` | review app (or `start_review.bat`) |
| `finalize` | render approved slots to `final/` (or `export_final.bat`) |
| `excel` / `apply-excel` | manual sync (normally automatic) |
| `export-plain` | uncropped, upright, named copies per brand -> `ready_for_crop/` |
| `db-dump` / `db-restore` | text export / rebuild of the SQLite database (`db/state.sql`) |
| `status` | progress counts |
| `python -m studio.batch --input <folder> --brand <key> --preset` | headless studio batch (keys: `biryani_rules`, `sunday_biryani`, `day_meal`) |
| `classify` | optional CLIP hints (needs torch, see `app/requirements.txt`) |

## Repository layout

```
app/kitchen/   Python package        app/tests/   pytest suite (python -m pytest -q)
menu/          menu PDFs + catalog.csv (113 items, availability)
labels/        photo_groups.csv (hand-made labels)
db/            state.sql (text export of the database)
doc/           design notes
*.bat          start_review.bat, export_final.bat
```

Not in git: `assets/`, `work/`, `output/`, `final/`, `ready_for_crop/`, `Image_Tracker.xlsx`, `.venv/`.
