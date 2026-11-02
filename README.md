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
   |  Approve     saves the final photo straight away (from the original RAW, <= 1 MB)
   v            -> final_output/<brand>/<category>/<name>.jpg
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

### The review app (`start.bat`)
Free-form crop (drag / corners / wheel; `A` cycles 1:1, 4:3, 3:4), 90 deg rotation, **tilt slider** (-15..+15,
0.1 steps, level grid, auto-level), enhance on/off, rename, other-photo cycling, preview, approve / reject.
Every action is saved to SQLite immediately. Shortcuts are listed at the bottom of the window.

**Approve saves the photo.** *Approve + Next* writes the final file to `final_output/<brand>/<category>/<name>.jpg`
in the background (crop, rotation, tilt, look, <= 1 MB, the name you typed), so the next photo opens instantly and the
green status in the bottom bar tells you when it is written ("Saved ... KB"). Change an approved photo, rename it or reject
it and the file follows: replaced, renamed or removed. Anything approved earlier but not yet written is saved when the
app starts, and closing the window waits for pending saves. "Open final_output folder" shows the result.

**Look... (brand styling)** - per photo, optional. One small dialog with a checkbox per enhancement: brand table
background (Biryani Rules = dark rosewood, Sunday Biryani = festive teak, Day Meal = airy wood + jute), grounded
shadows, natural sunlight, rim / candlelight glow, straighten & frame, masala colour pop, steam, background blur,
brand colour grade and an optional brand badge + veg / egg / non-veg icon. The brand is taken from the item
(BR- / SB- / DM-), so you never pick it. "Brand preset" ticks the recommended set; every step is just a checkbox,
so nothing is destructive. The look is applied after your crop, at export time, and the file is still <= 1 MB.
Engine: `app/studio/` (OpenCV / numpy / PIL + `rembg` for the dish cut-out; backgrounds are generated offline, or
with the free HuggingFace API when you set an `HF_TOKEN`). The badge is off by default - Swiggy / Zomato usually
reject logos on item photos, so confirm their current image rules before using it.

**Reject vs Delete** - *Reject* (`X`) keeps the photo in the list marked "Rejected" (it is not exported to
`final_output/`). *Delete* (`Del`, red button, asks to confirm) removes it from the list **and** deletes its copies in
`output/`, `ready_for_crop/` and `final_output/`; that photo is never suggested again for that dish. Ranks stay stable
(delete rank 2 and ranks 1 and 3 keep their files). Originals in `assets/` are never touched.

### Excel <-> database sync
`Image_Tracker.xlsx` (Summary, Menu Items, Photo List, Photo Groups) is rebuilt automatically a few seconds after
every change; corrections typed in *Photo Groups* are applied automatically after you save the workbook. If the
workbook is open in Excel the sync waits and retries. Implemented in `kitchen/sync.py`.

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
| `gui` | the review app (or just `start.bat`) |
| `finalize` | re-render every approved slot to `final_output/` (the app already does this by itself on approve) |
| `excel` / `apply-excel` | manual sync (normally automatic) |
| `export-plain` | uncropped, upright, named copies per brand -> `ready_for_crop/` |
| `db-dump` / `db-restore` | text export / rebuild of the SQLite database (`db/state.sql`) |
| `status` | progress counts |
| `classify` | optional CLIP hints (needs torch, see `app/requirements.txt`) |

## Repository layout

```
app/kitchen/   Python package        app/tests/   pytest suite (python -m pytest -q)
menu/          menu PDFs + catalog.csv (113 items, availability)
labels/        photo_groups.csv (hand-made labels)
db/            state.sql (text export of the database)
doc/           design notes
start.bat      opens the review app
```

Not in git: `assets/`, `work/`, `output/`, `final_output/`, `ready_for_crop/`, `Image_Tracker.xlsx`, `.venv/`.
