# Design notes

## Goal
One real, well-framed photo per menu item for the Swiggy / Zomato partner apps (<= 1 MB each, named by item),
for three brands / 113 items, from ~515 phone photos (423 DNG + 92 JPEG; 88 videos ignored).

## Decisions
| Topic | Decision | Why |
|---|---|---|
| Originals | Never modified; every output is derived | nothing can be lost |
| Dish identity | Hand-labelled photo groups (`labels/photo_groups.csv`), correctable in Excel | CLIP zero-shot reached ~50 % and could not separate biryani variants; the shoot is in bursts, so labelling ~70 groups is fast and reliable |
| Look-alike dishes | Items with the same visual prompt share photos (e.g. a biryani sold by two brands) | user allowed reuse |
| Not-yet-cooked dishes | `availability=not_cooked` in `menu/catalog.csv` | excluded from matching, shown as "Pending cooking" in Excel |
| Crop | Free-form by default (`A` for 1:1 / 4:3 / 3:4) | user preference |
| Tilt | -15..+15 deg slider, 0.1 steps, level grid, auto-level (edge-based) | fine control of horizon |
| Enhancement | Photographic only, per-photo on/off, applied at export from the original RAW | platform policy: never misrepresent the dish |
| State | SQLite (`work/state.db`): `files`, `slots`, `meta`; text export `db/state.sql` | crash-safe, queryable, diff-friendly |
| Excel | Generated from the DB, synced automatically both ways (`sync.py`) | no manual "update" step |
| Size cap | JPEG quality search then dimension shrink until <= 1 MB | platform limit |

## Data model (SQLite)
* `files` - one row per source file: path, kind, status, downscaled path, sharpness / exposure / perceptual hash.
* `slots` - one row per output slot (`item_id:rank`): chosen photo, crop (fractions), rotation, tilt, aspect,
  enhance flag, review status, and export state (`exported`, `exported_enhanced`, `final_path`).
  Changing a decision automatically invalidates the previous export.
* `excluded` - (item, photo) pairs the user deleted; selection never suggests them again for that item.
* `meta` - key/value (which Excel corrections were already applied).

Slot status: `pending | approved | rejected | deleted`. Each pick in `work/selection.json` carries its `rank`
(1..3, or null for alternates), so deleting a rank leaves the other ranks - and their file names - unchanged.

## Sync design (`kitchen/sync.py`)
* DB -> Excel: any saved decision marks the workbook dirty; rebuilt after 2.5 s of quiet; atomic write
  (temp file + replace); if Excel holds the file the swap fails and is retried every 8 s.
* Excel -> DB: the workbook is watched; unapplied corrections in *Photo Groups* are applied
  (`labels/photo_groups.csv`), photos re-selected, workbook refreshed, GUI reloaded.
* Race safeguard: right before swapping the new workbook in, the file is re-checked; corrections saved
  mid-build cause the build to back off and apply them first.

## Open items
* Low-confidence photo groups need confirmation (Excel *Photo Groups*).
* Dishes without any photo need a shoot (see *Menu Items*, "No - photo needed").
* Confirm current Swiggy / Zomato image specs (ratio, minimum size) before final upload.
* Later: optional AI upscaling / background clean-up (phase 2).
