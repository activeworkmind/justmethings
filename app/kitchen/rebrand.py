"""Brand tag: say "this photo belongs to <brand>" and the photo moves to the same dish of that brand.

The brand of a photo is told by the cloth it was shot on:
    green-leaf linen mat -> Biryani Rules      royal gold-swirl mat -> The Sunday Biryani      plain cloth -> The Day Meal
When a file carries the wrong brand initials (BR- / SB- / DM-), click the right tag in the review app: the slot moves
to the matching dish, the file name is renamed (initials + rank), the saved file in final_output is replaced and the
Excel follows. Your crop, look and approval travel with the photo."""
from contextlib import closing
import time

from . import catalog, db, retag, select, slotstore

BRAND_NAMES = {"biryani-rules": "Biryani Rules", "sunday-biryani": "The Sunday Biryani", "day-meal": "The Day Meal"}
MAT_HINT = {"biryani-rules": "green leaf mat", "sunday-biryani": "royal swirl mat", "day-meal": "plain cloth"}


def equivalents(item_id: str, brand_slug: str) -> list[dict]:
    """Cooked items of `brand_slug` that are the same dish as `item_id` (same visual prompt)."""
    src = next(i for i in catalog.load() if i["item_id"] == item_id)
    return [i for i in catalog.active() if i["brand"] == brand_slug and i["clip_prompt"] == src["clip_prompt"]]


def menu_of(brand_slug: str) -> list[dict]:
    return [i for i in catalog.active() if i["brand"] == brand_slug]


def assign(photo_id: str, item_id: str) -> dict:
    """Record 'photo = item', re-select (only the photos still in your list) and move the affected slots."""
    with closing(db.connect()) as con:
        con.execute("INSERT INTO photo_items(photo_id,item_id,set_at) VALUES(?,?,?) "
                    "ON CONFLICT(photo_id) DO UPDATE SET item_id=excluded.item_id,set_at=excluded.set_at",
                    (photo_id, item_id, time.time()))
        con.commit()
        scope = retag.remaining_photos(con) | {photo_id}
        old = slotstore.load_all()
        select.run(con, scope=scope)
    return retag.migrate_slots(old)
