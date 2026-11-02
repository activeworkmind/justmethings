"""Core behaviour tests. Run from the app folder:  python -m pytest -q"""
import numpy as np
import pytest
from PIL import Image

from kitchen import catalog, config, db, finalize, groups, slotstore
from kitchen.downscale import encode_under


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "state.db")
    return tmp_path


def test_slug():
    assert catalog.slug("Chef's Hukum Veg Biryani") == "chef-s-hukum-veg-biryani"


def test_catalog_has_113_items_and_9_uncooked():
    items = catalog.load()
    assert len(items) == 113
    assert sum(it["availability"] == "not_cooked" for it in items) == 9
    assert len({it["item_id"] for it in items}) == 113


def test_encode_under_respects_the_1mb_cap():
    noise = np.random.default_rng(0).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)   # worst case for JPEG
    data = encode_under(Image.fromarray(noise))
    assert len(data) <= config.MAX_BYTES


def test_geometry_rotation_swaps_dimensions_and_tilt_keeps_size():
    img = Image.new("RGB", (400, 300), "white")
    assert finalize.apply_geometry(img, 1, 0).size == (300, 400)
    assert finalize.apply_geometry(img, 0, 3.0).size == (400, 300)


def test_group_expansion_adds_lookalikes_and_skips_uncooked():
    ids = groups.expand(["BR-001"])                      # Kaju Paneer Biryani is sold by two brands
    assert {"BR-001", "SB-010"} <= set(ids)
    assert "BR-008" not in groups.expand(["BR-008"])     # not cooked yet -> never a candidate


def test_slot_roundtrip_and_export_invalidation(tmp_db):
    slot = {"photo_id": "abc", "status": "approved", "name": "BR-001_x_1", "rot": 0, "tilt": 0.0,
            "crop": [0.1, 0.1, 0.9, 0.9], "aspect": "free", "enhance": True}
    slotstore.save("BR-001:1", slot)
    slotstore.mark_exported("BR-001:1", "final/x.jpg", enhanced=True)
    s = slotstore.load_all()["BR-001:1"]
    assert s["exported"] and s["exported_enhanced"] and s["crop"] == [0.1, 0.1, 0.9, 0.9]

    slotstore.save("BR-001:1", {**slot})                  # same decision -> export stays valid
    assert slotstore.load_all()["BR-001:1"]["exported"]

    slotstore.save("BR-001:1", {**slot, "tilt": 1.5})     # decision changed -> must be re-exported
    assert not slotstore.load_all()["BR-001:1"]["exported"]


def test_db_dump_and_restore(tmp_db, tmp_path):
    slotstore.save("BR-001:1", {"photo_id": "abc", "status": "pending", "name": "n", "rot": 0, "tilt": 0.0,
                                "crop": None, "aspect": "free", "enhance": True})
    dump = tmp_path / "state.sql"
    db.dump(dump)
    config.DB_PATH.unlink(missing_ok=True)
    for ext in ("-wal", "-shm"):
        config.DB_PATH.with_name(config.DB_PATH.name + ext).unlink(missing_ok=True)
    db.restore(dump)
    assert "BR-001:1" in slotstore.load_all()
