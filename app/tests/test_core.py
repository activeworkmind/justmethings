"""Core behaviour tests. Run from the app folder:  python -m pytest -q"""
from contextlib import closing

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


def test_mat_decides_the_brand_of_a_shared_dish():
    names = lambda ids: {i: next(x for x in catalog.load() if x["item_id"] == i)["brand"] for i in ids}
    both = groups.expand(["BR-001"])                               # no mat: both brands that sell Kaju Paneer Biryani
    assert {"BR-001", "SB-010"} <= set(both)
    assert groups.expand(["BR-001"], "green") == ["BR-001"]        # green leaf mat  -> Biryani Rules
    assert groups.expand(["BR-001"], "royal") == ["SB-010"]        # royal swirl mat -> The Sunday Biryani
    assert groups.expand(["SB-021"], "royal") == ["SB-021"] and groups.expand(["SB-021"], "plain") == ["DM-070"]
    assert groups.expand(["BR-014"], "green") == ["BR-014"]        # Biryani Rules sells gulab jamun too
    assert groups.expand(["BR-011"], "plain") == []                # manchurian is not on the Day Meal menu
    assert set(names(groups.expand(["DM-071"], "plain")).values()) == {"day-meal"}
    with pytest.raises(KeyError):
        groups.expand(["BR-001"], "polka-dot")


def test_retag_rewrites_initials_but_keeps_custom_dish_names():
    from kitchen import retag
    items = {i["item_id"]: i for i in catalog.load()}
    f = lambda *a: retag.new_name(*a, items)
    assert f("BR-003_paneer-nawabi-biryani_2", "BR-003", 2, "SB-012", 1) == "SB-012_paneer-biryani_1"      # default name regenerated
    assert f("BR-004_kaju-paneer-biryani_2", "BR-004", 2, "SB-013", 1) == "SB-013_kaju-paneer-biryani_1"   # custom dish name kept
    assert f("Sunday-biryani-006_hyderabadi-veg-dum-biryani_1", "BR-006", 1, "SB-015", 1) == \
        "SB-015_hyderabadi-veg-dum-biryani_1"                                                              # hand-typed initials fixed
    assert f("BR-001_kaju-paneer-shahi-biryani", "BR-001", 3, "BR-001", 3) == "BR-001_kaju-paneer-shahi-biryani"
    assert f("my-special", "BR-001", 1, "DM-001", 1) == "DM-001_my-special"


def test_slot_roundtrip_and_export_invalidation(tmp_db):
    slot = {"photo_id": "abc", "status": "approved", "name": "BR-001_x_1", "rot": 0, "tilt": 0.0,
            "crop": [0.1, 0.1, 0.9, 0.9], "aspect": "free", "enhance": True}
    slotstore.save("BR-001:1", slot)
    slotstore.mark_exported("BR-001:1", "final/x.jpg", enhanced=True)
    s = slotstore.load_all()["BR-001:1"]
    assert s["exported"] and s["exported_enhanced"] and s["crop"] == [0.1, 0.1, 0.9, 0.9] and s["look"] == []

    look = [{"op": "isolate", "params": {}}, {"op": "shadows", "params": {}}]
    slotstore.save("BR-001:1", {**slot, "look": look})                 # choosing a look is a new decision
    s = slotstore.load_all()["BR-001:1"]
    assert s["look"] == look and not s["exported"]
    slotstore.mark_exported("BR-001:1", "final/x.jpg", enhanced=True)
    slotstore.save("BR-001:1", {**slot, "look": look})                 # same look again -> export stays valid
    assert slotstore.load_all()["BR-001:1"]["exported"]
    slotstore.save("BR-001:1", slot)                                   # look removed -> must be re-rendered
    assert not slotstore.load_all()["BR-001:1"]["exported"]
    slotstore.mark_exported("BR-001:1", "final/x.jpg", enhanced=True)  # back to the original scenario below

    slotstore.save("BR-001:1", {**slot})                  # same decision -> export stays valid
    assert slotstore.load_all()["BR-001:1"]["exported"]

    slotstore.save("BR-001:1", {**slot, "tilt": 1.5})     # decision changed -> must be re-exported
    assert not slotstore.load_all()["BR-001:1"]["exported"]


SLOT = {"photo_id": "abc", "status": "approved", "name": "BR-001_x_2", "rot": 0, "tilt": 0.0,
        "crop": [0.1, 0.1, 0.9, 0.9], "aspect": "free", "enhance": True}


def test_delete_clears_export_and_excludes_the_photo_for_that_dish(tmp_db):
    slotstore.save("BR-001:2", SLOT)
    slotstore.mark_exported("BR-001:2", "final/x.jpg", enhanced=True)
    assert slotstore.mark_deleted("BR-001:2") == "final/x.jpg"          # caller removes this file
    s = slotstore.load_all()["BR-001:2"]
    assert s["status"] == "deleted" and not s["exported"] and s["final_path"] is None
    with closing(db.connect()) as con:
        rows = con.execute("SELECT item_id,photo_id FROM excluded").fetchall()
    assert [tuple(r) for r in rows] == [("BR-001", "abc")]


def test_delete_of_an_unsaved_slot_is_rejected(tmp_db):
    with pytest.raises(KeyError):
        slotstore.mark_deleted("BR-001:1")


def test_approving_saves_the_final_file_and_keeps_it_in_step(tmp_db, tmp_path, monkeypatch):
    from kitchen import finalize
    monkeypatch.setattr(config, "FINAL", tmp_path / "final_output")
    photo = tmp_path / "p.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(photo)
    with closing(db.connect()) as con:
        con.execute("INSERT INTO files(id,path,kind,size,status) VALUES('abc',?,'image',1,'downscaled')", (str(photo),))
        con.commit()
    slot = {"photo_id": "abc", "status": "approved", "name": "MY-name_1", "rot": 0, "tilt": 0.0,
            "crop": [0.1, 0.1, 0.9, 0.9], "aspect": "free", "enhance": True}
    folder = tmp_path / "final_output" / "biryani-rules" / "shahi-veg-paneer-biryanis"

    slotstore.save("BR-001:1", slot)
    res = finalize.export_slot("BR-001:1")
    assert res["status"] == "saved" and (folder / "MY-name_1.jpg").stat().st_size <= config.MAX_BYTES
    assert slotstore.load_all()["BR-001:1"]["exported"]
    assert finalize.export_slot("BR-001:1")["status"] == "up to date"             # nothing changed -> not re-rendered

    slotstore.save("BR-001:1", {**slot, "name": "Renamed_1"})                    # new name -> old file replaced
    assert finalize.export_slot("BR-001:1")["status"] == "saved"
    assert sorted(p.name for p in folder.glob("*.jpg")) == ["Renamed_1.jpg"]

    slotstore.save("BR-001:1", {**slot, "name": "Renamed_1", "status": "rejected"})   # rejected -> file removed
    assert finalize.export_slot("BR-001:1")["status"] == "removed" and not list(folder.glob("*.jpg"))
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
