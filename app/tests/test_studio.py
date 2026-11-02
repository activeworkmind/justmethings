"""Studio engine tests (synthetic images, no ML model needed). Run from the app folder: python -m pytest -q"""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from studio import backgrounds, batch
from studio import image_engine as E
from studio.brands import BRANDS
from studio.noise import fbm, perlin2d


# ------------------------------------------------------------------ fixtures
def disc_cutout(w=400, h=300):
    """A fake matte: bright 'plate' ellipse on the left, a small 'bowl' on the right."""
    a = np.zeros((h, w), np.float32)
    cv2.ellipse(a, (int(w * 0.40), int(h * 0.55)), (int(w * 0.30), int(h * 0.30)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.circle(a, (int(w * 0.82), int(h * 0.28)), int(w * 0.09), 1.0, -1, cv2.LINE_AA)
    fg = np.zeros((h, w, 3), np.uint8)
    fg[...] = (200, 150, 90)
    return E.Cutout(model="fake", alpha=a, fg=fg)


@pytest.fixture()
def photo():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(60, 200, (300, 400, 3), dtype=np.uint8))


@pytest.fixture()
def fake_isolate(monkeypatch):
    calls = []

    def fake(src, model):
        calls.append(model)
        return disc_cutout(*src.size)
    monkeypatch.setattr(E, "isolate", fake)
    return calls


# ------------------------------------------------------------------ noise / helpers
def test_perlin_is_deterministic_and_bounded():
    a, b = perlin2d((64, 96), (4, 6), seed=3), perlin2d((64, 96), (4, 6), seed=3)
    assert a.shape == (64, 96) and np.array_equal(a, b)
    assert -1.5 < a.min() and a.max() < 1.5
    f = fbm((80, 80), (3, 3), octaves=5, seed=1)
    assert 0 <= f.min() and f.max() <= 1 and f.std() > 0.02


def test_guided_filter_keeps_edges():
    g = np.zeros((64, 64), np.float32)
    g[:, 32:] = 1
    noisy = np.clip(g + np.random.default_rng(0).normal(0, 0.15, g.shape), 0, 1).astype(np.float32)
    out = E.guided_filter(g, noisy, 4, 1e-3)
    assert out[:, :28].mean() < 0.1 and out[:, 36:].mean() > 0.9        # noise removed on both sides (clip biases the dark side up)
    assert out[:, 34].mean() - out[:, 29].mean() > 0.6                 # ... while the edge stays sharp
    assert out.std() < noisy.std()


# ------------------------------------------------------------------ 6. framing / tilt
def test_framing_levels_a_tilted_tray_and_sets_the_width():
    W = H = 1000
    mask = np.zeros((H, W), np.float32)
    box = cv2.boxPoints(((500, 500), (520, 300), 7))                  # tray tilted 7 degrees
    cv2.fillPoly(mask, [box.astype(np.int32)], 1.0)
    s, R, t = E.framing_transform(mask, (W, H), E.FramingParams(fill=0.76, y_anchor=0.56))
    fg = np.zeros((H, W, 3), np.uint8)
    _, a = E.place(fg, mask, (s, R, t), (W, H))
    ys, xs = np.where(a > 0.5)
    bw = xs.max() - xs.min() + 1
    assert abs(bw / W - 0.76) < 0.02                                   # 74-78 % of the canvas width
    assert abs((xs.max() + xs.min()) / 2 / W - 0.5) < 0.01             # centred horizontally
    assert (ys.max() + ys.min()) / 2 / H > 0.5                         # anchored below the midline
    fill_ratio = (a > 0.5).sum() / (bw * (ys.max() - ys.min() + 1))
    assert fill_ratio > 0.97                                           # tray is level: it fills its bounding box


def test_round_plate_is_not_rotated():
    mask = np.zeros((600, 600), np.float32)
    cv2.circle(mask, (300, 300), 200, 1.0, -1)
    _, R, _ = E.framing_transform(mask, (600, 600), E.FramingParams())
    assert np.allclose(R, np.eye(2), atol=1e-3)


# ------------------------------------------------------------------ 3, 8, 9: shadows / steam / bokeh
def blank_dish(W=600, H=600):
    a = np.zeros((H, W), np.float32)
    cv2.ellipse(a, (W // 2, int(H * 0.58)), (int(W * 0.30), int(H * 0.22)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    return a


def test_dual_shadows_darken_below_the_dish_only():
    a = blank_dish()
    bg = np.full((600, 600, 3), 0.8, np.float32)
    out = E.apply_shadows(bg, a, 135.0, E.ShadowParams())
    base = int(0.58 * 600 + 0.22 * 600)                                # bottom of the dish
    assert out[base + 6, 300].mean() < 0.55                            # dark contact shadow hugging the base
    assert out[20, 20].mean() == pytest.approx(0.8, abs=1e-3)          # far corner untouched
    right = out[base - 10, 300 + int(0.30 * 600) + 25].mean()          # ambient shadow falls away from an upper-left light
    left = out[base - 10, 300 - int(0.30 * 600) - 25].mean()
    assert right < left


def test_steam_rises_above_the_main_vessel():
    a = blank_dish()
    s = E.steam_layer(a, E.SteamParams(opacity=0.5, seed=3))
    top = int(0.58 * 600 - 0.22 * 600)
    assert s[:top - 5].max() > 0.15                                    # visible plume above the dish
    assert s[:top - 5].sum() > 5 * s[top + 40:].sum() + 1              # ... rising up, not spread over the table
    assert s[-40:].max() < 0.01


def test_bokeh_blurs_the_back_more_than_the_front():
    rng = np.random.default_rng(2)
    bg = np.clip(rng.random((400, 400, 3)).astype(np.float32), 0, 1)
    a = np.zeros((400, 400), np.float32)
    a[250:340, 120:280] = 1
    out = E.apply_bokeh(bg, a, E.BokehParams(strength=1.0))
    hf = lambda im: float(np.abs(np.diff(im, axis=1)).mean())
    assert hf(out[10:60]) < 0.5 * hf(out[300:395])                    # back (top) much blurrier than the front plane


# ------------------------------------------------------------------ 7, 4, grade, badge
def test_lab_pop_boosts_texture_and_warm_chroma():
    rng = np.random.default_rng(5)
    base = np.clip(0.55 + rng.normal(0, 0.03, (200, 200, 1)), 0, 1).astype(np.float32) * np.array([1.0, 0.78, 0.5], np.float32)
    out = E.lab_pop(base, None, E.LabParams(strength=1.0))
    lab = lambda x: cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    assert lab(out)[..., 0].std() > lab(base)[..., 0].std() * 1.3      # CLAHE on L: crisper texture
    assert (lab(out)[..., 2] - 128).mean() > (lab(base)[..., 2] - 128).mean()   # warmer yellows


def test_sunlight_is_warm_and_directional():
    img = np.full((300, 400, 3), 0.5, np.float32)
    out = E.apply_sunlight(img, None, 135.0, E.SunlightParams(strength=0.8, blinds=0.0))
    ul, lr = out[20:60, 20:60].mean(axis=(0, 1)), out[240:280, 340:380].mean(axis=(0, 1))
    assert ul.mean() > lr.mean()                                       # brighter toward the upper-left light
    assert ul[0] - ul[2] > 0.03                                        # warm: red above blue


def test_brand_grade_applies_the_profile():
    img = np.full((120, 120, 3), 0.5, np.float32)
    out = E.apply_grade(img, BRANDS["biryani_rules"], E.GradeParams())
    assert out[60, 60, 0] > out[60, 60, 2]                             # warmth
    assert out[2, 2].mean() < out[60, 60].mean()                       # vignette (0.25) darkens the corners
    plain = E.apply_grade(img, BRANDS["day_meal"], E.GradeParams())
    assert plain[2, 2].mean() == pytest.approx(plain[60, 60].mean(), abs=1e-3)   # day meal has no vignette


def test_badge_draws_inside_safe_margins_and_diet_icons_differ():
    img = Image.new("RGB", (800, 800), (120, 120, 120))
    out = E.apply_badge(img, BRANDS["sunday_biryani"], E.BadgeParams(diet="nonveg"))
    assert out.size == img.size and np.asarray(out).std() > 1
    a = np.asarray(E.draw_diet_icon(60, "veg"))[30, 30]
    b = np.asarray(E.draw_diet_icon(60, "nonveg"))[30, 30]
    assert not np.array_equal(a, b)


# ------------------------------------------------------------------ backgrounds
def test_procedural_background_is_deterministic_and_brand_specific():
    a = backgrounds.procedural_background(BRANDS["day_meal"], 256, 256, seed=1)
    b = backgrounds.procedural_background(BRANDS["day_meal"], 256, 256, seed=1)
    c = backgrounds.procedural_background(BRANDS["biryani_rules"], 256, 256, seed=1)
    assert np.array_equal(a, b) and a.mean() > c.mean() + 20           # airy light wood vs dark rosewood


def test_get_background_falls_back_without_token_or_network():
    arr, info = backgrounds.get_background("sunday_biryani", 200, 200, "hf", None)
    assert arr.shape == (200, 200, 3) and "procedural" in info


# ------------------------------------------------------------------ history / matte cache / recipe
def test_undo_redo_reset_and_single_matte_extraction(photo, fake_isolate):
    s = E.ImageSession("p", photo, "day_meal", "1:1", preview_edge=200)
    assert not s.can_undo and not s.can_redo
    s.apply("isolate")
    s.apply("background", source="procedural")
    s.apply("shadows")
    s.apply("sunlight")
    assert len(fake_isolate) == 1                                      # the matte was extracted exactly once
    assert [e.label for e in s.undo_stack][-1] == "Natural sunlight touch"
    img_after = s.current
    assert s.undo() == "Natural sunlight touch" and s.scene.sunlight is None and s.can_redo
    assert s.redo() == "Natural sunlight touch" and s.scene.sunlight is not None and s.current is img_after
    s.apply("bokeh")
    assert not s.can_redo                                              # a new action clears the redo stack
    s.reset()
    assert s.scene.cutout is None and s.steps == [] and s.can_undo
    s.apply("framing")                                                 # matte comes from the cache, not a new extraction
    assert len(fake_isolate) == 1 and s.alpha_matte is not None
    s.undo()
    s.undo()                                                           # back before 'reset'
    assert s.scene.bokeh is not None


def test_preset_is_one_undo_step_and_replays_from_a_recipe(photo, fake_isolate):
    s = E.ImageSession("p", photo, "biryani_rules", "1:1", preview_edge=200)
    s.apply_steps(E.master_preset("biryani_rules").steps, "preset")
    assert len(s.undo_stack) == 1
    rec = s.recipe()
    again = E.EnhancementRecipe.from_json(rec.to_json())
    assert again.steps == rec.steps and again.steps[0]["op"] == "isolate"
    scene = again.apply_to(photo)
    assert scene.grade is not None and scene.steam is not None and scene.cutout is not None
    assert not E.master_preset("day_meal").has("steam") and E.master_preset("day_meal").without("bokeh").has("bokeh") is False


def test_render_layers_and_transparent_export(photo, fake_isolate):
    sc = E.Scene(src=photo, brand="sunday_biryani", ratio="1:1")
    sc = E.apply_op(sc, "isolate", {})
    sc = E.apply_op(sc, "framing", {})
    out = E.render(sc, 300)
    assert out.size == (300, 300) and out.mode == "RGB"
    tr = E.render(E.apply_op(sc, "background", {"source": "transparent"}), 300)
    assert tr.mode == "RGBA" and np.asarray(tr)[..., 3].max() == 255 and np.asarray(tr)[..., 3].min() == 0


# ------------------------------------------------------------------ export profile
def test_upload_profile_respects_one_megabyte_and_strips_overlays(tmp_path):
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (2048, 2048, 3), dtype=np.uint8))
    info = E.save_image(noise, tmp_path / "x.jpg", E.PROFILES["swiggy_zomato"])
    assert info["bytes"] <= 1_000_000 and (tmp_path / "x.jpg").stat().st_size == info["bytes"]
    rec = E.master_preset("day_meal", include_badge=True)
    assert rec.has("badge") and not rec.without("badge").has("badge")
    assert E.PROFILES["swiggy_zomato"].allow_overlays is False


def test_batch_runs_a_recipe_on_a_folder(tmp_path, photo, fake_isolate, monkeypatch):
    monkeypatch.setattr(batch, "record_upload", lambda *a, **k: None)
    src = tmp_path / "in"
    src.mkdir()
    photo.save(src / "DM-001_test-dish_1.jpg")
    photo.save(src / "DM-002_other_1.jpg")
    (src / "broken.jpg").write_bytes(b"not an image")
    rec = E.master_preset("day_meal", include_badge=True)
    rep = batch.run_recipe_on_folder(rec, src, tmp_path / "out", "swiggy_zomato")
    assert rep["ok"] == 2 and rep["failed"] == 1                       # one bad photo never stops the batch
    outs = sorted((tmp_path / "out" / "day-meal").glob("*.jpg"))
    assert [p.name for p in outs] == ["DM-001_test-dish_1.jpg", "DM-002_other_1.jpg"]
    assert all(p.stat().st_size <= 1_000_000 for p in outs)
    assert (tmp_path / "out" / "day-meal" / "_batch_report.csv").exists()
    again = batch.run_recipe_on_folder(rec, src, tmp_path / "out", "swiggy_zomato", skip_existing=True)
    assert again["skipped"] == 2
