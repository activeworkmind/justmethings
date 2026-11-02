"""Studio engine tests (synthetic images, no ML model needed). Run from the app folder: python -m pytest -q"""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from studio import backgrounds
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


def test_render_layers_and_transparent_export(photo, fake_isolate):
    sc = E.Scene(src=photo, brand="sunday_biryani", ratio="1:1")
    sc = E.apply_op(sc, "isolate", {})
    sc = E.apply_op(sc, "framing", {})
    out = E.render(sc, 300)
    assert out.size == (300, 300) and out.mode == "RGB"
    tr = E.render(E.apply_op(sc, "background", {"source": "transparent"}), 300)
    assert tr.mode == "RGBA" and np.asarray(tr)[..., 3].max() == 255 and np.asarray(tr)[..., 3].min() == 0


# ------------------------------------------------------------------ look (what the review app uses)
def test_apply_look_renders_and_reuses_the_cutout(photo, fake_isolate):
    steps = E.preset_steps("day_meal")
    assert steps[0]["op"] == "isolate" and not any(s["op"] == "badge" for s in steps)
    assert E.preset_steps("day_meal", include_badge=True)[-1]["op"] == "badge"
    cache = {}
    a = E.apply_look(photo, "day_meal", steps, 300, cache, "k")
    b = E.apply_look(photo, "day_meal", [s for s in steps if s["op"] != "steam"], 300, cache, "k")
    assert a.size == b.size == (300, 224)                              # keeps the user's crop shape (even pixel sizes)
    assert len(fake_isolate) == 1                                      # second preview reused the cached cut-out
    plain = E.apply_look(photo, "day_meal", [{"op": "lab_pop", "params": {}}], 300, cache, "k")
    assert plain.size == (300, 224) and len(fake_isolate) == 1         # mask-free steps need no cut-out at all


def test_finalize_applies_the_look_after_the_crop(photo, fake_isolate):
    from kitchen import finalize
    slot = {"rot": 0, "tilt": 0.0, "crop": [0.1, 0.1, 0.9, 0.9], "enhance": True,
            "look": E.preset_steps("sunday_biryani"), "_brand": "sunday_biryani"}
    out = finalize.render_image(photo, slot, 400)
    assert out.size == (320, 240) and len(fake_isolate) == 1
    assert finalize.brand_key("day-meal") == "day_meal"


def test_dialog_turns_checkboxes_into_steps(photo):
    import tkinter as tk
    from kitchen.lookdialog import LookDialog
    root = tk.Tk()
    root.withdraw()
    tk.Toplevel.grab_set = lambda self: None                           # the window is not visible in a test
    try:
        d = LookDialog(root, "Day Meal", "day_meal", [], lambda steps: photo)
        assert d.steps() == []                                         # nothing ticked = plain photo
        d.preset()
        ops = [s["op"] for s in d.steps()]
        assert ops[0] == "isolate" and "background" in ops and "steam" not in ops      # Day Meal preset has no steam
        d.clear()
        d.vars["lab_pop"].set(True)
        assert [s["op"] for s in d.steps()] == ["lab_pop"]            # a mask-free step does not force isolation
        d.destroy()
    finally:
        root.destroy()
