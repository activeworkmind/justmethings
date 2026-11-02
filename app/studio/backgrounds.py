"""Brand table backgrounds.

Sources (tried in this order for 'auto'):
  1. HuggingFace free serverless inference (FLUX.1-schnell / SDXL) - needs a free HF token, text prompt only
  2. a local template folder  studio_assets/backgrounds/<brand-slug>/*.jpg|png
  3. a procedural studio backdrop (OpenCV + numpy) - always available, works fully offline
Every generated/loaded background is cover-fitted to the canvas; HF results are cached on disk."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .brands import BRANDS, BrandProfile
from .noise import fbm

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "studio_cache" / "backgrounds"
ASSETS_DIR = ROOT / "studio_assets" / "backgrounds"
HF_MODELS = ("black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_MEM: dict = {}


def _rgb(hex_):
    return np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    s = max(w / img.width, h / img.height)
    r = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


# ------------------------------------------------------------------ procedural
STYLES = {
    "royal": dict(horizon=0.30, wood_light="#6B3320", wood_dark="#2A110A", wall_top="#070A14", wall_bottom="#1A1008",
                  plank=0.16, grain=1.0, dim=0.80),
    "festive": dict(horizon=0.34, wood_light="#B07A3F", wood_dark="#6A3F1B", wall_top="#3B2414", wall_bottom="#6B4526",
                    plank=0.14, grain=1.2, dim=1.00),
    "homestyle": dict(horizon=0.36, wood_light="#E3C79C", wood_dark="#B58E5E", wall_top="#DCCDB4", wall_bottom="#CDB894",
                      plank=0.20, grain=0.8, dim=0.90),
}


def _wood(h, w, light, dark, seed, plank_h, grain):
    """Wood-grain texture (float32 HxWx3, 0..1): horizontal grain, ring lines, plank seams, per-plank tone."""
    res_y = max(int(h / 14), 4)
    streak = fbm((h, w), base_res=(res_y, 3), octaves=4, persistence=0.55, seed=seed)
    warp = fbm((h, w), base_res=(5, 3), octaves=3, seed=seed + 7)
    y = np.arange(h, dtype=np.float32)[:, None]
    rings = 0.5 + 0.5 * np.sin((y / (h / (16 * grain)) + warp * 5.0) * 2 * np.pi)
    t = np.clip(0.55 * streak + 0.30 * rings * streak + 0.12, 0, 1)[..., None]
    img = _rgb(dark) * (1 - t) + _rgb(light) * t
    plank = (np.arange(h) // max(plank_h, 8))
    rng = np.random.default_rng(seed + 3)
    tone = (0.88 + 0.2 * rng.random(plank.max() + 1))[plank][:, None, None].astype(np.float32)
    d = (np.arange(h) % max(plank_h, 8)).astype(np.float32)
    seam = (1 - 0.65 * np.exp(-(d / 1.8) ** 2))[:, None, None]
    return np.clip(img * tone * seam, 0, 1).astype(np.float32)


def _sprite_scale(y, y_h, h):
    return 0.55 + 0.9 * np.clip((y - y_h) / max(h - y_h, 1), 0, 1)


def _avoid_dish(x, y, w, h):
    return ((x - 0.5 * w) / (0.30 * w)) ** 2 + ((y - 0.64 * h) / (0.24 * h)) ** 2 < 1.0


def _draw_royal_props(layer, mask, w, h, y_h, rng):
    """Cardamom pods, star anise and saffron strands scattered on the dark wood."""
    for _ in range(18):
        x, y = rng.uniform(0.04 * w, 0.96 * w), rng.uniform(y_h + 0.08 * h, 0.97 * h)
        if _avoid_dish(x, y, w, h):
            continue
        s = _sprite_scale(y, y_h, h) * 0.022 * w
        ang = rng.uniform(0, 180)
        for col, k in (((86, 124, 52), 1.0), ((150, 186, 96), 0.45)):
            cv2.ellipse(layer, (int(x), int(y)), (int(s * k * 1.5), int(s * k * 0.8)), ang, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (int(x), int(y)), (int(s * 1.5), int(s * 0.8)), ang, 0, 360, 255, -1, cv2.LINE_AA)
    for _ in range(7):
        x, y = rng.uniform(0.05 * w, 0.95 * w), rng.uniform(y_h + 0.1 * h, 0.96 * h)
        if _avoid_dish(x, y, w, h):
            continue
        s = _sprite_scale(y, y_h, h) * 0.03 * w
        a0 = rng.uniform(0, np.pi / 4)
        pts = np.array([[x + (s if i % 2 == 0 else s * 0.42) * np.cos(a0 + i * np.pi / 8),
                         y + (s if i % 2 == 0 else s * 0.42) * np.sin(a0 + i * np.pi / 8) * 0.8] for i in range(16)], np.int32)
        cv2.fillPoly(layer, [pts], (98, 54, 30), cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
        cv2.circle(layer, (int(x), int(y)), max(int(s * 0.16), 1), (52, 26, 14), -1, cv2.LINE_AA)
    for _ in range(16):                                           # saffron: short, gently curved, tapering threads
        x, y = rng.uniform(0.05 * w, 0.95 * w), rng.uniform(y_h + 0.1 * h, 0.97 * h)
        if _avoid_dish(x, y, w, h):
            continue
        s = _sprite_scale(y, y_h, h) * 0.022 * w
        ang, bend = rng.uniform(0, np.pi), rng.uniform(-0.35, 0.35)
        for k in range(rng.integers(3, 6)):                       # a small pinch of threads
            ox, oy = rng.normal(0, s * 0.18, 2)
            t = np.linspace(-1, 1, 12)
            px = x + ox + (t * np.cos(ang) - bend * (t ** 2 - 0.33) * np.sin(ang)) * s
            py = y + oy + (t * np.sin(ang) + bend * (t ** 2 - 0.33) * np.cos(ang)) * s * 0.8
            pts = np.stack([px, py], 1).astype(np.int32)
            cv2.polylines(layer, [pts], False, (222, 78, 24), max(int(s * 0.07), 1), cv2.LINE_AA)
            cv2.polylines(mask, [pts], False, 255, max(int(s * 0.07), 1), cv2.LINE_AA)


def _draw_festive_props(layer, mask, w, h, y_h, rng):
    """Banana leaf and brass katori bowls, to be blurred into the background."""
    leaf = np.array([[0.00 * w, y_h + 0.02 * h], [0.30 * w, y_h - 0.04 * h], [0.46 * w, y_h + 0.10 * h],
                     [0.30 * w, y_h + 0.20 * h], [0.04 * w, y_h + 0.16 * h]], np.int32)
    cv2.fillPoly(layer, [leaf], (40, 82, 36), cv2.LINE_AA)
    cv2.fillPoly(mask, [leaf], 255, cv2.LINE_AA)
    cv2.line(layer, (int(0.02 * w), int(y_h + 0.09 * h)), (int(0.44 * w), int(y_h + 0.07 * h)), (86, 128, 62), 3, cv2.LINE_AA)
    for cx, cy, r in ((0.66, 0.0, 0.075), (0.80, 0.045, 0.060), (0.92, -0.01, 0.050)):
        c = (int(cx * w), int(y_h + cy * h)), (int(r * w), int(r * w * 0.42))
        cv2.ellipse(layer, c[0], c[1], 0, 0, 360, (206, 150, 56), -1, cv2.LINE_AA)
        cv2.ellipse(layer, c[0], (int(c[1][0] * 0.82), int(c[1][1] * 0.7)), 0, 0, 360, (120, 74, 28), -1, cv2.LINE_AA)
        cv2.ellipse(layer, (c[0][0] - int(c[1][0] * 0.35), c[0][1] - int(c[1][1] * 0.45)),
                    (int(c[1][0] * 0.25), int(c[1][1] * 0.12)), 0, 0, 360, (255, 232, 170), -1, cv2.LINE_AA)
        cv2.ellipse(mask, c[0], c[1], 0, 0, 360, 255, -1, cv2.LINE_AA)


def _draw_home_props(layer, mask, w, h, y_h, rng):
    """Terracotta clay bowl and fresh coriander leaves, to be blurred into the background."""
    c = (int(0.18 * w), int(y_h + 0.03 * h))
    cv2.ellipse(layer, c, (int(0.07 * w), int(0.03 * w)), 0, 0, 360, (186, 98, 60), -1, cv2.LINE_AA)
    cv2.ellipse(layer, c, (int(0.058 * w), int(0.02 * w)), 0, 0, 360, (96, 46, 28), -1, cv2.LINE_AA)
    cv2.ellipse(mask, c, (int(0.07 * w), int(0.03 * w)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    for _ in range(22):
        x, y = rng.uniform(0.55 * w, 0.97 * w), rng.uniform(y_h - 0.04 * h, y_h + 0.16 * h)
        r = int(rng.uniform(0.012, 0.024) * w)
        for k in range(3):
            a = k * 2.1 + rng.uniform(0, 1)
            cv2.circle(layer, (int(x + r * 0.7 * np.cos(a)), int(y + r * 0.5 * np.sin(a))), max(int(r * 0.7), 2), (70, 128, 54), -1, cv2.LINE_AA)
            cv2.circle(mask, (int(x + r * 0.7 * np.cos(a)), int(y + r * 0.5 * np.sin(a))), max(int(r * 0.7), 2), 255, -1, cv2.LINE_AA)


def _jute_runner(img, w, h, y_h, seed):
    """Woven jute table runner across the middle of the table."""
    x0, x1 = int(0.10 * w), int(0.90 * w)
    y0, y1 = int(y_h + 0.17 * h), int(0.97 * h)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    f = 2 * np.pi / (0.009 * w)
    weave = 0.5 + 0.25 * np.sin(xx * f) * np.sin(yy * f * 0.9) + 0.25 * np.sin(xx * f * 0.5 + yy * f * 0.3)
    rough = fbm((y1 - y0, x1 - x0), base_res=(40, 40), octaves=3, seed=seed + 11)
    tone = np.clip(0.55 * weave + 0.45 * rough, 0, 1)[..., None]
    jute = _rgb("#9C7C4C") * (1 - tone) + _rgb("#D8BC88") * tone
    edge = np.minimum.reduce([xx - x0, x1 - xx, yy - y0, y1 - yy])
    jute *= (0.78 + 0.22 * np.clip(edge / (0.012 * w), 0, 1))[..., None]
    img[y0:y1, x0:x1] = jute


def _bokeh_lights(img, w, h, y_h, rng, color, n, strength):
    """Soft out-of-focus light discs in the upper background."""
    layer = np.zeros((h, w, 3), np.float32)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0.02 * h, y_h * 1.05)
        r = int(rng.uniform(0.012, 0.035) * w)
        cv2.circle(layer, (int(x), int(y)), r, tuple(float(c) * rng.uniform(0.5, 1.0) for c in color), -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.004 * w)
    img += layer * strength


def procedural_background(brand: BrandProfile, w: int, h: int, seed: int = 0) -> np.ndarray:
    """Studio table backdrop (uint8 RGB, h x w): perspective wood table, brand props, lighting, grain, vignette."""
    rng = np.random.default_rng(seed)
    st = STYLES[brand.style]
    y_h = int(h * st["horizon"])
    tex = _wood(2 * h, w, st["wood_light"], st["wood_dark"], seed, int(h * st["plank"]), st["grain"])
    rows = np.arange(h)
    t = np.clip((rows - y_h) / max(h - y_h, 1), 0, 1) ** 0.65
    img = tex[(t * (tex.shape[0] - 1)).astype(int)].copy()
    wall = np.linspace(0, 1, max(y_h, 1), dtype=np.float32)[:, None, None]
    img[:y_h] = _rgb(st["wall_top"]) * (1 - wall) + _rgb(st["wall_bottom"]) * wall
    if brand.style == "homestyle":
        _jute_runner(img, w, h, y_h, seed)

    layer = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    {"royal": _draw_royal_props, "festive": _draw_festive_props, "homestyle": _draw_home_props}[brand.style](
        layer, mask, w, h, y_h, rng)
    blur = {"royal": 0.0012, "festive": 0.016, "homestyle": 0.011}[brand.style] * w      # props behind the dish are out of focus
    layer = cv2.GaussianBlur(layer, (0, 0), max(blur, 0.6)).astype(np.float32) / 255
    a = (cv2.GaussianBlur(mask, (0, 0), max(blur, 0.6)).astype(np.float32) / 255)[..., None]
    img = img * (1 - a) + layer * a

    if brand.style == "royal":                                    # dark, amber rim light from the upper right
        _bokeh_lights(img, w, h, y_h, rng, (0.95, 0.62, 0.22), 14, 0.55)
    elif brand.style == "festive":
        _bokeh_lights(img, w, h, y_h, rng, (1.0, 0.72, 0.30), 12, 0.45)
    else:
        _bokeh_lights(img, w, h, y_h, rng, (1.0, 0.95, 0.80), 8, 0.30)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lx, ly = {"royal": (0.78, 0.25), "festive": (0.30, 0.20), "homestyle": (0.0, 0.35)}[brand.style]
    d = np.sqrt(((xx / w - lx) ** 2) * 1.0 + ((yy / h - ly) ** 2) * 1.4)
    glow = np.exp(-(d / 0.55) ** 2)[..., None]
    warm = {"royal": (1.0, 0.62, 0.25), "festive": (1.0, 0.78, 0.45), "homestyle": (1.0, 0.95, 0.82)}[brand.style]
    k_mul, k_add = {"royal": (0.45, 0.16), "festive": (0.45, 0.16), "homestyle": (0.22, 0.05)}[brand.style]
    img = img * (1 - k_mul * 0.5 + k_mul * glow) + glow * np.array(warm, np.float32) * k_add
    img *= st["dim"]
    img = cv2.GaussianBlur(img, (0, 0), 0.002 * w)                  # DOF proper is applied later by the bokeh step
    # soften the horizon line
    band = int(0.012 * h)
    if y_h - band > 0:
        seg = img[y_h - band:y_h + band]
        img[y_h - band:y_h + band] = cv2.GaussianBlur(seg, (0, 0), band / 2.2)
    img += rng.normal(0, 0.010, (h, w, 1)).astype(np.float32)                                   # film grain
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ local + HuggingFace
def local_background(brand: BrandProfile, w: int, h: int, seed: int = 0, assets_dir: Path = None):
    folder = (assets_dir or ASSETS_DIR) / brand.slug
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else []
    if not files:
        return None
    img = Image.open(files[seed % len(files)]).convert("RGB")
    return np.asarray(cover_resize(img, w, h))


def hf_background(brand: BrandProfile, w: int, h: int, token: str, model: str, seed: int = 0) -> np.ndarray:
    """Generate a themed table background with the free HuggingFace serverless inference API (text only, no photos)."""
    gw, gh = (1024, 1024) if abs(w / h - 1) < 0.12 else ((1024, 768) if w > h else (768, 1024))
    key = hashlib.sha1(f"{model}|{brand.bg_prompt}|{gw}x{gh}|{seed}".encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{brand.slug}_{key}.png"
    if cache.exists():
        img = Image.open(cache).convert("RGB")
    else:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token, timeout=150)
        kw = dict(model=model, width=gw, height=gh, seed=seed)
        if "schnell" in model:
            kw["num_inference_steps"] = 4
        else:
            kw["negative_prompt"] = brand.negative_prompt              # FLUX.1-schnell has no negative prompt
        try:
            img = client.text_to_image(brand.bg_prompt, **kw)
        except TypeError:
            kw.pop("seed", None)
            img = client.text_to_image(brand.bg_prompt, **kw)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(cache)
    out = np.asarray(cover_resize(img.convert("RGB"), w, h)).astype(np.float32)
    if max(w, h) > 1.4 * max(gw, gh):                                       # hide upscaling softness with fine grain
        out += np.random.default_rng(seed).normal(0, 2.5, (h, w, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def get_background(brand_key: str, w: int, h: int, source: str = "auto", token: str = None,
                   hf_model: str = HF_MODELS[0], seed: int = 0, assets_dir: Path = None):
    """Return (uint8 RGB array h x w, info string). Never raises: falls back to the procedural backdrop."""
    ck = (brand_key, w, h, source, bool(token), hf_model, seed, str(assets_dir))
    if ck in _MEM:
        return _MEM[ck]
    brand = BRANDS[brand_key]
    notes = []
    order = {"auto": ["hf", "local", "procedural"], "hf": ["hf", "procedural"], "local": ["local", "procedural"],
             "procedural": ["procedural"]}.get(source, ["procedural"])
    result = None
    for src in order:
        try:
            if src == "hf":
                if not token:
                    notes.append("no HuggingFace token")
                    continue
                result = (hf_background(brand, w, h, token, hf_model, seed), f"AI background ({hf_model.split('/')[-1]})")
            elif src == "local":
                arr = local_background(brand, w, h, seed, assets_dir)
                if arr is None:
                    notes.append(f"no templates in studio_assets/backgrounds/{brand.slug}")
                    continue
                result = (arr, "local template")
            else:
                result = (procedural_background(brand, w, h, seed), "procedural studio backdrop")
            break
        except Exception as e:                                           # network / quota / model errors -> fall back
            notes.append(f"{src} failed: {type(e).__name__}: {str(e)[:120]}")
    arr, info = result
    if notes:
        info += "  [" + "; ".join(notes) + "]"
    if len(_MEM) > 12:
        _MEM.pop(next(iter(_MEM)))
    _MEM[ck] = (arr, info)
    return arr, info
