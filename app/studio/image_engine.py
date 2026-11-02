"""Image engine: non-destructive scene model + the ten "Enhance with AI" steps + undo/redo history + recipes.

Design
------
* A `Scene` is an immutable description of an image: the source photo, the cached foreground matte (`Cutout`) and
  one small parameter block per enhancement. `render(scene, long_edge)` composes it from scratch in a fixed layer
  order (background -> shadows -> dish -> relight -> rim -> steam -> grade -> badge), so enhancements can be
  stacked in any order and re-rendered at any resolution (preview or export) with identical results.
* The matte is computed once (rembg) and shared by every later scene, so lighting / framing / background tweaks
  never re-run background removal.
* `ImageSession` keeps independent undo / redo stacks of (PIL image, label, scene) per photo.
* An `EnhancementRecipe` is the recorded list of steps; it replays on any photo (batch mode).
All maths is OpenCV / numpy / PIL - no placeholders."""
from __future__ import annotations

import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

from . import backgrounds

ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 1 << 26)   # large optimised JPEGs overflow PIL's default encoder buffer
from .brands import BRANDS, DIET_LABELS
from .noise import fbm

MAX_SRC_EDGE = 3072
MATTE_EDGE = 1600
RATIOS = {"original": None, "1:1": 1.0, "4:5": 4 / 5, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9}
ISOLATE_MODELS = ("birefnet-general", "isnet-general-use", "u2net")


# ============================================================================ configuration (never in recipes)
@dataclass
class EngineConfig:
    hf_token: str = ""                       # free HuggingFace token (text prompts only are sent)
    hf_model: str = backgrounds.HF_MODELS[0]
    isolate_model: str = "isnet-general-use"
    allow_remote_relight: bool = False       # IC-Light uploads the dish photo to a public HF Space -> opt-in
    bg_gen_edge: int = 1280                  # procedural / AI backgrounds are generated at this size, then fitted
    warnings: list = field(default_factory=list)


CONFIG = EngineConfig()


def _warn(msg: str):
    if msg not in CONFIG.warnings:
        CONFIG.warnings.append(msg)


# ============================================================================ parameter blocks
@dataclass(frozen=True)
class BackgroundParams:
    source: str = "auto"                     # auto | hf | local | procedural | neutral | transparent
    seed: int = 0


@dataclass(frozen=True)
class FramingParams:
    fill: float = 0.76                       # dish width as a fraction of the canvas width (74-78 %)
    y_anchor: float = 0.56                   # dish centre, fraction of canvas height (just below the midline)
    auto_tilt: bool = True
    tilt: float = 0.0                        # extra manual rotation, degrees clockwise


@dataclass(frozen=True)
class ShadowParams:
    contact_opacity: float = 0.75
    ambient_opacity: float = 0.35
    softness: float = 1.0


@dataclass(frozen=True)
class SunlightParams:
    strength: float = 0.45
    angle: Optional[float] = None            # None = brand default; degrees the light comes from (135 = upper-left)
    blinds: float = 0.35                     # window-blind light falloff
    use_iclight: bool = False
    iclight_prompt: str = "warm natural sunlight from a window, golden hour, soft shadows, food photography"


@dataclass(frozen=True)
class RimParams:
    intensity: float = 0.7


@dataclass(frozen=True)
class LabParams:
    strength: float = 1.0


@dataclass(frozen=True)
class SteamParams:
    opacity: float = 0.4
    seed: int = 3


@dataclass(frozen=True)
class BokehParams:
    strength: float = 0.6


@dataclass(frozen=True)
class GradeParams:
    strength: float = 1.0                    # scales the brand's warmth / contrast / saturation / vignette
    warmth: Optional[float] = None           # None = brand profile value
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    vignette: Optional[float] = None


@dataclass(frozen=True)
class BadgeParams:
    corner: str = "bottom-left"              # brand badge corner
    diet_corner: str = "top-right"           # dietary icon corner
    diet: Optional[str] = None               # veg | egg | nonveg ; None = brand default
    show_brand: bool = True
    show_diet: bool = True
    border: bool = True


# ============================================================================ matte (cached foreground cut-out)
@dataclass(frozen=True, eq=False)
class Cutout:
    model: str
    alpha: np.ndarray                        # float32, matte resolution
    fg: np.ndarray                           # uint8 RGB, colour-decontaminated, matte resolution
    cache: dict = field(default_factory=dict)

    def at_src_size(self, src: Image.Image):
        """(fg uint8 RGB, alpha float32) at the source resolution: alpha is edge-refined with a guided filter."""
        key = src.size
        if key in self.cache:
            return self.cache[key]
        w, h = src.size
        if self.alpha.shape[::-1] == (w, h):
            out = (self.fg, self.alpha)
        else:
            a = cv2.resize(self.alpha, (w, h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            a = np.clip(guided_filter(gray, np.clip(a, 0, 1), max(3, int(0.003 * w)), 1e-3), 0, 1)
            fg_up = cv2.resize(self.fg, (w, h), interpolation=cv2.INTER_CUBIC)
            interior = (a >= 0.98)[..., None]
            out = (np.where(interior, np.asarray(src), fg_up), a)
        self.cache[key] = out
        return out


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (single-channel guide). Snaps a soft matte to real image edges."""
    mi, mp = _box(guide, r), _box(p, r)
    cov = _box(guide * p, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


_SESSIONS: dict = {}


def _session(model: str):
    if model not in _SESSIONS:
        import rembg
        _SESSIONS[model] = rembg.new_session(model)
    return _SESSIONS[model]


def clean_alpha(a: np.ndarray) -> np.ndarray:
    """Keep real objects (plate + side bowls), drop specks, fill pin-holes, feather the edge slightly."""
    hard = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, ar in enumerate(areas) if ar >= max(0.004 * hard.size, 0.04 * areas.max())]
        keep_mask = np.isin(lab, keep)
        keep_mask = cv2.dilate(keep_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        a = np.where(keep_mask, a, 0)
        hard = ((a > 0.5)).astype(np.uint8)
    inv = (1 - hard).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.0015 * hard.size and not (lab[0, :] == i).any() and not (lab[-1, :] == i).any() \
                and not (lab[:, 0] == i).any() and not (lab[:, -1] == i).any():
            a[lab == i] = 1.0                                   # enclosed pin-hole inside the dish
    return np.clip(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.8), 0, 1)


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove the old background's colour from edge pixels (normalised convolution of solid-interior colours)."""
    solid = (cv2.erode((alpha > 0.98).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.float32)
    f = rgb.astype(np.float32)
    sigma = max(2.0, 0.004 * rgb.shape[1])
    num = cv2.GaussianBlur(f * solid[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(solid, (0, 0), sigma)[..., None]
    ext = num / np.maximum(den, 1e-4)
    out = np.where((alpha < 0.98)[..., None] & (den > 1e-3), ext, f)
    return np.clip(out, 0, 255).astype(np.uint8)


def isolate(src: Image.Image, model: str) -> Cutout:
    """Clean Dish Isolation: rembg matte -> cleanup -> colour decontamination (cached by the caller)."""
    w, h = src.size
    k = min(1.0, MATTE_EDGE / max(w, h))
    small = src.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS) if k < 1 else src
    import rembg
    mask = rembg.remove(small, session=_session(model), only_mask=True, post_process_mask=True)
    alpha = clean_alpha(np.asarray(mask, np.float32) / 255)
    return Cutout(model=model, alpha=alpha, fg=decontaminate(np.asarray(small), alpha))


# ============================================================================ scene
@dataclass(frozen=True)
class Scene:
    src: Image.Image
    brand: str
    ratio: str = "original"
    cutout: Optional[Cutout] = None
    background: Optional[BackgroundParams] = None
    framing: Optional[FramingParams] = None
    shadows: Optional[ShadowParams] = None
    sunlight: Optional[SunlightParams] = None
    rim: Optional[RimParams] = None
    lab: Optional[LabParams] = None
    steam: Optional[SteamParams] = None
    bokeh: Optional[BokehParams] = None
    grade: Optional[GradeParams] = None
    badge: Optional[BadgeParams] = None


def prepare_source(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    k = MAX_SRC_EDGE / max(img.size)
    return img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS) if k < 1 else img


def canvas_size(scene: Scene, long_edge: int):
    sw, sh = scene.src.size
    r = RATIOS.get(scene.ratio) or sw / sh
    w, h = (long_edge, long_edge / r) if r >= 1 else (long_edge * r, long_edge)
    return int(round(w / 2) * 2), int(round(h / 2) * 2)


# ============================================================================ small helpers
def _f32(img) -> np.ndarray:
    return np.asarray(img, np.float32) / 255.0


def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def toward_light(angle_deg: float):
    """Unit vector (image coords, y down) pointing from the dish toward the light source."""
    a = math.radians(angle_deg)
    return math.cos(a), -math.sin(a)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _mask_info(a: np.ndarray):
    """Bounding box, centre and bottom contour of the placed dish."""
    m = a > 0.5
    if not m.any():
        h, w = a.shape
        return dict(x0=0, y0=0, x1=w - 1, y1=h - 1, cx=w / 2, cy=h / 2, valid=False)
    ys, xs = np.where(m)
    return dict(x0=int(xs.min()), x1=int(xs.max()), y0=int(ys.min()), y1=int(ys.max()),
                cx=float(xs.mean()), cy=float(ys.mean()), valid=True)


# ============================================================================ 6. angle, tilt and framing
def framing_transform(alpha_src: np.ndarray, canvas, p: FramingParams):
    """Return (s, R, t): canvas = s * R @ src + t. Dish occupies p.fill of the canvas width, centred horizontally,
    anchored at p.y_anchor; minor horizontal tilt of rectangular vessels is corrected with minAreaRect."""
    W, H = canvas
    sh, sw = alpha_src.shape
    k = min(1.0, 640 / max(sw, sh))
    small = (cv2.resize(alpha_src, (max(1, int(sw * k)), max(1, int(sh * k))), interpolation=cv2.INTER_AREA) > 0.5)
    cnts, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.concatenate([c.reshape(-1, 2) for c in cnts]).astype(np.float32) / k
    hull = cv2.convexHull(pts).reshape(-1, 2)
    (_, _), (rw, rh), ang = cv2.minAreaRect(hull)
    c0 = hull.mean(axis=0)

    def rot(theta):
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        return np.array([[c, -s], [s, c]], np.float32)

    def bbox(theta):
        q = (hull - c0) @ rot(theta).T + c0
        return q.min(0), q.max(0)

    theta = 0.0
    if p.auto_tilt and min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) >= 1.12:      # round plates have no tilt
        dev = ang % 90
        dev = dev - 90 if dev > 45 else dev
        if abs(dev) <= 10:
            theta = min((-dev, dev), key=lambda t: np.prod(bbox(t)[1] - bbox(t)[0]))   # whichever levels the box
    theta += p.tilt
    R = rot(theta)
    lo, hi = bbox(theta)
    bw, bh = hi - lo
    s = p.fill * W / max(bw, 1)
    if bh * s > 0.82 * H:
        s = 0.82 * H / max(bh, 1)
    b = (lo + hi) / 2
    target = np.array([0.5 * W, p.y_anchor * H], np.float32)
    t = target + s * (c0 - R @ c0 - b)
    return float(s), R, t


def cover_transform(src_size, canvas):
    sw, sh = src_size
    W, H = canvas
    s = max(W / sw, H / sh)
    return s, np.eye(2, dtype=np.float32), np.array([W / 2 - s * sw / 2, H / 2 - s * sh / 2], np.float32)


def place(fg: np.ndarray, alpha: np.ndarray, tf, canvas):
    """Warp the cut-out onto the canvas. Resample by `s` first (INTER_AREA when shrinking), then rotate/translate."""
    s, R, t = tf
    W, H = canvas
    h, w = alpha.shape
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    fg_s = cv2.resize(fg, (nw, nh), interpolation=interp)
    a_s = cv2.resize(alpha, (nw, nh), interpolation=interp)
    M = np.hstack([R, t[:, None]]).astype(np.float32)
    fg_c = cv2.warpAffine(fg_s, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a_c = cv2.warpAffine(a_s, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _f32(fg_c), np.clip(a_c, 0, 1)


# ============================================================================ 2. background layer
def background_layer(scene: Scene, W: int, H: int):
    p = scene.background
    if p is None or p.source == "neutral":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx / W - 0.5) ** 2) + ((yy / H - 0.45) ** 2))
        g = np.clip(0.97 - 0.22 * d ** 1.4 * 2.2, 0.55, 1.0)
        return np.repeat(g[..., None], 3, axis=2)
    gw = min(max(W, H), CONFIG.bg_gen_edge)
    gx, gy = max(2, round(W * gw / max(W, H))), max(2, round(H * gw / max(W, H)))
    arr, info = backgrounds.get_background(scene.brand, gx, gy, p.source if p.source != "transparent" else "procedural",
                                           CONFIG.hf_token or None, CONFIG.hf_model, p.seed)
    if "failed" in info:                                      # a real error (network / quota); "no token" is just fallback
        _warn("Background: " + info)
    bg = _f32(arr)
    if (gx, gy) != (W, H):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
        bg += np.random.default_rng(p.seed).normal(0, 0.006, (H, W, 1)).astype(np.float32)
    return np.clip(bg, 0, 1)


# ============================================================================ 3. grounded dual shadows
def apply_shadows(bg: np.ndarray, a: np.ndarray, light_angle: float, p: ShadowParams) -> np.ndarray:
    H, W = a.shape
    info = _mask_info(a)
    if not info["valid"]:
        return bg
    lx, ly = toward_light(light_angle)
    m = a > 0.5
    valid = m.any(axis=0)
    last = H - 1 - np.argmax(m[::-1], axis=0)                     # lowest opaque row of every column = bottom contour
    cols = valid.astype(np.float32)
    k = max(3, int(0.02 * W)) | 1
    kern = np.ones(k, np.float32)
    yb = np.convolve(np.where(valid, last, 0).astype(np.float32), kern, "same") / np.maximum(np.convolve(cols, kern, "same"), 1)
    # tight contact shadow hugging the base of the vessel
    yy = np.arange(H, dtype=np.float32)[:, None]
    band = ((yy >= yb[None, :] - 0.004 * H) & (yy <= yb[None, :] + 0.012 * H * p.softness)) & valid[None, :]
    contact = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 0.0045 * W * p.softness)
    contact = cv2.warpAffine(contact, np.float32([[1, 0, -lx * 0.004 * W], [0, 1, 0]]), (W, H))
    contact = np.clip(contact / max(float(contact.max()), 1e-6), 0, 1)
    # wide ambient drop shadow thrown away from the light, flattened onto the table plane
    off = 0.045 * W
    sy = 0.45
    M = np.float32([[1, 0, -lx * off], [0, sy, -ly * off * 0.5 + (1 - sy) * info["y1"]]])
    amb = cv2.warpAffine(a.astype(np.float32), M, (W, H))
    amb = cv2.GaussianBlur(amb, (0, 0), 0.03 * W * p.softness)
    shade = (1 - p.contact_opacity * contact) * (1 - p.ambient_opacity * np.clip(amb, 0, 1))
    return bg * shade[..., None]


# ============================================================================ 4. natural sunlight
def apply_sunlight(img: np.ndarray, a: Optional[np.ndarray], angle: float, p: SunlightParams) -> np.ndarray:
    """Directional golden-hour shader: warm gradient across the upper-left quadrant (for angle=135), window-blind
    falloff stripes, and gentle darkening away from the light."""
    H, W = img.shape[:2]
    info = _mask_info(a) if a is not None else dict(cx=W / 2, cy=H / 2, valid=False)
    lx, ly = toward_light(angle)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    diag = math.hypot(W, H)
    t = ((xx - info["cx"]) * lx + (yy - info["cy"]) * ly) / diag             # >0 toward the light
    g = np.clip(0.5 + 1.15 * t, 0, 1) ** 1.35
    warm = np.array([1.00, 0.84, 0.58], np.float32)
    s = p.strength
    out = 1 - (1 - img) * (1 - warm * (0.40 * s) * g[..., None])             # screen blend: warm golden light
    out *= (1 - 0.14 * s * (1 - g))[..., None]                                # falloff away from the window
    if p.blinds > 0:
        u = (-xx * ly + yy * lx) / W                                         # coordinate across the light direction
        stripes = _smoothstep(0.30, 0.70, 0.5 + 0.5 * np.sin(2 * np.pi * (u / 0.115 + 0.13)))
        out *= (1 - (0.13 * p.blinds * s) * (1 - stripes) * g ** 1.2)[..., None]
    return np.clip(out, 0, 1)


def iclight_relight(rgb_u8: np.ndarray, alpha: np.ndarray, prompt: str, angle: float) -> Optional[np.ndarray]:
    """Optional neural relighting through the public IC-Light Space (gradio_client). Uploads the dish cut-out - so
    it is opt-in. The endpoint's parameters are discovered at run time; on any failure the local shader is used."""
    try:
        from gradio_client import Client, handle_file
        client = Client("lllyasviel/IC-Light", hf_token=CONFIG.hf_token or None, verbose=False)
        api = client.view_api(return_format="dict", print_info=False)
        named = api.get("named_endpoints", {})
        ep = next((n for n in named if "relight" in n.lower()), None)
        if ep is None:
            raise RuntimeError("no relight endpoint found")
        h, w = alpha.shape
        k = min(1.0, 768 / max(h, w))
        small = cv2.resize(rgb_u8, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(alpha, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        rgba = np.dstack([small, _u8(a_s)])
        tmp = Path(tempfile.mkdtemp()) / "fg.png"
        Image.fromarray(rgba, "RGBA").save(tmp)
        lx, ly = toward_light(angle)
        side = "Left Light" if lx < -0.4 else "Right Light" if lx > 0.4 else ("Top Light" if ly < 0 else "Bottom Light")
        kwargs = {}
        for prm in named[ep]["parameters"]:
            name = prm["parameter_name"]
            low = name.lower()
            if "prompt" in low and not low.startswith(("a_", "n_")):
                kwargs[name] = prompt
            elif low in ("input_fg", "fg", "image", "input_image") or low.endswith("_fg"):
                kwargs[name] = handle_file(str(tmp))
            elif low in ("input_bg", "bg"):
                kwargs[name] = None
            elif low == "bg_source":
                kwargs[name] = side
            elif low == "image_width":
                kwargs[name] = small.shape[1] // 64 * 64 or 512
            elif low == "image_height":
                kwargs[name] = small.shape[0] // 64 * 64 or 512
            elif low == "num_samples":
                kwargs[name] = 1
            elif prm.get("parameter_has_default"):
                kwargs[name] = prm.get("parameter_default")
        res = client.predict(api_name=ep, **kwargs)
        first = res[0] if isinstance(res, (list, tuple)) else res
        first = first[0] if isinstance(first, (list, tuple)) else first
        path = first.get("image") if isinstance(first, dict) else first
        if isinstance(path, dict):
            path = path.get("path")
        out = np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS))
        return out
    except Exception as e:
        _warn(f"IC-Light unavailable ({type(e).__name__}: {str(e)[:100]}) - used the local sunlight shader")
        return None


# ============================================================================ 5. shahi rim + candlelight glow
def rim_background(bg: np.ndarray, a: np.ndarray, p: RimParams) -> np.ndarray:
    """Warm candlelight glow behind the dish and deeper ambient shadows around it."""
    H, W = a.shape
    info = _mask_info(a)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - info["cx"]) / (0.55 * W)) ** 2 + ((yy - info["cy"]) / (0.55 * H)) ** 2)
    glow = np.exp(-(r ** 2) * 2.2)
    amber = np.array([1.0, 0.62, 0.22], np.float32)
    # a pool of candlelight on the table: lifts the dark wood around the dish so its shadows can be seen
    out = bg + glow[..., None] * (0.06 + amber * (0.34 * p.intensity))
    edge = _smoothstep(0.55, 1.35, r)
    return np.clip(out * (1 - 0.35 * p.intensity * edge)[..., None], 0, 1)


def rim_foreground(comp: np.ndarray, a: np.ndarray, p: RimParams) -> np.ndarray:
    """Amber inner-rim specular highlight along the top-right edge of the vessel + deeper shadow tones."""
    H, W = a.shape
    k = max(2, int(0.006 * W))
    inner = np.clip(a - cv2.erode(a, np.ones((k * 2 + 1, k * 2 + 1), np.uint8)), 0, 1)       # rim band, inside the mask
    sm = cv2.GaussianBlur(a, (0, 0), max(1.5, 0.004 * W))
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    nrm = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / nrm, -gy / nrm                                                              # outward normal
    d = np.array(toward_light(45.0), np.float32)                                               # top-right light
    facing = np.clip(nx * d[0] + ny * d[1], 0, 1) ** 2.2
    rim = cv2.GaussianBlur(inner * facing, (0, 0), 0.0025 * W)
    amber = np.array([1.0, 0.68, 0.26], np.float32)
    out = 1 - (1 - comp) * (1 - amber * (1.1 * p.intensity) * rim[..., None])                  # screen: amber rim light
    lum = comp.mean(axis=2, keepdims=True)
    shade = 1 - 0.16 * p.intensity * (1 - lum) * a[..., None]                                  # deeper shadow tones on the dish
    return np.clip(out * shade, 0, 1)


# ============================================================================ 7. LAB colour: CLAHE + warm chroma + glisten
def lab_pop(rgb: np.ndarray, a: Optional[np.ndarray], p: LabParams) -> np.ndarray:
    """CLAHE on L only (crisp rice grains / paneer / roti), +12 % warm chroma on A and B, specular glisten."""
    u8 = _u8(rgb)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    s = p.strength
    Lc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    Lf = L.astype(np.float32) + (Lc.astype(np.float32) - L.astype(np.float32)) * s
    boost = 1 + 0.12 * s
    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)
    Af = np.where(Af > 128, 128 + (Af - 128) * boost, Af)                                      # warm reds
    Bf = np.where(Bf > 128, 128 + (Bf - 128) * boost, Bf)                                      # warm yellows
    hi0 = float(np.percentile(Lf, 90))
    hi = np.clip((Lf - hi0) / max(255 - hi0, 1), 0, 1) ** 1.5                                    # specular highlight map
    detail = Lf - cv2.GaussianBlur(Lf, (0, 0), max(1.0, 0.01 * lab.shape[1]))
    Lf = Lf + s * (0.9 * detail * hi + 7.0 * hi)
    out = cv2.cvtColor(cv2.merge([np.clip(Lf, 0, 255), np.clip(Af, 0, 255), np.clip(Bf, 0, 255)]).astype(np.uint8),
                       cv2.COLOR_LAB2RGB)
    out = _f32(out)
    if a is not None:
        m = np.clip(a, 0, 1)[..., None]
        return out * m + rgb * (1 - m)
    return out


# ============================================================================ 8. steam
def steam_layer(a: np.ndarray, p: SteamParams) -> np.ndarray:
    """Wispy white steam rising from the top centre of the food: multi-octave Perlin noise, column envelopes that
    widen with height, vertical motion blur and feathered alpha. Returns alpha (H, W) in [0, 1]."""
    H, W = a.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((H, W), np.float32)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))          # the main vessel (not a side bowl)
    cx0, cy0, bw, bh = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    ph = int(0.30 * H)                                         # plume height
    top = cy0 + int(0.04 * bh)                                 # start slightly inside the dish so it blends
    top_cols = np.where((lab[max(top - 2, 0):top + 3] == i).any(axis=0))[0]
    xc = float(top_cols.mean()) if top_cols.size else cx0 + bw / 2
    layer = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(p.seed)
    for k, dx in enumerate((-0.14, 0.0, 0.15)):
        x_mid = xc + dx * bw
        pw = int(0.34 * bw)
        n = fbm((ph // 2, pw // 2), base_res=(3, 2), octaves=5, persistence=0.55, seed=p.seed + 17 * k)
        n = cv2.resize(n, (pw, ph), interpolation=cv2.INTER_CUBIC)
        t = np.linspace(1, 0, ph, dtype=np.float32)[:, None]                  # row 0 = plume tip (t=1), last row = base (t=0)
        xs = (np.arange(pw, dtype=np.float32)[None, :] - pw / 2) / (pw / 2)
        sway = 0.25 * np.sin(t * 6.0 + rng.uniform(0, 6.28)) * t
        sigma = 0.30 + 0.55 * t                                                 # plume widens as it rises
        env = np.exp(-(((xs - sway) / sigma) ** 2)) * (1 - t) ** 1.15 * (0.35 + 0.65 * _smoothstep(0.0, 0.12, 1 - t))
        wisp = np.clip((n - 0.40) * 3.4, 0, 1) ** 1.2 * env
        y0, x0 = top - ph, int(x_mid - pw / 2)
        ys, ye = max(y0, 0), min(y0 + ph, H)
        xs0, xe = max(x0, 0), min(x0 + pw, W)
        if ye > ys and xe > xs0:
            layer[ys:ye, xs0:xe] = np.maximum(layer[ys:ye, xs0:xe], wisp[ys - y0:ye - y0, xs0 - x0:xe - x0])
    klen = max(3, int(0.05 * H)) | 1
    kern = np.zeros((klen, 1), np.float32)
    kern[:, 0] = 1.0 / klen
    layer = cv2.filter2D(layer, -1, kern)                                      # vertical motion blur
    layer = cv2.GaussianBlur(layer, (0, 0), 0.006 * W)                         # feather
    return np.clip(layer * 1.6, 0, 1)


# ============================================================================ 9. DSLR bokeh depth of field
def _disc(r: int) -> np.ndarray:
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1, cv2.LINE_AA)
    return k / k.sum()


def apply_bokeh(bg: np.ndarray, a: np.ndarray, p: BokehParams) -> np.ndarray:
    """Progressive lens blur on the background only: strongest at the back/top of the table, zero at the front plane
    of the dish. Disc kernels give round bokeh balls from bright highlights."""
    if p.strength <= 0:
        return bg
    H, W = a.shape
    info = _mask_info(a)
    y_front = max(info["y1"], int(0.5 * H))
    scale = 0.5 if max(W, H) > 1200 else 1.0
    sw, sh = max(2, int(W * scale)), max(2, int(H * scale))
    small = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_AREA)
    small = small + 0.8 * np.maximum(small - 0.78, 0) / 0.22                    # emphasise highlights -> bokeh balls
    rmax = max(2, int(0.028 * sw * p.strength))
    radii = [0] + [max(1, int(round(rmax * k / 4))) for k in range(1, 5)]
    levels = [small] + [cv2.filter2D(small, -1, _disc(r), borderType=cv2.BORDER_REFLECT) for r in radii[1:]]
    levels = [cv2.resize(l, (W, H), interpolation=cv2.INTER_CUBIC) for l in levels]
    y = np.arange(H, dtype=np.float32)
    depth = np.clip((y_front * 0.92 - y) / max(y_front * 0.92, 1), 0, 1) ** 1.1       # 1 far/top, 0 at the front plane
    pos = depth * (len(levels) - 1)
    out = np.zeros_like(bg)
    for i, lev in enumerate(levels):
        w = np.clip(1 - np.abs(pos - i), 0, 1)[:, None, None]
        out += lev * w
    return np.clip(out, 0, 1)


# ============================================================================ brand colour grade
def apply_grade(img: np.ndarray, brand, p: GradeParams) -> np.ndarray:
    s = p.strength
    warmth = (brand.warmth if p.warmth is None else p.warmth) * s
    contrast = (brand.contrast if p.contrast is None else p.contrast) * s
    sat = (brand.saturation if p.saturation is None else p.saturation) * s
    vig = (brand.vignette if p.vignette is None else p.vignette) * s
    out = img.copy()
    out[..., 0] *= 1 + 0.60 * warmth
    out[..., 1] *= 1 + 0.10 * warmth
    out[..., 2] *= 1 - 0.60 * warmth
    m = float(out.mean())
    out = (out - m) * (1 + contrast) + m
    hsv = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + sat), 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if vig > 0:
        H, W = out.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx / W - 0.5) * 2) ** 2 + ((yy / H - 0.5) * 2) ** 2) / math.sqrt(2)
        out *= (1 - vig * _smoothstep(0.35, 1.0, r) ** 1.4)[..., None]
    return np.clip(out, 0, 1)


# ============================================================================ 10. brand badge + dietary indicator
def _font(size: int, bold=True):
    for name in (("georgiab.ttf", "arialbd.ttf") if bold else ("georgia.ttf", "arial.ttf")):
        for base in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", ""):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _corner_xy(corner, W, H, w, h, m):
    x = m if "left" in corner else W - w - m
    y = m if "top" in corner else H - h - m
    return int(x), int(y)


def draw_diet_icon(size: int, diet: str) -> Image.Image:
    """Indian dietary mark: white rounded square with a coloured outline and a green circle (veg),
    amber circle (contains egg) or red triangle (non-veg)."""
    colour = {"veg": (14, 138, 60), "egg": (232, 163, 23), "nonveg": (179, 38, 30)}[diet]
    ss = 4
    S = size * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.14), fill=(255, 255, 255, 235), outline=colour + (255,),
                        width=max(int(S * 0.07), 2))
    c, r = S / 2, S * 0.27
    if diet == "nonveg":
        d.polygon([(c, c - r * 1.15), (c - r * 1.1, c + r * 0.85), (c + r * 1.1, c + r * 0.85)], fill=colour + (255,))
    else:
        d.ellipse((c - r, c - r, c + r, c + r), fill=colour + (255,))
    return im.resize((size, size), Image.LANCZOS)


def apply_badge(img: Image.Image, brand, p: BadgeParams) -> Image.Image:
    W, H = img.size
    base = img.convert("RGBA")
    m = int(0.04 * W)                                                      # safe margin
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    acc, pri = _hex(brand.secondary), _hex(brand.primary)
    if p.border:
        inset, t = int(0.022 * W), max(2, int(0.003 * W))
        d.rectangle((inset, inset, W - inset, H - inset), outline=acc + (230,), width=t)
        if brand.style == "royal":
            d.rectangle((inset + 3 * t, inset + 3 * t, W - inset - 3 * t, H - inset - 3 * t), outline=acc + (120,), width=max(1, t // 2))
    if p.show_brand:
        font = _font(max(12, int(0.030 * W)))
        text = brand.name.upper()
        bb = d.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = int(0.014 * W)
        bw, bh = tw + 2 * pad, th + 2 * pad
        x, y = _corner_xy(p.corner, W, H, bw, bh, m)
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=bh // 2, fill=pri + (235,), outline=acc + (255,), width=max(2, int(0.0025 * W)))
        d.text((x + pad - bb[0], y + pad - bb[1]), text, font=font, fill=acc + (255,))
    if p.show_diet:
        diet = p.diet if p.diet in DIET_LABELS else brand.default_diet
        if diet not in brand.diets:
            diet = brand.default_diet
        size = max(24, int(0.058 * W))
        icon = draw_diet_icon(size, diet)
        x, y = _corner_xy(p.diet_corner, W, H, size, size, m)
        over.alpha_composite(icon, (x, y))
    return Image.alpha_composite(base, over).convert("RGB")


# ============================================================================ render
def render(scene: Scene, long_edge: int = 1280) -> Image.Image:
    """Compose the scene. Fixed layer order, so any combination / click order of enhancements gives the same result."""
    CONFIG.warnings.clear()
    W, H = canvas_size(scene, long_edge)
    brand = BRANDS[scene.brand]
    light = scene.sunlight.angle if (scene.sunlight and scene.sunlight.angle is not None) else brand.light_angle

    if scene.cutout is None:                                              # photo as shot (+ global, mask-free steps)
        img = _f32(np.asarray(_cover(scene.src, W, H)))
        a = None
    else:
        fg_src, a_src = scene.cutout.at_src_size(scene.src)
        tf = None
        if scene.framing is not None:
            tf = framing_transform(a_src, (W, H), scene.framing)
        tf = tf or cover_transform(scene.src.size, (W, H))
        fg, a = place(fg_src, a_src, tf, (W, H))
        transparent = scene.background is not None and scene.background.source == "transparent"
        if scene.lab is not None:
            fg = lab_pop(fg, a, scene.lab)
        if scene.sunlight is not None and scene.sunlight.use_iclight and CONFIG.allow_remote_relight:
            relit = iclight_relight(_u8(fg), a, scene.sunlight.iclight_prompt, light)
            if relit is not None:
                m = np.clip(a, 0, 1)[..., None]
                fg = fg * (1 - 0.8 * m) + _f32(relit) * 0.8 * m
        if scene.sunlight is not None and scene.sunlight.use_iclight and not CONFIG.allow_remote_relight:
            _warn("IC-Light was requested but remote relighting is switched off in the sidebar - local shader used")
        if transparent:                                                    # clean cut-out on transparency, no backdrop
            img = fg
        else:
            bg = background_layer(scene, W, H)
            if scene.rim is not None:
                bg = rim_background(bg, a, scene.rim)
            if scene.bokeh is not None:
                bg = apply_bokeh(bg, a, scene.bokeh)
            if scene.shadows is not None:
                bg = apply_shadows(bg, a, light, scene.shadows)
            img = bg * (1 - a[..., None]) + fg * a[..., None]
        if scene.rim is not None:
            img = rim_foreground(img, a, scene.rim)

    if scene.lab is not None and scene.cutout is None:
        img = lab_pop(img, None, scene.lab)
    if scene.sunlight is not None:
        img = apply_sunlight(img, a, light, scene.sunlight)
    if scene.steam is not None and a is not None:
        s = steam_layer(a, scene.steam)[..., None]
        img = img * (1 - s * scene.steam.opacity) + np.array([0.97, 0.96, 0.94], np.float32) * s * scene.steam.opacity
    if scene.grade is not None:
        img = apply_grade(img, brand, scene.grade)

    out = Image.fromarray(_u8(img), "RGB")
    if scene.cutout is not None and scene.background is not None and scene.background.source == "transparent":
        alpha = Image.fromarray(_u8(a), "L")
        out = out.convert("RGBA")
        out.putalpha(alpha)
        return out
    if scene.badge is not None:
        out = apply_badge(out, brand, scene.badge)
    return out


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    return backgrounds.cover_resize(img, w, h)


# ============================================================================ ops (what the UI buttons / recipes call)
def _params(cls, kw: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in names})


def ensure_cutout(scene: Scene, model: Optional[str] = None) -> Scene:
    model = model or CONFIG.isolate_model
    if scene.cutout is not None:
        return scene                                              # matte already cached: never re-run removal
    return replace(scene, cutout=isolate(scene.src, model))


def op_isolate(scene, model=None, **_):
    if scene.cutout is not None and model and model != scene.cutout.model:
        return replace(scene, cutout=isolate(scene.src, model))
    return ensure_cutout(scene, model)


def _needs_mask(setter):
    def f(scene, **kw):
        return setter(ensure_cutout(scene), **kw)
    return f


OPS = {
    "isolate": op_isolate,
    "background": _needs_mask(lambda s, **kw: replace(s, background=_params(BackgroundParams, kw))),
    "shadows": _needs_mask(lambda s, **kw: replace(s, shadows=_params(ShadowParams, kw))),
    "framing": _needs_mask(lambda s, **kw: replace(s, framing=_params(FramingParams, kw))),
    "rim_glow": _needs_mask(lambda s, **kw: replace(s, rim=_params(RimParams, kw))),
    "steam": _needs_mask(lambda s, **kw: replace(s, steam=_params(SteamParams, kw))),
    "bokeh": _needs_mask(lambda s, **kw: replace(s, bokeh=_params(BokehParams, kw))),
    "sunlight": lambda s, **kw: replace(s, sunlight=_params(SunlightParams, kw)),
    "lab_pop": lambda s, **kw: replace(s, lab=_params(LabParams, kw)),
    "brand_grade": lambda s, **kw: replace(s, grade=_params(GradeParams, kw)),
    "badge": lambda s, **kw: replace(s, badge=_params(BadgeParams, kw)),
}

LABELS = {
    "isolate": "Clean dish isolation", "background": "AI theme background", "shadows": "Grounded dual shadows",
    "sunlight": "Natural sunlight touch", "rim_glow": "Shahi rim & candlelight glow", "framing": "Angle, tilt & framing fix",
    "lab_pop": "Ghee, gravy & masala pop", "steam": "Fresh steam effect", "bokeh": "DSLR bokeh depth of field",
    "badge": "Brand badge & dietary icon", "brand_grade": "Brand colour grade",
}


MASK_OPS = {"isolate", "background", "shadows", "framing", "rim_glow", "steam", "bokeh"}   # steps that need the matte


def apply_op(scene: Scene, op: str, params: dict) -> Scene:
    return OPS[op](scene, **params)


# ============================================================================ recipe
@dataclass
class EnhancementRecipe:
    brand: str
    ratio: str = "1:1"
    isolate_model: str = "isnet-general-use"
    steps: list = field(default_factory=list)            # [{"op": "...", "params": {...}}]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "EnhancementRecipe":
        d = json.loads(text)
        return EnhancementRecipe(**{k: d[k] for k in ("brand", "ratio", "isolate_model", "steps", "version") if k in d})

    def has(self, op: str) -> bool:
        return any(s["op"] == op for s in self.steps)

    def without(self, *ops: str) -> "EnhancementRecipe":
        return replace(self, steps=[s for s in self.steps if s["op"] not in ops])

    def with_ratio(self, ratio: str) -> "EnhancementRecipe":
        return replace(self, ratio=ratio)

    def apply_to(self, src: Image.Image, brand: Optional[str] = None, cutout: Optional[Cutout] = None) -> Scene:
        scene = Scene(src=prepare_source(src), brand=brand or self.brand, ratio=self.ratio, cutout=cutout)
        for st in self.steps:
            params = dict(st.get("params", {}))
            if st["op"] == "isolate":
                params.setdefault("model", self.isolate_model)
            scene = apply_op(scene, st["op"], params)
        return scene


def master_preset(brand_key: str, ratio: str = "1:1", include_badge: bool = False, model: Optional[str] = None) -> EnhancementRecipe:
    """The brand's 1-click recipe (badge only if explicitly requested)."""
    steps = [{"op": op, "params": dict(p)} for op, p in BRANDS[brand_key].preset]
    if include_badge:
        steps.append({"op": "badge", "params": {}})
    return EnhancementRecipe(brand=brand_key, ratio=ratio, isolate_model=model or CONFIG.isolate_model, steps=steps)


# ============================================================================ undo / redo session
@dataclass
class HistoryEntry:
    image: Image.Image
    label: str                                           # the action that was applied AFTER this state
    scene: Scene
    steps: list


class ImageSession:
    """One photo with its own non-destructive history. `undo_stack` / `redo_stack` hold PIL images + action labels."""

    def __init__(self, name: str, src: Image.Image, brand: str, ratio: str = "1:1", preview_edge: int = 1100):
        self.name = name
        self.preview_edge = preview_edge
        self.original = prepare_source(src)
        self.initial = Scene(src=self.original, brand=brand, ratio=ratio)
        self.scene = self.initial
        self.steps: list = []
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self.before = render(self.initial, preview_edge)
        self.current = self.before
        self.cached_cutout: Optional[Cutout] = None
        self.warnings: list = []

    # -- cached matte, exposed for inspection
    @property
    def alpha_matte(self) -> Optional[np.ndarray]:
        return self.scene.cutout.alpha if self.scene.cutout is not None else None

    def _commit(self, label, new_scene, new_steps):
        self.undo_stack.append(HistoryEntry(self.current, label, self.scene, self.steps))
        self.redo_stack.clear()
        if new_scene.cutout is not None:
            self.cached_cutout = new_scene.cutout                         # the matte is computed once per photo
        self.scene, self.steps = new_scene, new_steps
        self.current = render(new_scene, self.preview_edge)
        self.warnings = list(CONFIG.warnings)
        del self.undo_stack[:-25]                                       # bound memory

    def _step(self, scene: Scene, steps: list, op: str, params: dict):
        """Apply one op to (scene, steps) without rendering; reuses the cached matte and records implicit isolation."""
        steps = list(steps)
        if op in MASK_OPS and scene.cutout is None:
            if self.cached_cutout is not None and not (op == "isolate" and params.get("model") not in (None, self.cached_cutout.model)):
                scene = replace(scene, cutout=self.cached_cutout)          # reuse the cached matte, no re-extraction
            if op != "isolate" and not any(s["op"] == "isolate" for s in steps):
                steps.append({"op": "isolate", "params": {}})              # make the implicit isolation explicit in the recipe
        scene = apply_op(scene, op, params)
        if scene.cutout is not None:
            self.cached_cutout = scene.cutout
        return scene, steps + [{"op": op, "params": dict(params)}]

    def apply(self, op: str, **params) -> str:
        scene, steps = self._step(self.scene, self.steps, op, params)
        label = LABELS.get(op, op)
        self._commit(label, scene, steps)
        return label

    def apply_steps(self, steps: list, label: str) -> str:
        """Apply several ops (e.g. a brand master preset) as ONE undoable action; renders only once."""
        scene, rec = self.scene, list(self.steps)
        for st in steps:
            scene, rec = self._step(scene, rec, st["op"], dict(st.get("params", {})))
        self._commit(label, scene, rec)
        return label

    def set_brand(self, brand: str):
        if brand != self.scene.brand:
            self._commit(f"Switch brand to {BRANDS[brand].name}", replace(self.scene, brand=brand), self.steps)

    def set_ratio(self, ratio: str):
        if ratio != self.scene.ratio:
            self._commit(f"Canvas ratio {ratio}", replace(self.scene, ratio=ratio), self.steps)

    def reset(self) -> str:
        self._commit("Reset to original", self.initial, [])        # the cached matte stays available for later steps
        return "Reset to original"

    def undo(self) -> Optional[str]:
        if not self.undo_stack:
            return None
        e = self.undo_stack.pop()
        self.redo_stack.append(HistoryEntry(self.current, e.label, self.scene, self.steps))
        self.current, self.scene, self.steps = e.image, e.scene, e.steps
        return e.label

    def redo(self) -> Optional[str]:
        if not self.redo_stack:
            return None
        e = self.redo_stack.pop()
        self.undo_stack.append(HistoryEntry(self.current, e.label, self.scene, self.steps))
        self.current, self.scene, self.steps = e.image, e.scene, e.steps
        return e.label

    @property
    def can_undo(self):
        return bool(self.undo_stack)

    @property
    def can_redo(self):
        return bool(self.redo_stack)

    def recipe(self, isolate_model: Optional[str] = None) -> EnhancementRecipe:
        return EnhancementRecipe(brand=self.scene.brand, ratio=self.scene.ratio,
                                 isolate_model=isolate_model or (self.scene.cutout.model if self.scene.cutout else CONFIG.isolate_model),
                                 steps=[dict(s) for s in self.steps])

    def render_full(self, long_edge: int) -> Image.Image:
        return render(self.scene, long_edge)


# ============================================================================ upload-ready export
@dataclass(frozen=True)
class ExportProfile:
    key: str
    name: str
    fmt: str                 # JPEG | PNG
    long_edge: int
    max_bytes: Optional[int]
    ratio: str
    min_edge: int
    allow_overlays: bool     # badge / border


PROFILES = {
    "swiggy_zomato": ExportProfile("swiggy_zomato", "Swiggy / Zomato upload-ready", "JPEG", 2048, 1_000_000, "1:1", 1000, False),
    "hires_png": ExportProfile("hires_png", "High-resolution PNG (no size cap)", "PNG", 3072, None, "1:1", 1000, True),
    "hires_jpg": ExportProfile("hires_jpg", "High-resolution JPEG (q95)", "JPEG", 3072, None, "1:1", 1000, True),
}


def save_image(img: Image.Image, path, profile: ExportProfile) -> dict:
    """Write the final file. JPEG profiles with a byte cap search quality (then size) until the file fits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    if profile.fmt == "PNG":
        img.save(tmp, "PNG", optimize=True)
    else:
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img = img.convert("RGB")
        data = None
        cur = img
        while data is None:
            for q in (95, 92, 88, 84, 80, 76, 72):
                buf = io.BytesIO()
                cur.save(buf, "JPEG", quality=q, subsampling=0 if q >= 92 else 2, optimize=True)
                if profile.max_bytes is None or buf.tell() <= profile.max_bytes:
                    data = buf.getvalue()
                    break
            else:
                cur = cur.resize((int(cur.width * 0.92), int(cur.height * 0.92)), Image.LANCZOS)
        tmp.write_bytes(data)
        img = cur
    os.replace(tmp, path)
    return {"path": str(path), "bytes": path.stat().st_size, "width": img.width, "height": img.height}
