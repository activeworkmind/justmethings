"""Photo-quality helpers used by the review GUI / finalize: daylight tone, sharpening, tilt estimation and
dish detection. Strictly photographic and non-generative - food is never added, removed or repainted."""
import math

import cv2
import numpy as np

WORK_EDGE = 2600          # develop at this long edge, final frame is cropped from it
TARGET_LUM = 0.52         # exposure target (0-1)
WARMTH = 1.04             # R gain; B gets 1/WARMTH -> subtle golden-hour/daylight warmth
SAT_BOOST = 1.10
SHADOW_LIFT = 0.18
CLAHE_CLIP = 1.6


def estimate_tilt(bgr: np.ndarray):
    """Median angle (degrees, positive = rotate counter-clockwise to level) of long near-horizontal/vertical
    lines (mat, box, table edges). None when there is no reliable evidence."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(g, 60, 160)
    h, w = g.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 120, minLineLength=w // 4, maxLineGap=20)
    if lines is None:
        return None
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45          # fold to [-45, 45): distance from nearest axis
        if abs(a) <= 6:
            angs.append(a)
    if len(angs) < 4:
        return None
    return float(np.median(angs))


def _dish_cluster(bgr: np.ndarray):
    """Bounding box of the white crockery (plate + side bowls). Returns (x0,y0,x1,y1) or None.
    Large low-saturation regions (table, box, mat) are rejected by size and compactness."""
    h, w = bgr.shape[:2]
    s = 600 / max(h, w)
    hsv = cv2.cvtColor(cv2.resize(bgr, None, fx=s, fy=s), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 205) & (hsv[..., 1] < 45)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    # plates/bowls are rings (white rim, food inside): fill each outer contour to get a solid disc
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = mask.shape[0] * mask.shape[1]
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        fill = a / max(bw * bh, 1)
        round_ish = 0.6 < bw / max(bh, 1) < 1.6 and 0.60 < fill < 0.88   # circle ~0.785, rectangle ~1
        if 0.012 * area_img < bw * bh < 0.45 * area_img and round_ish:
            boxes.append((x, y, x + bw, y + bh))
    if not boxes:
        return None
    b = np.array(boxes)
    return tuple(v / s for v in (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()))


def _daylight(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32) / 255.0
    # 1. daylight warmth only (gray-world is fooled by coloured backgrounds, so no auto-WB here;
    #    the RAW develop already applies the camera white balance)
    f[..., 2] *= WARMTH
    f[..., 0] /= WARMTH
    f = np.clip(f, 0, 1)
    # 2. exposure toward target (limited to +-0.6 EV)
    lum = float(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).mean() / 255)
    ev = float(np.clip(math.log2(TARGET_LUM / max(lum, 1e-3)), -0.6, 0.6))
    f = np.clip(f * (2 ** ev), 0, 1)
    # 3. shadow lift + soft highlight roll-off (tone curve on luminance)
    y = f.mean(axis=2, keepdims=True)
    lifted = y + SHADOW_LIFT * (1 - y) ** 2 * y ** 0.5 * 2
    lifted = lifted / (1 + 0.08 * lifted ** 3)
    f = np.clip(f * (lifted / (y + 1e-4)), 0, 1)
    u8 = (f * 255).astype(np.uint8)
    # 4. local contrast on lightness + vibrance
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(lab[..., 0])
    u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    boost = 1 + (SAT_BOOST - 1) * (1 - hsv[..., 1] / 255)     # protect already-saturated colours
    hsv[..., 1] = np.clip(hsv[..., 1] * boost, 0, 255)
    u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return u8


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.6)
    return cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
