"""Gradient (Perlin) noise and fractal Brownian motion, fully vectorised with numpy."""
import numpy as np


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin2d(shape, res, seed=0) -> np.ndarray:
    """2-D Perlin gradient noise on a (h, w) grid with `res` = (cells_y, cells_x). Returns float32 in ~[-1, 1]."""
    h, w = shape
    ry, rx = max(int(res[0]), 1), max(int(res[1]), 1)
    rng = np.random.default_rng(seed)
    ang = rng.random((ry + 1, rx + 1)) * 2 * np.pi
    gx, gy = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    y = np.arange(h, dtype=np.float32) * (ry / h)
    x = np.arange(w, dtype=np.float32) * (rx / w)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    y0, x0 = y0[:, None], x0[None, :]

    def corner(iy, ix, dy, dx):
        return gx[iy, ix] * dx + gy[iy, ix] * dy

    n00 = corner(y0, x0, fy, fx)
    n10 = corner(y0, x0 + 1, fy, fx - 1)
    n01 = corner(y0 + 1, x0, fy - 1, fx)
    n11 = corner(y0 + 1, x0 + 1, fy - 1, fx - 1)
    u, v = _fade(fx), _fade(fy)
    return ((n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v) * np.float32(1.41421356)


def fbm(shape, base_res=(4, 4), octaves=5, persistence=0.5, lacunarity=2.0, seed=0) -> np.ndarray:
    """Fractal noise normalised to [0, 1] (sum of `octaves` Perlin layers)."""
    total = np.zeros(shape, np.float32)
    amp, norm = 1.0, 0.0
    ry, rx = base_res
    for o in range(octaves):
        total += amp * perlin2d(shape, (ry * lacunarity ** o, rx * lacunarity ** o), seed + 101 * o)
        norm += amp
        amp *= persistence
    total = total / norm
    return np.clip(total * 0.5 + 0.5, 0, 1)
