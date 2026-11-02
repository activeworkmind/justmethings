"""Manual review GUI (Tkinter, keyboard-driven). Launch: python -m kitchen gui

One card per output slot (menu item x rank). Every decision is saved to the SQLite database right away
(table `slots`) and the Excel tracker is kept in sync automatically (see sync.py), so closing the
window never loses work."""
import json
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import catalog, config, db, enhance, finalize, slotstore
from .sync import Syncer

ASPECTS = {"free": None, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}   # free-style is the default
HANDLE = 14
MIN_PX = 60
TILT_MAX = 15.0
BG = "#1b1b1b"
HELP = ("Enter/Space = approve + next  |  <- -> = prev/next  |  X = reject  |  R / Shift+R = rotate 90  |  "
        "TILT: slider, or , . = 0.5 deg, < > = 0.1 deg, 0 = reset, L = auto-level, G = level grid  |  "
        "E = enhance on/off  |  A = aspect  |  [ ] = other photo  |  N = edit name  |  P = preview  |  "
        "wheel = resize crop  |  drag = move / corners resize")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Cloud Kitchen - image review")
        root.configure(bg=BG)
        try:
            root.state("zoomed")
        except tk.TclError:
            root.geometry("1400x900")

        self.sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
        con = db.connect()
        self.files = {r["id"]: r["down_path"] for r in
                      con.execute("SELECT id,down_path FROM files WHERE status='downscaled'")}
        self._build_keys()
        self.state = slotstore.load_all()
        self.cache_base = {}
        self.exporting = False
        self._scale_busy = False
        self._tilting = False
        self._tilt_job = None
        self.sync = Syncer()

        self._build_ui()
        self.idx = next((i for i, k in enumerate(self.keys)
                         if self.state.get(k, {}).get("status", "pending") == "pending"), 0)
        self.load_slot()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._tick()

    def _build_keys(self):
        self.items = {it["item_id"]: it for it in catalog.active()}      # uncooked items have no slots
        self.keys = [f"{iid}:{r}" for iid in self.items
                     for r in range(1, min(len(self.sel.get(iid, [])), config.TOP_N) + 1)]

    # ---------- UI ----------
    def _build_ui(self):
        r = self.root
        self.title_var, self.info_var = tk.StringVar(), tk.StringVar()
        tk.Label(r, textvariable=self.title_var, bg=BG, fg="white",
                 font=("Segoe UI", 14, "bold"), anchor="w").pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(r, textvariable=self.info_var, bg=BG, fg="#aaa",
                 font=("Segoe UI", 10), anchor="w").pack(fill="x", padx=10)
        self.canvas = tk.Canvas(r, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=6)

        bar = tk.Frame(r, bg=BG)
        bar.pack(fill="x", padx=10, pady=2)
        tk.Label(bar, text="File name:", bg=BG, fg="white").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = tk.Entry(bar, textvariable=self.name_var, width=52, font=("Consolas", 11))
        self.name_entry.pack(side="left", padx=6)
        tk.Label(bar, text=".jpg", bg=BG, fg="#aaa").pack(side="left")
        self.enh_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Enhance (E)", variable=self.enh_var, bg=BG, fg="white",
                       selectcolor="#333", activebackground=BG, activeforeground="white",
                       command=self.commit).pack(side="left", padx=14)
        self.goto_var = tk.StringVar()
        tk.Label(bar, text="Go to item:", bg=BG, fg="white").pack(side="left", padx=(14, 2))
        g = tk.Entry(bar, textvariable=self.goto_var, width=8)
        g.pack(side="left")
        g.bind("<Return>", self.goto)
        self.status_var = tk.StringVar()
        tk.Label(bar, textvariable=self.status_var, bg=BG, fg="#7fd18b",
                 font=("Segoe UI", 11, "bold")).pack(side="right")

        # tilt panel
        tb = tk.Frame(r, bg=BG)
        tb.pack(fill="x", padx=10, pady=2)
        tk.Label(tb, text="Tilt", bg=BG, fg="white", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(tb, text="clockwise", bg=BG, fg="#888", font=("Segoe UI", 8)).pack(side="left", padx=(8, 2))
        self.scale = tk.Scale(tb, from_=-TILT_MAX, to=TILT_MAX, resolution=0.1, orient="horizontal", length=560,
                              showvalue=True, bg=BG, fg="white", troughcolor="#444", highlightthickness=0,
                              command=self._on_scale)
        self.scale.pack(side="left")
        self.scale.bind("<ButtonPress-1>", lambda e: self._tilt_drag(True))
        self.scale.bind("<ButtonRelease-1>", lambda e: self._tilt_drag(False))
        tk.Label(tb, text="counter-clockwise", bg=BG, fg="#888", font=("Segoe UI", 8)).pack(side="left", padx=(2, 10))
        for text, d in (("-0.1", -0.1), ("+0.1", 0.1), ("-0.5", -0.5), ("+0.5", 0.5)):
            tk.Button(tb, text=text, width=5, command=lambda d=d: self.tilt(d)).pack(side="left", padx=1)
        tk.Button(tb, text="Reset (0)", command=lambda: self.set_tilt(0.0)).pack(side="left", padx=(10, 3))
        tk.Button(tb, text="Auto-level (L)", command=self.auto_level).pack(side="left", padx=3)
        self.grid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tb, text="Level grid (G)", variable=self.grid_var, bg=BG, fg="white", selectcolor="#333",
                       activebackground=BG, activeforeground="white",
                       command=self.draw_overlay).pack(side="left", padx=10)

        btns = tk.Frame(r, bg=BG)
        btns.pack(fill="x", padx=10, pady=2)
        for text, cmd in [("<  Prev", self.prev), ("Approve + Next  (Enter)", self.approve),
                          ("Reject (X)", self.reject), ("Rotate left", lambda: self.rotate(-1)),
                          ("Rotate right", lambda: self.rotate(1)), ("Other photo  [", lambda: self.alt(-1)),
                          ("]", lambda: self.alt(1)), ("Preview (P)", self.preview),
                          ("Export approved -> final/", self.export)]:
            tk.Button(btns, text=text, command=cmd, padx=8).pack(side="left", padx=3)
        self.excel_var = tk.StringVar(value=self.sync.status)
        tk.Label(btns, textvariable=self.excel_var, bg=BG, fg="#8ab4f8", font=("Segoe UI", 9)).pack(side="right")
        tk.Label(r, text=HELP, bg=BG, fg="#777", font=("Segoe UI", 9), anchor="w",
                 wraplength=1500, justify="left").pack(fill="x", padx=10, pady=(2, 8))

        self.canvas.bind("<Configure>", lambda e: self.rebuild_view())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.commit())
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Motion>", self.on_hover)
        r.bind("<Key>", self.on_key)
        self.name_entry.bind("<Return>", lambda e: (self.canvas.focus_set(), self.commit()))
        self.name_entry.bind("<Escape>", lambda e: self.canvas.focus_set())
        g.bind("<Escape>", lambda e: self.canvas.focus_set())

    # ---------- automatic Excel <-> DB sync ----------
    def _tick(self):
        self.sync.tick()
        self.excel_var.set(self.sync.status)
        if self.sync.take_applied():
            self.reload_selection()
        self.root.after(1000, self._tick)

    def reload_selection(self):
        """Corrections from Excel were applied: reload picks; untouched pending slots follow the new picks."""
        old_key = self.key
        self.sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
        con = db.connect()
        for key, s in list(self.state.items()):
            pids = [p["photo_id"] for p in self.sel.get(key.split(":")[0], [])]
            if s["status"] == "pending" and s["photo_id"] not in pids:
                con.execute("DELETE FROM slots WHERE slot_key=?", (key,))
                del self.state[key]
        con.commit()
        self._build_keys()
        self.idx = self.keys.index(old_key) if old_key in self.keys else 0
        self.load_slot()
        self.sync.status = "Excel: your corrections were applied - photos re-selected"

    def on_close(self):
        self.commit()
        self.root.title("Syncing Excel...")
        self.root.update_idletasks()
        try:
            self.sync.flush()
        finally:
            self.root.destroy()

    # ---------- slot handling ----------
    def default_slot(self, key):
        iid, rank = key.split(":")
        picks = self.sel[iid]
        it = self.items[iid]
        return {"photo_id": picks[int(rank) - 1]["photo_id"], "rot": 0, "tilt": 0.0, "crop": None,
                "enhance": True, "aspect": "free", "status": "pending", "exported": False,
                "exported_enhanced": False, "final_path": None,
                "name": f"{iid}_{it['slug']}_{rank}"}

    def load_slot(self):
        self.key = self.keys[self.idx]
        self.slot = self.state.setdefault(self.key, self.default_slot(self.key))
        iid, rank = self.key.split(":")
        it = self.items[iid]
        self.title_var.set(f"{it['item_name']}   -   {it['brand']} / {it['category']}")
        self.name_var.set(self.slot["name"])
        self.enh_var.set(self.slot["enhance"])
        self.aspect = self.slot["aspect"]
        self._set_scale(self.slot["tilt"])
        self.load_photo(reset_crop=self.slot["crop"] is None)
        self.update_status()
        self.canvas.focus_set()

    def load_photo(self, reset_crop=False):
        pid = self.slot["photo_id"]
        if pid not in self.cache_base:
            if len(self.cache_base) > 12:
                self.cache_base.pop(next(iter(self.cache_base)))
            self.cache_base[pid] = Image.open(self.files[pid]).convert("RGB")
        self.base = self.cache_base[pid]
        self.disp = finalize.apply_geometry(self.base, self.slot["rot"], self.slot["tilt"])
        iw, ih = self.disp.size
        if reset_crop or not self.slot["crop"]:
            self.slot["crop"] = finalize.auto_crop(self.disp)
        x0, y0, x1, y1 = self.slot["crop"]
        self.c = [x0 * iw, y0 * ih, x1 * iw, y1 * ih]
        self.rebuild_view()

    def commit(self, save=True):
        iw, ih = self.disp.size
        self.slot["crop"] = [self.c[0] / iw, self.c[1] / ih, self.c[2] / iw, self.c[3] / ih]
        self.slot["enhance"] = bool(self.enh_var.get())
        self.slot["aspect"] = self.aspect
        name = re.sub(r"[^A-Za-z0-9_\-]+", "-", self.name_var.get().strip().removesuffix(".jpg")).strip("-")
        if name:
            self.slot["name"] = name
        if save:
            slotstore.save(self.key, self.slot)
            self.slot["exported"] = slotstore.is_exported(self.key)   # save() resets it when the decision changed
            self.sync.mark_dirty()                          # DB changed -> Excel follows automatically
        self.update_status()

    def update_status(self):
        st = self.slot["status"]
        counts = {s: sum(1 for v in self.state.values() if v.get("status") == s) for s in ("approved", "rejected")}
        mark = {"approved": "APPROVED", "rejected": "REJECTED", "pending": "pending"}[st]
        if st == "approved" and self.slot.get("exported"):
            mark += " - SAVED" + (" + ENHANCED" if self.slot.get("exported_enhanced") else " (not enhanced)")
        self.status_var.set(mark)
        iw, ih = self.disp.size
        scale = 2600 / max(iw, ih)       # crop size estimated at the export working resolution
        cw, ch = (self.c[2] - self.c[0]) * scale, (self.c[3] - self.c[1]) * scale
        warn = "   !! small crop (<1000 px)" if max(cw, ch) < 1000 else ""
        alt = [p["photo_id"] for p in self.sel[self.key.split(":")[0]]]
        self.info_var.set(
            f"Slot {self.idx + 1}/{len(self.keys)}   |   approved {counts['approved']}  rejected {counts['rejected']}"
            f"  pending {len(self.keys) - counts['approved'] - counts['rejected']}   |   photo "
            f"{self.slot['photo_id']} ({alt.index(self.slot['photo_id']) + 1 if self.slot['photo_id'] in alt else '?'}/{len(alt)})"
            f"   |   rot {self.slot['rot'] * 90}  tilt {self.slot['tilt']:+.1f}   aspect {self.aspect}"
            f"   |   crop ~{int(cw)}x{int(ch)} px{warn}")

    def go(self, i):
        self.commit()
        self.idx = max(0, min(len(self.keys) - 1, i))
        self.load_slot()

    def next(self): self.go(self.idx + 1)
    def prev(self): self.go(self.idx - 1)

    def approve(self):
        self.commit(save=False)
        # file names must be unique inside a brand/category folder
        iid = self.key.split(":")[0]
        it = self.items[iid]
        for k, v in self.state.items():
            o = self.items.get(k.split(":")[0])
            if o and k != self.key and v.get("status") == "approved" and v["name"] == self.slot["name"] \
                    and (o["brand"], o["category"]) == (it["brand"], it["category"]):
                messagebox.showwarning("Duplicate name", f"'{self.slot['name']}' is already used by slot {k}.")
                return
        self.slot["status"] = "approved"
        self.next()

    def reject(self):
        self.commit(save=False)
        self.slot["status"] = "rejected"
        self.next()

    def rotate(self, d):
        self.slot["rot"] = (self.slot["rot"] + d) % 4
        self.load_photo(reset_crop=True)
        self.commit()

    # ---------- tilt ----------
    def _set_scale(self, v):
        self._scale_busy = True
        self.scale.set(v)
        self._scale_busy = False

    def _on_scale(self, value):
        v = round(float(value), 1)
        if self._scale_busy or v == self.slot["tilt"]:        # programmatic set / late duplicate callback
            return
        self.slot["tilt"] = v
        if self._tilt_job is None:                           # throttle live re-render while dragging
            self._tilt_job = self.root.after(30, self._render_tilt)

    def _tilt_drag(self, on: bool):
        self._tilting = on
        if not on:
            self._render_tilt()
            self.commit()
        else:
            self.draw_overlay()

    def _render_tilt(self):
        self._tilt_job = None
        self.disp = finalize.apply_geometry(self.base, self.slot["rot"], self.slot["tilt"])
        self.rebuild_view()
        self.update_status()

    def set_tilt(self, v: float):
        v = round(max(-TILT_MAX, min(TILT_MAX, v)), 1)
        self.slot["tilt"] = v
        self._set_scale(v)
        self._render_tilt()
        self.commit()

    def tilt(self, d: float):
        self.set_tilt(self.slot["tilt"] + d)

    def auto_level(self):
        """Estimate the tilt from long straight edges (mat, box, table) and apply it."""
        upright = finalize.apply_geometry(self.base, self.slot["rot"], 0.0)
        a = enhance.estimate_tilt(cv2.cvtColor(np.asarray(upright), cv2.COLOR_RGB2BGR))
        if a is None or abs(a) > TILT_MAX:
            messagebox.showinfo("Auto-level", "No clear straight lines found - use the slider and the level grid.")
            return
        self.set_tilt(a)

    def alt(self, d):
        iid = self.key.split(":")[0]
        pids = [p["photo_id"] for p in self.sel[iid]]
        i = pids.index(self.slot["photo_id"]) if self.slot["photo_id"] in pids else 0
        self.slot["photo_id"] = pids[(i + d) % len(pids)]
        self.slot["rot"], self.slot["tilt"], self.slot["crop"] = 0, 0.0, None
        self._set_scale(0.0)
        self.load_photo(reset_crop=True)
        self.commit()

    def goto(self, _e=None):
        q = self.goto_var.get().strip().upper()
        for i, k in enumerate(self.keys):
            if k.startswith(q):
                self.go(i)
                return
        messagebox.showinfo("Go to", f"No item id starting with '{q}'")

    def preview(self):
        self.commit()
        img = finalize.render_image(self.base, self.slot, final_edge=1000)
        win = tk.Toplevel(self.root)
        win.title("Preview of final result (Esc to close)")
        tkimg = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=tkimg)
        lbl.image = tkimg
        lbl.pack()
        win.bind("<Escape>", lambda e: win.destroy())
        win.focus_set()

    def export(self):
        if self.exporting:
            return
        self.commit()
        self.exporting = True
        self.status_var.set("exporting...")

        def work():
            p = subprocess.run([sys.executable, "-m", "kitchen", "finalize"], capture_output=True, text=True,
                               cwd=str(config.ROOT / "app"))
            lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
            out = "\n".join(lines[-2:]) or p.stderr[-400:]
            self.root.after(0, lambda: self.after_export("Export finished", f"{out}\n\nFolder: {config.FINAL}"))
        threading.Thread(target=work, daemon=True).start()

    def after_export(self, title, text):
        self.exporting = False
        self.state = slotstore.load_all()                  # pick up exported / enhanced flags
        self.slot = self.state.get(self.key, self.slot)
        self.sync.mark_dirty()
        self.update_status()
        messagebox.showinfo(title, text)

    # ---------- view / crop geometry ----------
    def rebuild_view(self):
        if not hasattr(self, "disp"):
            return
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        iw, ih = self.disp.size
        self.s = min((cw - 20) / iw, (ch - 20) / ih)
        dw, dh = max(1, int(iw * self.s)), max(1, int(ih * self.s))
        self.ox, self.oy = (cw - dw) // 2, (ch - dh) // 2
        self.tkimg = ImageTk.PhotoImage(self.disp.resize((dw, dh), Image.BILINEAR))
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.tkimg)
        self.draw_overlay()

    def cv(self, x, y):
        return self.ox + x * self.s, self.oy + y * self.s

    def draw_overlay(self):
        if not hasattr(self, "disp"):
            return
        self.canvas.delete("ov")
        iw, ih = self.disp.size
        L, T = self.cv(0, 0)
        R, B = self.cv(iw, ih)
        if self.grid_var.get() or self._tilting:               # level grid: judge horizontals / verticals by eye
            for k in range(1, 12):
                x = L + (R - L) * k / 12
                self.canvas.create_line(x, T, x, B, fill="#ffffff" if k == 6 else "#9ad1ff",
                                        dash=() if k == 6 else (1, 5), tags="ov")
            for k in range(1, 9):
                y = T + (B - T) * k / 9
                self.canvas.create_line(L, y, R, y, fill="#ffffff" if k == 4 or k == 5 else "#9ad1ff",
                                        dash=() if k in (4, 5) else (1, 5), tags="ov")
        x0, y0 = self.cv(self.c[0], self.c[1])
        x1, y1 = self.cv(self.c[2], self.c[3])
        for box in ((L, T, R, y0), (L, y1, R, B), (L, y0, x0, y1), (x1, y0, R, y1)):
            self.canvas.create_rectangle(*box, fill="black", stipple="gray50", outline="", tags="ov")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#4ade80", width=2, tags="ov")
        for f in (1 / 3, 2 / 3):       # rule-of-thirds guides
            self.canvas.create_line(x0 + (x1 - x0) * f, y0, x0 + (x1 - x0) * f, y1, fill="#4ade80", dash=(2, 4), tags="ov")
            self.canvas.create_line(x0, y0 + (y1 - y0) * f, x1, y0 + (y1 - y0) * f, fill="#4ade80", dash=(2, 4), tags="ov")
        for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.canvas.create_rectangle(hx - 6, hy - 6, hx + 6, hy + 6, fill="#4ade80", outline="black", tags="ov")

    def to_img(self, ex, ey):
        iw, ih = self.disp.size
        return (min(max((ex - self.ox) / self.s, 0), iw), min(max((ey - self.oy) / self.s, 0), ih))

    def hit_corner(self, ex, ey):
        for cx, cy, ax, ay in ((self.c[0], self.c[1], self.c[2], self.c[3]), (self.c[2], self.c[1], self.c[0], self.c[3]),
                               (self.c[0], self.c[3], self.c[2], self.c[1]), (self.c[2], self.c[3], self.c[0], self.c[1])):
            px, py = self.cv(cx, cy)
            if abs(ex - px) <= HANDLE and abs(ey - py) <= HANDLE:
                return (ax, ay)
        return None

    def on_hover(self, e):
        self.canvas.config(cursor="sizing" if self.hit_corner(e.x, e.y) else
                           ("fleur" if self.inside(e.x, e.y) else "crosshair"))

    def inside(self, ex, ey):
        x, y = self.to_img(ex, ey)
        return self.c[0] <= x <= self.c[2] and self.c[1] <= y <= self.c[3]

    def on_press(self, e):
        self.canvas.focus_set()
        corner = self.hit_corner(e.x, e.y)
        x, y = self.to_img(e.x, e.y)
        if corner:
            self.mode, self.anchor = "resize", corner
        elif self.inside(e.x, e.y):
            self.mode, self.last = "move", (x, y)
        else:
            self.mode, self.anchor = "resize", (x, y)

    def on_drag(self, e):
        x, y = self.to_img(e.x, e.y)
        iw, ih = self.disp.size
        if self.mode == "move":
            dx, dy = x - self.last[0], y - self.last[1]
            w, h = self.c[2] - self.c[0], self.c[3] - self.c[1]
            nx = min(max(self.c[0] + dx, 0), iw - w)
            ny = min(max(self.c[1] + dy, 0), ih - h)
            self.c = [nx, ny, nx + w, ny + h]
            self.last = (x, y)
        else:
            ax, ay = self.anchor
            sx, sy = (1 if x >= ax else -1), (1 if y >= ay else -1)
            w, h = abs(x - ax), abs(y - ay)
            ar = ASPECTS[self.aspect]
            if ar:
                if w / ar > h:
                    w = h * ar
                else:
                    h = w / ar
            if w >= MIN_PX / self.s * 0.3 and h >= MIN_PX / self.s * 0.3:
                self.c = sorted_box(ax, ay, ax + sx * w, ay + sy * h)
        self.draw_overlay()
        self.update_status()

    def on_wheel(self, e):
        f = 0.94 if e.delta > 0 else 1.06            # wheel up = zoom in (smaller crop)
        iw, ih = self.disp.size
        cx, cy = (self.c[0] + self.c[2]) / 2, (self.c[1] + self.c[3]) / 2
        w, h = (self.c[2] - self.c[0]) * f, (self.c[3] - self.c[1]) * f
        lim = min(2 * cx / w, 2 * (iw - cx) / w, 2 * cy / h, 2 * (ih - cy) / h, 1.0)   # keep inside image
        w, h = w * lim, h * lim
        if w * self.s < 40:
            return
        self.c = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        self.draw_overlay()
        self.update_status()
        self.commit()

    def cycle_aspect(self):
        names = list(ASPECTS)
        self.aspect = names[(names.index(self.aspect) + 1) % len(names)]
        ar = ASPECTS[self.aspect]
        if ar:
            iw, ih = self.disp.size
            cx, cy = (self.c[0] + self.c[2]) / 2, (self.c[1] + self.c[3]) / 2
            area = (self.c[2] - self.c[0]) * (self.c[3] - self.c[1])
            w = (area * ar) ** 0.5
            h = w / ar
            k = min(1.0, iw / w, ih / h)
            w, h = w * k, h * k
            x0 = min(max(cx - w / 2, 0), iw - w)
            y0 = min(max(cy - h / 2, 0), ih - h)
            self.c = [x0, y0, x0 + w, y0 + h]
        self.draw_overlay()
        self.commit()

    # ---------- keyboard ----------
    def on_key(self, e):
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        k = e.keysym
        if k in ("Return", "space"): self.approve()
        elif k == "Right": self.next()
        elif k == "Left": self.prev()
        elif k in ("x", "X", "Delete"): self.reject()
        elif k == "r": self.rotate(1)
        elif k == "R": self.rotate(-1)
        elif k == "comma": self.tilt(0.5)
        elif k == "period": self.tilt(-0.5)
        elif k == "less": self.tilt(0.1)
        elif k == "greater": self.tilt(-0.1)
        elif k == "0": self.set_tilt(0.0)
        elif k in ("l", "L"): self.auto_level()
        elif k in ("g", "G"): self.grid_var.set(not self.grid_var.get()); self.draw_overlay()
        elif k in ("e", "E"): self.enh_var.set(not self.enh_var.get()); self.commit()
        elif k in ("a", "A"): self.cycle_aspect()
        elif k == "bracketright": self.alt(1)
        elif k == "bracketleft": self.alt(-1)
        elif k in ("n", "N"): self.name_entry.focus_set(); self.name_entry.select_range(0, "end")
        elif k in ("p", "P"): self.preview()


def sorted_box(x0, y0, x1, y1):
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
