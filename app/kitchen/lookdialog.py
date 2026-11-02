"""'Look...' dialog of the review app: tick the studio enhancements for ONE photo, preview, OK.

Every enhancement is a checkbox, so nothing is destructive: untick to remove it, 'Clear' to go back to the plain
photo. The original file and your crop are never touched. The brand comes from the item (BR- / SB- / DM-)."""
import threading
import tkinter as tk

from PIL import ImageTk

BG = "#1b1b1b"
CHOICES = [
    ("background", "Brand table background"),
    ("shadows", "Grounded shadows (no 'floating sticker')"),
    ("sunlight", "Natural sunlight"),
    ("rim_glow", "Rim light & candlelight glow"),
    ("framing", "Straighten & frame the dish"),
    ("lab_pop", "Masala / gravy colour pop"),
    ("steam", "Fresh steam"),
    ("bokeh", "Background blur (DSLR look)"),
    ("brand_grade", "Brand colour grade"),
    ("badge", "Brand badge + veg / non-veg icon"),
]


class LookDialog(tk.Toplevel):
    """result = list of steps ({"op", "params"}) after OK, or None after Cancel."""

    def __init__(self, parent, brand_name, brand_key, current_steps, make_preview):
        super().__init__(parent)
        from studio import image_engine as studio
        self._studio, self._brand_key, self._make_preview = studio, brand_key, make_preview
        self._defaults = {s["op"]: s["params"] for s in studio.preset_steps(brand_key, include_badge=True)}
        self.result = None
        self._busy = False
        self.title(f"Look - {brand_name}")
        self.configure(bg=BG)
        self.transient(parent)
        self.grab_set()

        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="y", padx=12, pady=12)
        tk.Label(left, text=f"{brand_name} look", bg=BG, fg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(left, text="Tick what you want. Preview any time.", bg=BG, fg="#aaa").pack(anchor="w", pady=(0, 8))
        have = {s["op"] for s in current_steps}
        self.vars = {}
        for op, text in CHOICES:
            self.vars[op] = tk.BooleanVar(value=op in have)
            tk.Checkbutton(left, text=text, variable=self.vars[op], bg=BG, fg="white", selectcolor="#333",
                           activebackground=BG, activeforeground="white", anchor="w").pack(fill="x")
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x", pady=(12, 4))
        tk.Button(row, text="Brand preset", command=self.preset).pack(side="left", padx=2)
        tk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=2)
        self.status = tk.StringVar(value="")
        tk.Label(left, textvariable=self.status, bg=BG, fg="#8ab4f8", wraplength=250, justify="left").pack(anchor="w", pady=6)
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x", side="bottom")
        tk.Button(row, text="Preview", command=self.preview, padx=10).pack(side="left", padx=2)
        tk.Button(row, text="OK", command=self.ok, padx=14).pack(side="left", padx=2)
        tk.Button(row, text="Cancel", command=self.destroy, padx=10).pack(side="left", padx=2)

        self.image_label = tk.Label(self, bg="#2b2b2b", width=64, height=32)
        self.image_label.pack(side="left", padx=(0, 12), pady=12)
        self.bind("<Escape>", lambda e: self.destroy())

    # ---- selection <-> steps
    def preset(self):
        have = {s["op"] for s in self._studio.preset_steps(self._brand_key)}
        for op, var in self.vars.items():
            var.set(op in have)

    def clear(self):
        for var in self.vars.values():
            var.set(False)

    def steps(self) -> list:
        chosen = [op for op, _ in CHOICES if self.vars[op].get()]
        out = [{"op": "isolate", "params": {}}] if any(op in self._studio.MASK_OPS for op in chosen) else []
        return out + [{"op": op, "params": dict(self._defaults.get(op, {}))} for op in chosen]

    def ok(self):
        self.result = self.steps()
        self.destroy()

    # ---- preview (runs in a thread so the window stays alive; the first one loads the model)
    def preview(self):
        if self._busy:
            return
        steps = self.steps()
        if not steps:
            self.status.set("Nothing ticked - the photo stays as cropped.")
        self._busy = True
        self.status.set("Working... (the first preview loads the AI model, ~10 s)")

        def work():
            try:
                img = self._make_preview(steps)
                err = None
            except Exception as e:
                img, err = None, f"{type(e).__name__}: {e}"
            self.after(0, lambda: self._show(img, err))
        threading.Thread(target=work, daemon=True).start()

    def _show(self, img, err):
        self._busy = False
        if not self.winfo_exists():
            return
        if err:
            self.status.set("Preview failed: " + err[:160])
            return
        img = img.copy()
        img.thumbnail((520, 520))
        self._tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._tk, width=img.width, height=img.height)
        self.status.set("Preview ready.")
