"""Food Photo Studio - Streamlit UI.   Launch:  start_studio.bat   (or: streamlit run studio/app.py from the app folder)

Studio tab : load photos -> brand theme -> click enhancements (any order) or the 1-click brand preset ->
             undo / redo / reset / before-after compare -> save an upload-ready file.
Batch tab  : run the recorded recipe on a whole folder and export Swiggy / Zomato upload-ready images."""
import base64
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # make `studio` / `kitchen` importable
from studio import batch as B                                         # noqa: E402
from studio import image_engine as E                                  # noqa: E402
from studio.brands import BRANDS, DIET_LABELS                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "studio_recipes"
SOURCES = [("final", ROOT / "final"), ("ready_for_crop", ROOT / "ready_for_crop")]

st.set_page_config(page_title="Food Photo Studio", layout="wide")
ss = st.session_state
ss.setdefault("sessions", {})
ss.setdefault("active", None)
ss.setdefault("seen_uploads", set())
ss.setdefault("batch_report", None)


# ------------------------------------------------------------------ helpers
def show(img, **kw):
    st.image(img, width="stretch", **kw)


def compare_slider(before: Image.Image, after: Image.Image, width: int = 880):
    """Self-contained before/after slider (no extra packages): two stacked images, the top one clipped by a range input."""
    def b64(im):
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()
    h = int(width * after.height / after.width)
    html = f"""
    <div style="max-width:{width}px;margin:auto;font-family:sans-serif">
      <div style="position:relative;width:100%;line-height:0">
        <img src="data:image/jpeg;base64,{b64(after)}" style="width:100%;display:block;border-radius:6px">
        <img id="b" src="data:image/jpeg;base64,{b64(before)}" style="position:absolute;inset:0;width:100%;height:100%;clip-path:inset(0 50% 0 0);border-radius:6px">
        <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#fff;box-shadow:0 0 6px #000"></div>
        <span style="position:absolute;left:10px;top:10px;background:#000a;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;line-height:1.4">BEFORE</span>
        <span style="position:absolute;right:10px;top:10px;background:#000a;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;line-height:1.4">AFTER</span>
      </div>
      <input type="range" min="0" max="100" value="50" style="width:100%;margin-top:8px"
             oninput="document.getElementById('b').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';document.getElementById('bar').style.left=this.value+'%'">
    </div>"""
    html = html.strip()                                              # built only from the user's own images: trusted
    if hasattr(st, "iframe"):                                        # components.html is removed after 2026-06-01
        st.iframe(html, height=h + 70)
    else:
        components.html(html, height=h + 70)


def add_session(name: str, img: Image.Image):
    base, n = name, 1
    while name in ss.sessions:
        n += 1
        name = f"{base} ({n})"
    with st.spinner(f"Loading {name}..."):
        ss.sessions[name] = E.ImageSession(name, img, ss.brand, ss.ratio, ss.preview)
    ss.active = name


def remove_active():
    """Callback (runs before the next script run, so widget state may change)."""
    ss.sessions.pop(ss.active, None)
    ss.active = next(iter(ss.sessions), None)


def guard(fn, spinner="Working..."):
    try:
        with st.spinner(spinner):
            return fn()
    except Exception as e:                                           # show, never crash the app
        st.error(f"{type(e).__name__}: {e}")
        return None


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar: global settings
with st.sidebar:
    st.header("Food Photo Studio")
    st.radio("Brand theme", list(BRANDS), format_func=lambda k: BRANDS[k].name, key="brand")
    brand = BRANDS[ss.brand]
    st.caption(f"{brand.vibe}")
    st.selectbox("Canvas ratio", list(E.RATIOS), index=list(E.RATIOS).index("1:1"), key="ratio",
                 help="Swiggy / Zomato item photos are normally square (1:1) - confirm in your partner app.")
    st.selectbox("Dish isolation model", E.ISOLATE_MODELS, index=E.ISOLATE_MODELS.index("isnet-general-use"), key="model",
                 help="isnet-general-use: ~2 s per photo and matches BiRefNet's accuracy on these dishes. "
                      "birefnet-general: best edges but ~20 s per photo on CPU. u2net: fastest, over-selects the table mat.")
    st.selectbox("Background source", ["auto", "procedural", "hf", "local", "neutral", "transparent"], key="bgsrc",
                 help="auto = free HuggingFace AI background if a token is set, else local templates, else the procedural "
                      "studio backdrop (works offline). transparent = PNG cut-out.")
    st.text_input("HuggingFace token (free, optional)", type="password", key="hf_token",
                  value=os.environ.get("HF_TOKEN", ""),
                  help="Only the text prompt is sent to generate backgrounds - never your photos.")
    st.selectbox("AI background model", E.backgrounds.HF_MODELS, key="hf_model")
    st.checkbox("Allow IC-Light neural relighting (uploads the dish photo)", key="iclight",
                help="Sends the dish cut-out to the public lllyasviel/IC-Light HuggingFace Space. Off = fully local.")
    st.slider("Preview size (px)", 700, 1600, 1100, 100, key="preview")

E.CONFIG.isolate_model = ss.model
E.CONFIG.hf_token = ss.hf_token.strip()
E.CONFIG.hf_model = ss.hf_model
E.CONFIG.allow_remote_relight = ss.iclight

tab_studio, tab_batch = st.tabs(["Studio", "Batch / Export"])

# ====================================================================== BATCH
with tab_batch:
    st.subheader("Run a recipe on a whole folder")
    b1, b2 = st.columns([1.3, 1])
    with b1:
        default_in = str(ROOT / "final") if (ROOT / "final").exists() else str(ROOT / "ready_for_crop")
        in_dir = st.text_input("Input folder (raw / reviewed food photos)", default_in)
        recursive = st.checkbox("Include sub-folders", True)
        skip = st.checkbox("Skip files already exported", True)
        src_choice = st.radio("Recipe", ["Current photo's recorded recipe", "Brand master preset", "Recipe file (.json)"], horizontal=True)
        recipe_file = st.file_uploader("Recipe .json", type=["json"]) if src_choice.startswith("Recipe file") else None
    with b2:
        bprof = st.selectbox("Export profile", list(E.PROFILES), format_func=lambda k: E.PROFILES[k].name, key="bprof")
        bbrand = st.selectbox("Brand theme for this batch", list(BRANDS), index=list(BRANDS).index(ss.brand),
                              format_func=lambda k: BRANDS[k].name, key="bbrand")
        st.caption(f"Output: {B.DEFAULT_OUT / BRANDS[bbrand].slug}")
        st.info("Upload checklist: " + ("square JPEG, <= 1 MB, clean frame (no logo / badge / watermark)."
                                       if not E.PROFILES[bprof].allow_overlays else "high-resolution file for your own use."))
        if not E.PROFILES[bprof].allow_overlays:
            st.caption("Badge / border steps are stripped automatically for this profile. "
                       "Check the current Swiggy / Zomato image rules in your partner app.")

    if st.button("Run recipe on folder", type="primary"):
        folder = Path(in_dir)
        if not folder.exists():
            st.error("Input folder not found.")
        else:
            if src_choice.startswith("Brand"):
                rec = E.master_preset(bbrand, "1:1", False, ss.model)
            elif src_choice.startswith("Recipe file"):
                rec = E.EnhancementRecipe.from_json(recipe_file.getvalue().decode("utf-8")) if recipe_file else None
            else:
                s = ss.sessions.get(ss.active)
                rec = s.recipe(ss.model) if s else None
            if rec is None or not rec.steps:
                st.error("There is no recipe to run - enhance a photo first, or choose the brand master preset.")
            else:
                bar = st.progress(0.0, text="Starting...")
                rep = B.run_recipe_on_folder(rec, folder, None, bprof, bbrand, recursive, skip,
                                             progress=lambda i, n, name: bar.progress(i / n, text=f"{i}/{n}  {name}"))
                bar.progress(1.0, text="Done")
                ss.batch_report = rep
    rep = ss.batch_report
    if rep:
        (st.success if not rep["failed"] else st.warning)(
            f"Exported {rep['ok']}, skipped {rep['skipped']}, failed {rep['failed']}  ->  {rep['out_dir']}")
        if rep["files"]:
            big = [f for f in rep["files"] if E.PROFILES[bprof].max_bytes and f["bytes"] > E.PROFILES[bprof].max_bytes]
            st.caption(f"Largest file: {max(f['bytes'] for f in rep['files']) / 1000:.0f} KB" + (f"  -  {len(big)} over the limit!" if big else ""))
        for name, err in rep["errors"]:
            st.error(f"{name}: {err}")


# ====================================================================== STUDIO
with tab_studio:
    left, mid, right = st.columns([1.1, 2.6, 1.7], gap="medium")

    # ---------------- left: photos
    with left:
        st.subheader("1. Photos")
        up = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
        for f in up or []:
            if (f.name, f.size) not in ss.seen_uploads:
                ss.seen_uploads.add((f.name, f.size))
                add_session(Path(f.name).stem, Image.open(f))
        src_name = st.selectbox("...or load from project folder", [n for n, p in SOURCES if p.exists()] or ["(none)"])
        folder = dict(SOURCES).get(src_name)
        if folder:
            files = B.list_images(folder, recursive=True)
            pick = st.multiselect(f"{len(files)} photos in {src_name}/", [str(p.relative_to(folder)) for p in files], max_selections=12)
            if st.button("Load selected", disabled=not pick, width="stretch"):
                for rel in pick:
                    guard(lambda rel=rel: add_session(Path(rel).stem, B.load_photo(folder / rel)))
        if ss.sessions:
            st.selectbox("Active photo", list(ss.sessions), key="active")
            st.button("Remove this photo", width="stretch", on_click=remove_active)

    sess = ss.sessions.get(ss.active) if ss.active else None
    if sess is None:
        with mid:
            st.info("Upload photos or load some from the project folder to begin. "
                    "Tip: the photos in **final/** are already cropped and approved.")
        st.stop()

    # brand / ratio changes are undoable actions on the active photo
    if sess.scene.brand != ss.brand:
        guard(lambda: sess.set_brand(ss.brand))
    if sess.scene.ratio != ss.ratio:
        guard(lambda: sess.set_ratio(ss.ratio))
    sess.preview_edge = ss.preview

    # ---------------- middle: image + history toolbar (rendered after actions below, into this container)
    with mid:
        st.subheader("2. Preview")
        t1, t2, t3, t4 = st.columns([1, 1, 1, 1.4])
        # every action ends with st.rerun() so the toolbar (drawn above the action) shows the fresh undo / redo state
        if t1.button("Undo", disabled=not sess.can_undo, width="stretch"):
            sess.undo()
            st.rerun()
        if t2.button("Redo", disabled=not sess.can_redo, width="stretch"):
            sess.redo()
            st.rerun()
        if t3.button("Reset to original", width="stretch"):
            if guard(sess.reset) is not None:
                st.rerun()
        view = t4.radio("View", ["After", "Compare"], horizontal=True, label_visibility="collapsed")
        img_area = st.container()
        warn_area = st.container()

    # ---------------- right: Enhance with AI
    with right:
        st.subheader("3. Enhance with AI")
        incl_badge = st.checkbox("Include brand badge in the 1-click preset", value=False,
                                 help="Swiggy / Zomato usually reject logos and overlays on item photos - off by default.")
        if st.button(f"1-Click {brand.name} Master Preset", type="primary", width="stretch"):
            steps = E.master_preset(ss.brand, ss.ratio, incl_badge, ss.model).steps
            steps = [dict(s, params=dict(s["params"], **({"source": ss.bgsrc} if s["op"] == "background" and ss.bgsrc != "auto" else {})))
                     for s in steps]
            if guard(lambda: sess.apply_steps(steps, f"{brand.name} master preset"), "Applying preset...") is not None:
                st.rerun()

        def tool(title, op, key, body=None, btn=None):
            """One enhancement: an expander with its sliders and an apply button."""
            with st.expander(title):
                params = body() if body else {}
                if st.button(btn or "Apply", key=f"btn_{key}", width="stretch"):
                    if guard(lambda: sess.apply(op, **params), f"{title}...") is not None:
                        st.rerun()

        tool("1. Clean dish isolation", "isolate", "iso",
             lambda: {"model": ss.model}, "Isolate the dish")
        tool("2. AI theme background", "background", "bg",
             lambda: {"source": ss.bgsrc, "seed": st.number_input("Variation seed", 0, 999, 0, key="bgseed")}, "Apply brand background")
        tool("3. Grounded dual shadows", "shadows", "sh",
             lambda: {"contact_opacity": st.slider("Contact shadow", 0.0, 1.0, 0.75, 0.05, key="s1"),
                      "ambient_opacity": st.slider("Ambient shadow", 0.0, 1.0, 0.35, 0.05, key="s2"),
                      "softness": st.slider("Softness", 0.5, 2.0, 1.0, 0.1, key="s3")}, "Ground the dish")

        def sun():
            ang = st.selectbox("Light from", ["Brand default", "Upper-left (135)", "Left (180)", "Upper-right (45)", "Top (90)"], key="sa")
            val = {"Brand default": None, "Upper-left (135)": 135.0, "Left (180)": 180.0, "Upper-right (45)": 45.0, "Top (90)": 90.0}[ang]
            return {"strength": st.slider("Strength", 0.0, 1.0, 0.45, 0.05, key="su1"), "angle": val,
                    "blinds": st.slider("Window-blind falloff", 0.0, 1.0, 0.35, 0.05, key="su2"),
                    "use_iclight": st.checkbox("Neural relight (IC-Light)", key="su3")}
        tool("4. Natural sunlight touch", "sunlight", "sun", sun, "Add sunlight")
        tool("5. Shahi moody rim & candlelight", "rim_glow", "rim",
             lambda: {"intensity": st.slider("Intensity", 0.0, 1.0, 0.7, 0.05, key="r1")}, "Add rim glow")
        tool("6. Angle, tilt & framing fix", "framing", "fr",
             lambda: {"fill": st.slider("Dish width (% of canvas)", 0.60, 0.90, 0.76, 0.01, key="f1"),
                      "y_anchor": st.slider("Vertical position", 0.45, 0.65, 0.56, 0.01, key="f2"),
                      "auto_tilt": st.checkbox("Auto-correct tilt", True, key="f3"),
                      "tilt": st.slider("Extra tilt (deg clockwise)", -10.0, 10.0, 0.0, 0.5, key="f4")}, "Fix framing")
        tool("7. Ghee, gravy & masala pop", "lab_pop", "lab",
             lambda: {"strength": st.slider("Strength", 0.0, 1.5, 1.0, 0.1, key="l1")}, "Add pop")
        tool("8. Fresh out-of-dum steam", "steam", "st",
             lambda: {"opacity": st.slider("Steam opacity", 0.0, 1.0, 0.4, 0.05, key="st1"),
                      "seed": st.number_input("Steam seed", 0, 99, 3, key="st2")}, "Add steam")
        tool("9. DSLR bokeh depth of field", "bokeh", "bk",
             lambda: {"strength": st.slider("Blur strength", 0.0, 1.5, 0.6, 0.05, key="b1")}, "Add bokeh")

        def badge():
            corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
            diets = list(brand.diets)
            return {"corner": st.selectbox("Brand badge corner", corners, 2, key="bd1"),
                    "diet_corner": st.selectbox("Dietary icon corner", corners, 1, key="bd2"),
                    "diet": st.selectbox("Dietary indicator", diets, diets.index(brand.default_diet),
                                         format_func=lambda d: DIET_LABELS[d], key="bd3"),
                    "show_brand": st.checkbox("Brand badge", True, key="bd4"),
                    "show_diet": st.checkbox("Dietary icon", True, key="bd5"),
                    "border": st.checkbox("Frame border", True, key="bd6")}
        tool("10. Brand badge & dietary icon", "badge", "bd", badge, "Add badge")
        tool("+ Brand colour grade", "brand_grade", "gr",
             lambda: {"strength": st.slider("Grade strength", 0.0, 1.5, 1.0, 0.1, key="g1")}, "Apply grade")
        if "Sunday" in brand.name:
            st.caption("Dietary icons: " + " | ".join(DIET_LABELS[d] for d in brand.diets))

    # ---------------- now render (after any action above ran)
    with img_area:
        if view == "Compare":
            before = E.render(replace(sess.initial, brand=sess.scene.brand, ratio=sess.scene.ratio), ss.preview)
            compare_slider(before, sess.current)
        else:
            show(sess.current)
        st.caption(f"{sess.name}  |  {sess.current.width}x{sess.current.height}px preview  |  "
                   f"matte cached: {'yes' if sess.alpha_matte is not None else 'not yet'}")
    with warn_area:
        for w in sess.warnings:
            st.warning(w)

    # ---------------- history + recipe + export
    st.divider()
    h1, h2, h3 = st.columns([1.2, 1.6, 1.6], gap="large")
    with h1:
        st.subheader("History")
        applied = [e.label for e in sess.undo_stack]
        if applied:
            for i, lab in enumerate(applied, 1):
                st.write(f"{i}. {lab}")
        else:
            st.caption("Nothing applied yet.")
        if sess.redo_stack:
            st.caption("Redo available: " + ", ".join(e.label for e in reversed(sess.redo_stack)))
    with h2:
        st.subheader("Recorded recipe")
        recipe = sess.recipe(ss.model)
        st.caption("Every enhancement you click is recorded. Run it on a whole folder in the Batch tab.")
        st.code(recipe.to_json(), language="json", height=220)
        c1, c2 = st.columns(2)
        c1.download_button("Download recipe", recipe.to_json(), f"{ss.brand}_recipe.json", "application/json", width="stretch")
        if c2.button("Save to studio_recipes/", width="stretch"):
            RECIPES.mkdir(exist_ok=True)
            p = RECIPES / f"{ss.brand}_{len(list(RECIPES.glob('*.json'))) + 1}.json"
            p.write_text(recipe.to_json(), encoding="utf-8")
            st.success(f"Saved {p.name}")
    with h3:
        st.subheader("Save upload-ready file")
        prof_key = st.selectbox("Export profile", list(E.PROFILES), format_func=lambda k: E.PROFILES[k].name, key="prof")
        prof = E.PROFILES[prof_key]
        st.caption(f"{prof.fmt}, {prof.ratio}, long edge {prof.long_edge}px" +
                   (f", at most {prof.max_bytes // 1000} KB" if prof.max_bytes else "") +
                   (", no badge / border overlays" if not prof.allow_overlays else ""))
        if st.button("Export this photo", type="primary", width="stretch"):
            def export():
                scene = replace(sess.scene, ratio=prof.ratio, badge=sess.scene.badge if prof.allow_overlays else None)
                img = E.render(scene, prof.long_edge)
                out = Path(B.DEFAULT_OUT) / BRANDS[sess.scene.brand].slug / (sess.name.split(" (")[0] + (".png" if prof.fmt == "PNG" else ".jpg"))
                info = E.save_image(img, out, prof)
                m = B.ITEM_ID.match(out.stem)
                B.record_upload(info, m.group(1) if m else None, sess.scene.brand, sess.recipe(ss.model).to_json())
                B.refresh_tracker()                                   # Excel: 'Upload Ready' + 'Ready to upload'
                return info, out
            res = guard(export, "Rendering at full resolution...")
            if res:
                info, out = res
                ok = prof.max_bytes is None or info["bytes"] <= prof.max_bytes
                (st.success if ok else st.error)(f"{'Saved' if ok else 'TOO LARGE'}: {out}  -  {info['width']}x{info['height']}px, {info['bytes'] / 1000:.0f} KB")
                st.download_button("Download file", out.read_bytes(), out.name, width="stretch")
