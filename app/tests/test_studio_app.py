"""Smoke test of the Streamlit studio through Streamlit's own test harness (no ML model, no network)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from studio import image_engine as E  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "studio" / "app.py")


def photo():
    rng = np.random.default_rng(4)
    return Image.fromarray(rng.integers(40, 220, (300, 400, 3), dtype=np.uint8))


def button(at, text):
    return next(b for b in at.button if text in (b.label or ""))


def test_app_starts_without_photos():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Studio", "Batch / Export"]


def test_undo_redo_and_compare_through_the_ui():
    at = AppTest.from_file(APP, default_timeout=120)
    sess = E.ImageSession("demo", photo(), "biryani_rules", "1:1", 300)       # brand matches the sidebar default
    at.session_state["sessions"] = {"demo": sess}
    at.session_state["active"] = "demo"
    at.run()
    assert not at.exception

    button(at, "Add pop").click().run()                                         # step 7: LAB colour pop (needs no matte)
    assert sess.scene.lab is not None and [e.label for e in sess.undo_stack][-1] == "Ghee, gravy & masala pop"
    button(at, "Undo").click().run()
    assert sess.scene.lab is None and sess.can_redo
    button(at, "Redo").click().run()
    assert sess.scene.lab is not None

    next(r for r in at.radio if "Compare" in r.options).set_value("Compare").run()
    assert not at.exception                                                     # before/after slider renders
    button(at, "Reset to original").click().run()
    assert sess.steps == [] and not at.exception
