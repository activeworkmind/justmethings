"""Stage 5: static HTML contact sheet so labels can be checked by eye in a few minutes.
To correct a pick, add `item_id,photo_id` lines to work/overrides.csv and re-run `select`."""
import html
import json
import os

from . import catalog, config


def run() -> str:
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    config.REVIEW.mkdir(parents=True, exist_ok=True)
    parts = ["<meta charset=utf-8><title>Review</title><style>body{font:14px sans-serif;margin:16px}"
             "h2{margin:24px 0 4px}.row{display:flex;gap:8px;flex-wrap:wrap}figure{margin:0;width:200px}"
             "img{width:200px;height:200px;object-fit:cover;border-radius:6px}"
             ".top img{outline:3px solid #2a9d4b}figcaption{font-size:11px;color:#555}</style>"
             "<h1>Top 3 per menu item (green) + alternates</h1>"]
    for it in catalog.load():
        parts.append(f"<h2>{it['item_id']} - {html.escape(it['item_name'])} "
                     f"<small>({it['brand']})</small></h2><div class=row>")
        for n, p in enumerate(sel[it["item_id"]]):
            rel = os.path.relpath(p["file"], config.REVIEW).replace("\\", "/")
            parts.append(f"<figure class='{'top' if n < config.TOP_N else ''}'><img loading=lazy src='{rel}'>"
                         f"<figcaption>{p['photo_id']} · {p['score']}</figcaption></figure>")
        parts.append("</div>")
    out = config.REVIEW / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out)
