"""Excel tracker, connected to the app through the SQLite database.

  run()    rebuilds Image_Tracker.xlsx from the DB (menu availability, selected photos, review / saved / enhanced
           status). Your unapplied corrections in 'Photo Groups' are carried over, never lost.
  apply()  reads your corrections from 'Photo Groups' into work/photo_groups.csv (then run `select`).

Sheets: Summary | Menu Items | Photo List | Photo Groups | Lists"""
import json
import os

from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image

from . import catalog, config, db, groups

XLSX = config.ROOT / "Image_Tracker.xlsx"
THUMBS = config.WORK / "xl_thumbs"
F = "Arial"
HEAD = PatternFill("solid", fgColor="1F3A5F")
INPUT = PatternFill("solid", fgColor="FFF2CC")          # cells you are meant to edit
CONF = {"H": "High", "M": "Medium", "L": "Low - please check", "C": "Confirmed"}
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _head(ws, labels, widths):
    for c, (lab, w) in enumerate(zip(labels, widths), 1):
        cell = ws.cell(1, c, lab)
        cell.font, cell.fill, cell.border = Font(name=F, bold=True, color="FFFFFF", size=10), HEAD, BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"


def _body(ws, r0, r1, c1, c2, wrap=False):
    for row in ws.iter_rows(min_row=r0, max_row=r1, min_col=1, max_col=c2):
        for cell in row:
            cell.font = cell.font if cell.font.bold else Font(name=F, size=10)
            cell.border = BORDER
            cell.alignment = Alignment(vertical="center", wrap_text=wrap)


def _corrections(path) -> dict:
    """group_id -> {'items': [ids], 'confirm': bool}  for rows where the user typed something."""
    out = {}
    if not path.exists():
        return out
    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return out
    try:                                              # always release the file: an open handle blocks the swap on Windows
        for row in wb["Photo Groups"].iter_rows(min_row=2, values_only=True):
            if not row or not row[0]:
                continue
            ids = [str(v).split(" - ")[0].strip() for v in row[11:14] if v]
            confirm = str(row[14] or "").strip().upper() == "Y"
            if ids or confirm:
                out[str(row[0])] = {"items": ids, "confirm": confirm, "raw": list(row[11:15])}
    except Exception:
        return out
    finally:
        wb.close()
    return out


def _sig(raw) -> str:
    return "|".join(str(v or "") for v in raw)


def pending_corrections(path=XLSX) -> dict:
    """Corrections typed in the workbook that have not been applied yet."""
    con = db.connect()
    return {gid: c for gid, c in _corrections(path).items()
            if db.meta_get(con, f"applied:{gid}") != _sig(c["raw"])}


def apply() -> dict:
    corr = pending_corrections()
    rows = groups.load()
    con = db.connect()
    changed = 0
    for g in rows:
        c = corr.get(g["group_id"])
        if not c:
            continue
        if c["items"]:
            g["items"] = c["items"]
        g["confirmed"] = "Y"
        g["confidence"] = "C"
        db.meta_set(con, f"applied:{g['group_id']}", _sig(c["raw"]))
        changed += 1
    if changed:
        groups.save(rows)
    con.commit()
    return {"groups_updated": changed}


def _thumbs(g, index, down):
    THUMBS.mkdir(parents=True, exist_ok=True)
    n = [g["n_from"], (g["n_from"] + g["n_to"]) // 2, g["n_to"]]
    out = []
    for k, num in enumerate(dict.fromkeys(n)):
        p = THUMBS / f"{g['group_id']}_{k}.jpg"
        if not p.exists():                                  # atomic write: another process may be reading it
            im = Image.open(down[index[num]]).convert("RGB")
            im.thumbnail((150, 112))
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.jpg")
            im.save(tmp, quality=80)
            os.replace(tmp, p)
        out.append(p)
    return out


def run() -> dict:
    con = db.connect()
    items = catalog.load()
    names = {it["item_id"]: it["item_name"] for it in items}
    sel = json.loads(config.SELECTION.read_text(encoding="utf-8"))
    slots = {r["slot_key"]: r for r in con.execute("SELECT * FROM slots")}
    src = {r["id"]: r["path"] for r in con.execute("SELECT id,path FROM files")}
    down = {r["id"]: r["down_path"] for r in con.execute("SELECT id,down_path FROM files")}
    grps = groups.load()
    index = groups.photo_index()
    carried = pending_corrections()          # applied corrections are consumed; only unapplied ones are kept

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Summary"
    ws_menu = wb.create_sheet("Menu Items")
    ws_pl = wb.create_sheet("Photo List")
    ws_gr = wb.create_sheet("Photo Groups")
    ws_li = wb.create_sheet("Lists")

    # ---------- Photo List (app-managed: selected photos + review / saved / enhanced status) ----------
    _head(ws_pl, ["Item ID", "Item name", "Rank", "File name", "Photo group", "Match confidence", "Review status",
                  "Saved to final/", "Enhanced & saved", "Final file", "Key", "Original camera file"],
          [10, 34, 6, 40, 10, 18, 14, 14, 16, 58, 10, 24])
    r = 2
    for it in items:
        for p in (q for q in sel.get(it["item_id"], []) if q.get("rank")):
            rank = p["rank"]
            s = slots.get(f"{it['item_id']}:{rank}")
            if s and s["status"] == "deleted":                       # deleted photos leave the list
                continue
            pid = s["photo_id"] if s else p["photo_id"]
            pick = next((q for q in sel[it["item_id"]] if q["photo_id"] == pid), p)
            status = {"approved": "Approved", "rejected": "Rejected", "pending": "Pending"}[s["status"]] if s else "Pending"
            saved = bool(s and s["exported"])
            ws_pl.append([it["item_id"], it["item_name"], rank,
                          (s["name"] if s and s["name"] else f"{it['item_id']}_{it['slug']}_{rank}") + ".jpg",
                          pick.get("group", ""), CONF.get(pick.get("confidence", ""), ""), status,
                          "Yes" if saved else "No",
                          ("Yes" if s["exported_enhanced"] else "No") if saved else "-",
                          os.path.relpath(s["final_path"], config.ROOT) if saved and s["final_path"] else "",
                          f"=A{r}&\"|\"&C{r}", os.path.basename(src[pid])])
            r += 1
    _body(ws_pl, 2, max(r - 1, 2), 1, 12)
    ws_pl.auto_filter.ref = f"A1:L{max(r - 1, 2)}"
    for rng, rules in (("G2:G%d" % r, [("Approved", "C6EFCE"), ("Rejected", "FFC7CE")]),
                       ("H2:H%d" % r, [("Yes", "C6EFCE")]), ("I2:I%d" % r, [("Yes", "C6EFCE")])):
        for text, color in rules:
            ws_pl.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{text}"'],
                                                             fill=PatternFill("solid", bgColor=color)))

    # ---------- Menu Items ----------
    _head(ws_menu, ["Item ID", "Brand", "Category", "Item name", "Diet", "Price (INR)", "Availability",
                    "Photos selected", "Photo available?", "Best match", "Approved photos", "Saved to final/",
                    "Enhanced & saved", "Notes"],
          [10, 16, 26, 38, 9, 10, 15, 11, 18, 18, 11, 11, 12, 40])
    for i, it in enumerate(items, 2):
        cooked = it["availability"] == "available"
        ws_menu.append([it["item_id"], it["brand"], it["category"], it["item_name"], it["diet"], int(it["price_inr"]),
                        "Cooked" if cooked else "Not cooked yet",
                        f"=COUNTIF('Photo List'!$A:$A,A{i})",
                        f'=IF(G{i}="Not cooked yet","Pending cooking",IF(H{i}>0,"Yes","No - photo needed"))',
                        f"=IFERROR(INDEX('Photo List'!$F:$F,MATCH(A{i}&\"|1\",'Photo List'!$K:$K,0)),\"\")",
                        f"=COUNTIFS('Photo List'!$A:$A,A{i},'Photo List'!$G:$G,\"Approved\")",
                        f"=COUNTIFS('Photo List'!$A:$A,A{i},'Photo List'!$H:$H,\"Yes\")",
                        f"=COUNTIFS('Photo List'!$A:$A,A{i},'Photo List'!$I:$I,\"Yes\")",
                        "" if cooked else "Will be cooked later - name to be provided again"])
        ws_menu.cell(i, 7).fill = INPUT
        ws_menu.cell(i, 7).font = Font(name=F, size=10, color="0000FF")
    last = len(items) + 1
    _body(ws_menu, 2, last, 1, 14)
    ws_menu.auto_filter.ref = f"A1:N{last}"
    for text, color in (("Yes", "C6EFCE"), ("No - photo needed", "FFC7CE"), ("Pending cooking", "D9D9D9")):
        ws_menu.conditional_formatting.add(f"I2:I{last}", CellIsRule(operator="equal", formula=[f'"{text}"'],
                                                                     fill=PatternFill("solid", bgColor=color)))
    ws_menu.cell(last + 2, 1, "Availability (yellow) is the input column: change a dish to 'Cooked' once it has been "
                              "photographed. Everything else is calculated.").font = Font(name=F, italic=True, size=9)

    # ---------- Photo Groups (you correct the labels here) ----------
    _head(ws_gr, ["Group", "From photo #", "To photo #", "Photos", "Assigned item IDs", "Assigned items",
                  "Confidence", "What I see", "Sample 1", "Sample 2", "Sample 3",
                  "Your correction 1", "Your correction 2", "Your correction 3", "Confirm as-is (Y)"],
          [8, 8, 8, 8, 20, 36, 12, 44, 21, 21, 21, 34, 34, 34, 11])
    cooked_items = [it for it in items if it["availability"] == "available"]
    for i, g in enumerate(grps, 2):
        raw = carried.get(g["group_id"], {}).get("raw", [None] * 4)
        ws_gr.append([g["group_id"], g["n_from"], g["n_to"], f"=C{i}-B{i}+1", " ".join(g["items"]),
                      "; ".join(names.get(x, x) for x in g["items"]) or "(none matched)",
                      CONF[g["confidence"]], g["note"], "", "", ""] + raw)
        ws_gr.row_dimensions[i].height = 88
        for k, p in enumerate(_thumbs(g, index, down)):
            im = XLImage(str(p))
            ws_gr.add_image(im, f"{'IJK'[k]}{i}")
        for c in range(12, 16):
            ws_gr.cell(i, c).fill = INPUT
    last_g = len(grps) + 1
    _body(ws_gr, 2, last_g, 1, 15, wrap=True)
    for row in ws_gr.iter_rows(min_row=2, max_row=last_g, min_col=12, max_col=15):
        for cell in row:
            cell.font = Font(name=F, size=10, color="0000FF")
    dv = DataValidation(type="list", formula1=f"=Lists!$A$2:$A${len(cooked_items) + 1}", allow_blank=True,
                        showErrorMessage=False)
    ws_gr.add_data_validation(dv)
    dv.add(f"L2:N{last_g}")
    dv2 = DataValidation(type="list", formula1='"Y"', allow_blank=True)
    ws_gr.add_data_validation(dv2)
    dv2.add(f"O2:O{last_g}")
    ws_gr.cell(last_g + 2, 1, "How to correct: pick the right dish(es) in 'Your correction' (yellow), or type Y in "
                              "'Confirm as-is'. Then save the workbook - the review app applies it automatically "
                              "within a few seconds. Identical-looking dishes (same dish sold by two brands) "
                              "are added automatically."
              ).font = Font(name=F, italic=True, size=9)

    # ---------- Lists ----------
    ws_li.append(["Cooked items (for the dropdown)"])
    ws_li["A1"].font = Font(name=F, bold=True)
    for it in cooked_items:
        ws_li.append([f"{it['item_id']} - {it['item_name']}"])
    ws_li.column_dimensions["A"].width = 60

    # ---------- Summary ----------
    ws = ws_sum
    ws["A1"] = "Cloud Kitchen - Image Tracker"
    ws["A1"].font = Font(name=F, bold=True, size=14)
    heads = ["Brand", "Total items", "Cooked", "Not cooked yet", "Photo available", "Photo needed",
             "Approved photos", "Saved to final/", "Enhanced & saved"]
    for c, h in enumerate(heads, 1):
        cell = ws.cell(3, c, h)
        cell.font, cell.fill, cell.border = Font(name=F, bold=True, color="FFFFFF", size=10), HEAD, BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[cell.column_letter].width = 18
    brands = list(dict.fromkeys(it["brand"] for it in items))
    M = "'Menu Items'!"
    for i, b in enumerate(brands, 4):
        ws.append([b, f"=COUNTIF({M}$B:$B,A{i})", f'=COUNTIFS({M}$B:$B,A{i},{M}$G:$G,"Cooked")',
                   f'=COUNTIFS({M}$B:$B,A{i},{M}$G:$G,"Not cooked yet")',
                   f'=COUNTIFS({M}$B:$B,A{i},{M}$I:$I,"Yes")',
                   f'=COUNTIFS({M}$B:$B,A{i},{M}$I:$I,"No - photo needed")',
                   f"=SUMIF({M}$B:$B,A{i},{M}$K:$K)", f"=SUMIF({M}$B:$B,A{i},{M}$L:$L)",
                   f"=SUMIF({M}$B:$B,A{i},{M}$M:$M)"])
    t = 4 + len(brands)
    ws.append(["TOTAL"] + [f"=SUM({chr(66 + k)}4:{chr(66 + k)}{t - 1})" for k in range(8)])
    for row in ws.iter_rows(min_row=4, max_row=t, min_col=1, max_col=9):
        for cell in row:
            cell.font, cell.border = Font(name=F, size=10, bold=(cell.row == t)), BORDER
    notes = ["Notes",
             "- 'Photo available' = a photo was matched to the dish. Check 'Photo Groups' for low-confidence matches.",
             "- 'Approved' / 'Saved' / 'Enhanced' come from the review app (SQLite database) and are synced "
             "automatically a few seconds after every change. If this file is open in Excel the sync waits until "
             "it is closed.",
             "- 'Not cooked yet' dishes are excluded from photo matching until their Availability is set to Cooked."]
    for k, n in enumerate(notes):
        ws.cell(t + 2 + k, 1, n).font = Font(name=F, size=10, bold=(k == 0))

    # ---------- save atomically; if Excel has the file open the swap fails and the caller retries later ----------
    tmp = XLSX.with_name("~Image_Tracker.tmp.xlsx")
    wb.save(tmp)
    # Safeguard: if you saved new corrections while this workbook was being built, do not overwrite them -
    # back off so they are applied first (the caller retries afterwards).
    if {g: _sig(c["raw"]) for g, c in pending_corrections().items()} != \
            {g: _sig(c["raw"]) for g, c in carried.items()}:
        tmp.unlink(missing_ok=True)
        return {"workbook": str(XLSX), "items": len(items), "photo_rows": r - 2, "groups": len(grps),
                "locked": False, "retry": True}
    try:
        os.replace(tmp, XLSX)
        locked = False
    except PermissionError:
        tmp.unlink(missing_ok=True)
        locked = True
    return {"workbook": str(XLSX), "items": len(items), "photo_rows": r - 2, "groups": len(grps),
            "locked": locked, "retry": False}
