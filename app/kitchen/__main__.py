import logging

import typer

from . import config, db, downscale, review, scan, select

app = typer.Typer(add_completion=False, help="Cloud-kitchen image pipeline.")


@app.callback()
def _setup():
    config.LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=config.LOGS / "run.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


@app.command("scan")
def scan_cmd():
    """1. Segregate: register images, list (and skip) videos."""
    con = db.connect()
    print(scan.run(con))
    db.backup()


@app.command("downscale")
def downscale_cmd(limit: int = typer.Option(None, help="only first N (for testing)"),
                  workers: int = config.WORKERS):
    """2. Decode + downscale every image to <=1 MB JPEG. Resumable."""
    con = db.connect()
    print(downscale.run(con, limit, workers))
    db.backup()


@app.command("sheets")
def sheets_cmd():
    """Numbered contact sheets of every photo (work/label_sheets) used to label photo groups."""
    from . import sheets
    print(sheets.run())


@app.command("classify")
def classify_cmd():
    """Optional: CLIP scores per photo (only a hint; group labels in photo_groups.csv decide)."""
    from . import classify
    print(classify.run(db.connect()))


@app.command("select")
def select_cmd():
    """3. Pick top-3 photos per cooked item from the photo groups; refresh output/ and ready_for_crop/."""
    print(select.run(db.connect()))
    print("review sheet:", review.run())


@app.command("export-plain")
def export_plain_cmd():
    """Plain export: named, upright, <=1 MB, no crop/enhancement, one folder per brand."""
    print(select.export_plain())


@app.command("gui")
def gui_cmd():
    """4. Manual review: crop / rotate / rename / enhance-or-not, one slot at a time."""
    from . import gui
    gui.main()


@app.command("finalize")
def finalize_cmd(workers: int = 3):
    """5. Render all approved GUI slots into final/<brand>/<category>/ (<=1 MB each), then update Excel."""
    from . import excel, finalize
    print(finalize.run(workers))
    print(excel.run())


@app.command("excel")
def excel_cmd():
    """Rebuild the Excel tracker from the database (menu availability, photos, review/enhance status)."""
    from . import excel
    print(excel.run())


@app.command("apply-excel")
def apply_excel_cmd():
    """Read your corrections from the 'Photo Groups' sheet, update photo_groups.csv and re-run select."""
    from . import excel
    print(excel.apply())
    print(select.run(db.connect()))
    print("review sheet:", review.run())
    print(excel.run())


@app.command("retag")
def retag_cmd(apply: bool = typer.Option(False, "--apply", help="make the changes (a backup is taken first)")):
    """Correct brand tags from the table mat (green = Biryani Rules, royal = Sunday Biryani, plain = Day Meal)."""
    from . import retag
    with db.connect() as con:
        plan = retag.plan(con)
    for r in plan:
        if r["status"] != "unchanged":
            print(f"{r['group']:6} {r['mat'] or '-':6} photos {len(r['photos']):3}  {r['status']:42} "
                  f"{','.join(r['before']) or '-'}  ->  {','.join(r['after']) or '-'}")
    print(f"{sum(1 for r in plan if r['status'] == 'unchanged')} groups already correct")
    if apply:
        res = retag.apply()
        print(f"moved {len(res['moved'])} saved decisions, dropped {len(res['dropped'])}; backup: {res['backup']}")
        for old, new, st, a, b in res["moved"]:
            if old != new or a != b:
                print(f"  {old:9} -> {new:9} {st:9} {a}  =>  {b}")
        for key, st, name in res["dropped"]:
            print(f"  DROPPED {key} ({st}) {name}")


@app.command("db-dump")
def db_dump_cmd():
    """Export the database to db/state.sql (text, versioned in git)."""
    print(db.dump())


@app.command("db-restore")
def db_restore_cmd(force: bool = typer.Option(False, help="replace an existing work/state.db")):
    """Rebuild work/state.db from db/state.sql."""
    print(db.restore(force=force))


@app.command("status")
def status_cmd():
    con = db.connect()
    for r in con.execute("SELECT kind,status,COUNT(*) n FROM files GROUP BY kind,status"):
        print(f"{r['kind']:6} {r['status']:11} {r['n']}")
    for r in con.execute("SELECT status,COUNT(*) n,SUM(exported) e,SUM(exported_enhanced) en FROM slots GROUP BY status"):
        print(f"slots  {r['status']:9} {r['n']}  (exported {r['e']}, enhanced {r['en']})")
    for r in con.execute("SELECT path,error FROM files WHERE status='error'"):
        print("ERROR", r["path"], r["error"])


if __name__ == "__main__":
    app()
