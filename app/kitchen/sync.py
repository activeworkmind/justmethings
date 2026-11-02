"""Automatic two-way sync between the SQLite database and Image_Tracker.xlsx.

  DB -> Excel   any saved review decision marks the workbook dirty; it is rebuilt a few seconds later
                (debounced). If Excel has the file open the write is retried until it is closed.
  Excel -> DB   when the workbook on disk contains corrections you have not applied yet (typed in
                'Photo Groups' and saved), they are applied, photos are re-selected and the workbook is
                refreshed. The GUI is told to reload.

The owner calls tick() regularly from its main loop; all heavy work runs on a background thread."""
import logging
import threading
import time

from . import db, excel, review, select

log = logging.getLogger(__name__)


class Syncer:
    PUSH_DELAY = 2.5      # seconds of quiet before rebuilding the workbook
    RETRY = 8.0           # retry interval while Excel holds the file
    WATCH = 4.0           # how often the workbook is checked for new corrections

    def __init__(self):
        self.status = "Excel: starting..."
        self._dirty, self._due = True, 0.0
        self._busy = False
        self._applied = False
        self._seen_mtime = None
        self._next_watch = 0.0

    # ----- called by the app -----
    def mark_dirty(self):
        self._dirty, self._due = True, time.time() + self.PUSH_DELAY

    def take_applied(self) -> bool:
        """True once after corrections from Excel were applied (the app should reload its selection)."""
        a, self._applied = self._applied, False
        return a

    def tick(self):
        if self._busy:
            return
        now = time.time()
        if now >= self._next_watch:
            self._next_watch = now + self.WATCH
            try:
                mtime = excel.XLSX.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime is not None and mtime != self._seen_mtime:
                self._seen_mtime = mtime
                if excel.pending_corrections():
                    return self._spawn(self._pull)
        if self._dirty and now >= self._due:
            self._spawn(self._push)

    def flush(self):
        """Blocking push (window close). Returns True if the workbook was written."""
        if self._busy:
            return False
        self._busy = True
        try:
            return self._push_now()
        finally:
            self._busy = False

    # ----- internals -----
    def _spawn(self, fn):
        self._busy = True

        def run():
            try:
                fn()
            except Exception:
                log.exception("Excel sync failed")
                self.status = "Excel: sync error (see work/logs/run.log)"
                self._due = time.time() + self.RETRY
            finally:
                self._busy = False
        threading.Thread(target=run, daemon=True).start()

    def _push_now(self) -> bool:
        res = excel.run()
        if res.get("retry"):                                  # new corrections arrived mid-build: apply them first
            self._seen_mtime = None
            self._dirty, self._due = True, time.time() + 1.0
            self._next_watch = 0.0
            return False
        if res["locked"]:
            self.status = "Excel: file is open in Excel - will sync when it is closed"
            self._dirty, self._due = True, time.time() + self.RETRY
            return False
        self._dirty = False
        try:
            self._seen_mtime = excel.XLSX.stat().st_mtime        # our own write is not a user edit
        except FileNotFoundError:
            pass
        self.status = "Excel: in sync  " + time.strftime("%H:%M:%S")
        return True

    def _push(self):
        self._push_now()

    def _pull(self):
        self.status = "Excel: applying your corrections..."
        res = excel.apply()
        if res["groups_updated"]:
            select.run(db.connect())
            review.run()
            self._applied = True
        self._push_now()
