"""SQLite progress ledger. One row per source file; commits are batched so a crash loses
at most BATCH_SIZE photos of work, and every stage skips rows it already finished."""
import shutil
import sqlite3
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id        TEXT PRIMARY KEY,   -- sha1(relpath|size|mtime)[:10]
    path      TEXT NOT NULL,
    kind      TEXT NOT NULL,      -- image | video | other
    size      INTEGER NOT NULL,
    status    TEXT NOT NULL DEFAULT 'scanned',  -- scanned | downscaled | error | skipped
    down_path TEXT,
    down_bytes INTEGER,
    width     INTEGER,
    height    INTEGER,
    sharp     REAL,
    lum       REAL,
    phash     TEXT,
    error     TEXT,
    updated   REAL
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(kind, status);

-- small key/value store (e.g. which Excel corrections were already applied)
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- photos the user deleted for a dish: never suggested again for that item
CREATE TABLE IF NOT EXISTS excluded (
    item_id    TEXT NOT NULL,
    photo_id   TEXT NOT NULL,
    deleted_at REAL,
    PRIMARY KEY (item_id, photo_id)
);

-- one row per output slot (menu item x rank); written by the review GUI, read by finalize + Excel
CREATE TABLE IF NOT EXISTS slots (
    slot_key   TEXT PRIMARY KEY,          -- 'BR-001:1'
    item_id    TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    photo_id   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected | deleted
    name       TEXT,
    rot        INTEGER NOT NULL DEFAULT 0,
    tilt       REAL NOT NULL DEFAULT 0,
    crop       TEXT,                      -- JSON [x0,y0,x1,y1] as fractions
    aspect     TEXT NOT NULL DEFAULT 'free',
    enhance    INTEGER NOT NULL DEFAULT 1,        -- decision: apply daylight enhancement?
    look       TEXT,                              -- JSON list of studio steps (brand background, shadows, ...) or NULL
    reviewed_at REAL,
    exported   INTEGER NOT NULL DEFAULT 0,        -- final_output/ file written for the current decision
    exported_enhanced INTEGER NOT NULL DEFAULT 0, -- that file was enhanced
    final_path TEXT,
    exported_at REAL
);
"""


def connect():
    config.WORK.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    if "look" not in {r[1] for r in con.execute("PRAGMA table_info(slots)")}:     # upgrade an older database in place
        con.execute("ALTER TABLE slots ADD COLUMN look TEXT")
        con.commit()
    return con


def meta_get(con, key, default=None):
    r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def meta_set(con, key, value):
    con.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))


DUMP_PATH = config.ROOT / "db" / "state.sql"


def dump(path=DUMP_PATH) -> str:
    """Text export of the whole database (schema + data): diff-friendly, safe to keep in git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect()
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in con.iterdump():
                f.write(line + "\n")
    finally:
        con.close()
    return str(path)


def restore(path=DUMP_PATH, force=False) -> str:
    """Rebuild work/state.db from a text export. Refuses to overwrite an existing database unless force."""
    if config.DB_PATH.exists() and not force:
        raise FileExistsError(f"{config.DB_PATH} exists - use --force to replace it")
    config.WORK.mkdir(parents=True, exist_ok=True)
    for ext in ("", "-wal", "-shm"):
        p = config.DB_PATH.with_name(config.DB_PATH.name + ext)
        if p.exists():
            p.unlink()
    con = sqlite3.connect(config.DB_PATH)
    try:
        with open(path, encoding="utf-8") as f:
            con.executescript(f.read())
    finally:
        con.close()
    return str(config.DB_PATH)


def backup():
    """Copy the ledger aside (call after each stage / batch group)."""
    config.BACKUPS.mkdir(parents=True, exist_ok=True)
    if config.DB_PATH.exists():
        shutil.copy2(config.DB_PATH, config.BACKUPS / f"state_{time.strftime('%Y%m%d_%H%M%S')}.db")
        old = sorted(config.BACKUPS.glob("state_*.db"))[:-10]
        for f in old:
            f.unlink()
