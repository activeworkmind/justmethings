"""Stage 1: segregate. Walk assets/, register images, record (but never open) videos."""
import csv
import hashlib
import os
import time

from . import config


def file_id(rel: str, size: int, mtime: float) -> str:
    return hashlib.sha1(f"{rel}|{size}|{int(mtime)}".encode()).hexdigest()[:10]


def run(con) -> dict:
    counts = {"image": 0, "video": 0, "other": 0, "new": 0}
    videos = []
    for dirpath, _, names in os.walk(config.ASSETS):
        for name in names:
            p = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            st = os.stat(p)
            if ext in config.RAW_EXT or ext in config.IMG_EXT:
                kind = "image"
            elif ext in config.VIDEO_EXT:
                kind = "video"
                videos.append((p, st.st_size))
            else:
                kind = "other"
            counts[kind] += 1
            rel = os.path.relpath(p, config.ASSETS)
            cur = con.execute(
                "INSERT OR IGNORE INTO files(id,path,kind,size,status,updated) VALUES(?,?,?,?,?,?)",
                (file_id(rel, st.st_size, st.st_mtime), p, kind, st.st_size,
                 "scanned" if kind == "image" else "skipped", time.time()),
            )
            counts["new"] += cur.rowcount
    con.commit()
    out = config.WORK / "skipped_videos_and_other.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes"])
        w.writerows(videos)
    return counts
