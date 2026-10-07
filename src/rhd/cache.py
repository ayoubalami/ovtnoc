from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .common import canonical, digest


class DetectionCache:
    """Every frame is explicit, including empty detections. No identity labels."""

    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"Detection cache not found: {path}")
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS frames (
                    camera INTEGER NOT NULL, frame INTEGER NOT NULL, payload TEXT NOT NULL,
                    detector_ms REAL NOT NULL, appearance_ms REAL NOT NULL,
                    PRIMARY KEY(camera, frame));
            """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def ensure_meta(self, key: str, value):
        old = self.get_meta(key)
        if old is not None and canonical(old) != canonical(value):
            raise ValueError(f"Cache provenance mismatch for {key}; choose a new cache path")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", (key, canonical(value)))
        self.db.commit()

    def add_frame(self, camera: int, frame: int, detections: list, detector_ms=0.0, appearance_ms=0.0):
        self.db.execute("INSERT INTO frames VALUES (?,?,?,?,?)", (camera, frame, canonical(detections), detector_ms, appearance_ms))

    def commit(self):
        self.db.commit()

    def next_frame(self, camera: int) -> int:
        count, end = self.db.execute("SELECT COUNT(*),MAX(frame) FROM frames WHERE camera=?", (camera,)).fetchone()
        if count and count != end + 1:
            raise ValueError(f"Cache has gaps for camera {camera}; extraction cannot resume")
        return count

    def validate_range(self, cameras: list[int], start: int, stop: int, fps: float):
        for camera in cameras:
            meta = self.get_meta(f"camera:{camera}")
            if meta is None or abs(meta["fps"] - fps) > 0.01:
                raise ValueError(f"Missing camera metadata or mismatched FPS for camera {camera}")
            n = self.db.execute("SELECT COUNT(*) FROM frames WHERE camera=? AND frame>=? AND frame<?", (camera, start, stop)).fetchone()[0]
            if n != stop - start:
                raise ValueError(f"Camera {camera}: need {stop-start} cached frames [{start},{stop}), found {n}. Resume extraction first")

    def frame(self, camera: int, frame: int) -> tuple[list, float, float]:
        row = self.db.execute("SELECT payload,detector_ms,appearance_ms FROM frames WHERE camera=? AND frame=?", (camera, frame)).fetchone()
        if row is None:
            raise ValueError(f"Frame {camera}/{frame} absent; missing data is not an empty detection")
        return json.loads(row[0]), row[1], row[2]

    def signature(self, cameras: list[int], start: int, stop: int) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(digest(dict(self.db.execute("SELECT key,value FROM metadata ORDER BY key"))).encode())
        for camera in cameras:
            for row in self.db.execute("SELECT camera,frame,payload,detector_ms,appearance_ms FROM frames WHERE camera=? AND frame>=? AND frame<? ORDER BY frame", (camera, start, stop)):
                h.update(canonical(row).encode())
        return h.hexdigest()

