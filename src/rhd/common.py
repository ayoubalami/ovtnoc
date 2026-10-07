from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersect = np.maximum(hi - lo, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersect / np.maximum(area_a[:, None] + area_b[None, :] - intersect, 1e-12)


def unit_vector(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def point_in_polygon(point, polygon) -> bool:
    """Ray casting, including polygon boundaries, in normalized image coordinates."""
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-10 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cameras = cfg.get("cameras", [])
    if not cameras or len(set(cameras)) != len(cameras) or any(type(c) is not int or c < 1 for c in cameras):
        raise ValueError("cameras must be distinct positive integer IDs")
    cfg.setdefault("fps", 15.0)
    cfg.setdefault("start_frame", 0)
    cfg.setdefault("stop_frame", 300)
    cfg.setdefault("seed", 7)
    cfg.setdefault("routes", [])
    cfg.setdefault("tracker", {})
    cfg.setdefault("scheduler", {})
    cfg.setdefault("association", {})
    cfg.setdefault("communication", {})
    kind = cfg["association"].get("kind", "immediate")
    if kind not in ("immediate", "bounded"):
        raise ValueError("association.kind must be immediate or bounded")
    if kind == "bounded":
        from .bounded import validate_options
        validate_options(cfg["association"])
    from .quality import MovingVehicleFilter
    MovingVehicleFilter(cfg.get("output_filter"), 1, 1)  # Validate without observing any data.
    if not math.isfinite(cfg["fps"]) or cfg["fps"] <= 0:
        raise ValueError("fps must be positive")
    if any(type(cfg[k]) is not int for k in ("start_frame", "stop_frame")) or cfg["start_frame"] < 0 or cfg["stop_frame"] <= cfg["start_frame"]:
        raise ValueError("stop_frame must exceed start_frame; stop_frame is exclusive")
    for route in cfg["routes"]:
        if route["source"] not in cameras or route["destination"] not in cameras or route["source"] == route["destination"]:
            raise ValueError("route endpoints must be distinct configured cameras")
        lo, hi = route["travel_time_s"]
        if not 0 <= lo < hi or not math.isfinite(hi):
            raise ValueError("travel_time_s must be [lower, upper] with 0 <= lower < upper")
        for key in ("exit_polygon", "entry_polygon"):
            points = route[key]
            if len(points) < 3 or any(len(p) != 2 or any(not 0 <= v <= 1 for v in p) for p in points):
                raise ValueError(f"{key} must have >=3 normalized [x,y] points")
        route.setdefault("direction", [0.0, 0.0])
        if len(route["direction"]) != 2 or not all(math.isfinite(v) for v in route["direction"]):
            raise ValueError("route direction must be a finite [dx,dy] vector")
    comm = cfg["communication"]
    if not math.isfinite(comm.get("delay_s", 0)) or comm.get("delay_s", 0) < 0 or not 0 <= comm.get("loss_probability", 0) <= 1:
        raise ValueError("invalid communication delay/loss")
    if cfg.get("phase") in ("validation", "test"):
        for name in ("route_calibration", "appearance_calibration", "bounded_calibration", "scheduling_calibration"):
            training = cfg.get(name)
            if training and "start_frame" in training and "stop_frame" in training:
                if cfg["start_frame"] < training["stop_frame"] and cfg["stop_frame"] > training["start_frame"]:
                    raise ValueError(f"Evaluation replay overlaps {name} development data")
    return cfg

