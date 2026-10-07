from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cache import DetectionCache
from .common import file_hash, point_in_polygon
from .evaluation import read_rows, segments


def fit_routes(cfg: dict, cache_path: Path, ground_truth: Path, start: int, stop: int, min_samples=5) -> dict:
    """Fit route windows from an explicit development interval; inference never loads labels."""
    if not cfg["routes"] or not 0 <= start < stop or min_samples < 1:
        raise ValueError("Provide manually audited route polygons, valid training range, and min_samples>=1")
    data = read_rows(ground_truth, cfg["cameras"], start, stop)
    lookup = {(r["camera"], r["id"], r["frame"]): r for r in data}
    spans = segments(data)
    fitted = copy.deepcopy(cfg)
    statistics = []
    with DetectionCache(cache_path, readonly=True) as cache:
        sizes = {c: (cache.get_meta(f"camera:{c}")["width"], cache.get_meta(f"camera:{c}")["height"]) for c in cfg["cameras"]}
        for route in fitted["routes"]:
            travel = []
            for identity, trajectory in spans.items():
                for index, source in enumerate(trajectory[:-1]):
                    dest = trajectory[index + 1]
                    if source[2] != route["source"] or dest[2] != route["destination"] or dest[0] <= source[1]:
                        continue
                    width, height = sizes[source[2]]
                    crossing = None
                    previous = None
                    for frame in range(source[0], source[1] + 1):
                        box = lookup[(source[2], identity, frame)]["box"]
                        point = [(box[0] + box[2]) / (2 * width), box[3] / height]
                        direction = np.asarray(route.get("direction", [0, 0]))
                        moving = np.linalg.norm(direction) == 0 or (previous is not None and np.dot(direction, np.asarray(point) - previous) > 0)
                        if moving and point_in_polygon(point, route["exit_polygon"]):
                            crossing = frame
                            break
                        previous = point
                    if crossing is None:
                        continue
                    width, height = sizes[dest[2]]
                    box = lookup[(dest[2], identity, dest[0])]["box"]
                    point = [(box[0] + box[2]) / (2 * width), box[3] / height]
                    if point_in_polygon(point, route["entry_polygon"]):
                        travel.append((dest[0] - crossing) / cfg["fps"])
            if len(travel) < min_samples:
                raise ValueError(f"Route {route['source']}->{route['destination']}: only {len(travel)} development examples, need {min_samples}; inspect zones or extend development interval")
            lo, hi = np.quantile(travel, [0.05, 0.95])
            route["travel_time_s"] = [float(max(0, lo - 1 / cfg["fps"])), float(hi + 1 / cfg["fps"])]
            statistics.append({"source": route["source"], "destination": route["destination"],
                               "samples": len(travel), "median_s": float(np.median(travel))})
    fitted["route_calibration"] = {"start_frame": start, "stop_frame": stop,
                                    "ground_truth_sha256": file_hash(ground_truth),
                                    "estimator": "5th/95th percentiles plus one frame margin", "routes": statistics}
    return fitted
