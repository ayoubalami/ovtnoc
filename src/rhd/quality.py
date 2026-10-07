"""Causal output eligibility for a moving-vehicle tracking study."""
from __future__ import annotations

import math
import numpy as np


class MovingVehicleFilter:
    def __init__(self, settings, width, height):
        self.settings = settings or {"kind": "none"}
        if self.settings.get("kind") not in ("none", "causal_motion"):
            raise ValueError("output_filter.kind must be none or causal_motion")
        self.width, self.height = width, height
        self.window = self.settings.get("window_frames", 30)
        self.minimum = self.settings.get("minimum_observations", 3)
        self.displacement = self.settings.get("displacement_fraction", .005)
        if (type(self.window) is not int or self.window < 2 or type(self.minimum) is not int
                or self.minimum < 2 or not math.isfinite(self.displacement) or not 0 < self.displacement <= 1):
            raise ValueError("Invalid causal output-filter parameters")
        self.states = {}
        self.statistics = {"candidate_observations": 0, "suppressed_observations": 0,
                           "confirmed_tracks": 0, "prediction_only_points_ignored": 0}

    def apply(self, tracks, frame, live_ids):
        if self.settings["kind"] == "none":
            return tracks
        output = []
        for track in tracks:
            self.statistics["candidate_observations"] += 1
            identity = track["local_id"]
            state = self.states.setdefault(identity, {"points": [], "confirmed": False})
            if not state["confirmed"] and track["age_since_detection"] == 0:
                box = track["box"]
                center = [(box[0] + box[2]) / (2 * self.width), (box[1] + box[3]) / (2 * self.height)]
                state["points"].append((frame, center))
                state["points"][:] = [(f, p) for f, p in state["points"] if frame - f < self.window]
                if len(state["points"]) >= self.minimum:
                    coords = np.asarray([point for _, point in state["points"]])
                    if np.linalg.norm(np.ptp(coords, axis=0)) >= self.displacement:
                        state["confirmed"] = True
                        self.statistics["confirmed_tracks"] += 1
                        state["points"].clear()
            elif not state["confirmed"]:
                self.statistics["prediction_only_points_ignored"] += 1
            if state["confirmed"]:
                output.append(track)
            else:
                self.statistics["suppressed_observations"] += 1
        self.states = {identity: state for identity, state in self.states.items() if identity in live_ids}
        return output
