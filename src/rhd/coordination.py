from __future__ import annotations

import heapq
import math
import random

import numpy as np

from .common import canonical, point_in_polygon, unit_vector


class Coordinator:
    """Causal exit messages, arrival-window hints, and fixed online identity association."""

    def __init__(self, cfg: dict):
        self.routes = cfg["routes"]
        self.fps = cfg["fps"]
        self.delay = cfg["communication"].get("delay_s", 0.0)
        self.loss = cfg["communication"].get("loss_probability", 0.0)
        self.random = random.Random(cfg["seed"])
        self.distance_limit = cfg["association"].get("max_cosine_distance", 0.25)
        self.margin = cfg["association"].get("ambiguity_margin", 0.05)
        self.mapping: dict[tuple[int, int], int] = {}
        self.active_local: dict[int, set[int]] = {}
        self.next_id = 1
        self.sent: set[tuple[int, int, int]] = set()
        self.pending: list[tuple[float, int, dict]] = []
        self.available: list[dict] = []
        self.used: set[int] = set()
        self.serial = 0
        self.messages = 0
        self.dropped = 0
        self.bytes = 0
        self.log: list[dict] = []

    def deliver(self, now: float):
        while self.pending and self.pending[0][0] <= now + 1e-10:
            delivery, _, message = heapq.heappop(self.pending)
            self.available.append(message)
            self.log.append({"event": "delivered", "time_s": now, "message_id": message["id"], "scheduled_delivery_s": delivery})
        self.available = [m for m in self.available if m["end_s"] >= now - 1e-10]

    def arrival_priority(self, camera: int, now: float) -> float:
        weights = []
        for message in self.available:
            if message["id"] in self.used or message["destination"] != camera or not message["begin_s"] <= now <= message["end_s"]:
                continue
            middle = (message["begin_s"] + message["end_s"]) / 2
            width = max((message["end_s"] - message["begin_s"]) / 2, 1 / self.fps)
            weights.append(message["confidence"] * math.exp(-0.5 * ((now - middle) / width) ** 2))
        return max(weights, default=0.0)

    def assign(self, camera: int, track: dict, now: float) -> int:
        key = (camera, track["local_id"])
        if key in self.mapping:
            return self.mapping[key]
        feature = unit_vector(track["feature"])
        candidates = []
        if np.linalg.norm(feature) > 0:
            for message in self.available:
                if message["id"] in self.used or message["destination"] != camera:
                    continue
                if not message["begin_s"] <= now <= message["end_s"] or not point_in_polygon(track["point"], message["entry_polygon"]):
                    continue
                # Prevent two active destination tracks from sharing a global ID.
                if any(self.mapping.get((camera, local)) == message["global_id"] for local in self.active_local.get(camera, set())):
                    continue
                other = unit_vector(message["feature"])
                if other.size != feature.size or np.linalg.norm(other) == 0:
                    continue
                distance = float(1 - feature @ other)
                if distance <= self.distance_limit:
                    candidates.append((distance, message["id"], message))
        candidates.sort(key=lambda c: (c[0], c[1]))
        # Multiple routes/messages for one identity are not competing identities.
        distinct = []
        seen_ids = set()
        for candidate in candidates:
            if candidate[2]["global_id"] not in seen_ids:
                distinct.append(candidate)
                seen_ids.add(candidate[2]["global_id"])
        if distinct and (len(distinct) == 1 or distinct[1][0] - distinct[0][0] >= self.margin):
            distance, message_id, message = distinct[0]
            global_id = message["global_id"]
            self.used.update(m["id"] for m in self.available if m["global_id"] == global_id)
            self.log.append({"event": "associated", "time_s": now, "camera": camera,
                             "local_id": track["local_id"], "global_id": global_id,
                             "message_id": message_id, "cosine_distance": distance})
        else:
            global_id = self.next_id
            self.next_id += 1
        self.mapping[key] = global_id
        return global_id

    def set_active(self, camera: int, tracks: list[dict]):
        self.active_local[camera] = {t["local_id"] for t in tracks}

    def observe(self, camera: int, tracks: list[dict], now: float):
        for track in tracks:
            key = (camera, track["local_id"])
            for index, route in enumerate(self.routes):
                token = (camera, track["local_id"], index)
                if route["source"] != camera or token in self.sent:
                    continue
                if not point_in_polygon(track["point"], route["exit_polygon"]):
                    continue
                direction = np.asarray(route.get("direction", [0, 0]))
                if np.linalg.norm(direction) > 0 and np.dot(direction, track["velocity"]) <= 0:
                    continue
                self.sent.add(token)
                self.serial += 1
                lo, hi = route["travel_time_s"]
                message = {"id": self.serial, "source": camera, "destination": route["destination"],
                           "exit_time_s": now, "begin_s": now + lo, "end_s": now + hi,
                           "global_id": self.mapping[key], "feature": track["feature"],
                           "confidence": track["score"] / (1 + track["age_since_detection"]),
                           "entry_polygon": route["entry_polygon"]}
                self.messages += 1
                self.bytes += len(canonical(message).encode())
                dropped = self.random.random() < self.loss
                self.log.append({"event": "sent", "time_s": now, "message": message, "dropped": dropped})
                if dropped:
                    self.dropped += 1
                else:
                    # A zero-delay message is consumed at the next replay tick, never
                    # by a later camera within the same tick. Camera order cannot leak.
                    delivery = now + max(self.delay, 1 / self.fps)
                    heapq.heappush(self.pending, (delivery, self.serial, message))


class Scheduler:
    """A shared token budget; adaptive policies spend identical available call counts."""

    policies = ("full", "fixed", "independent", "coordinated", "identity")

    def __init__(self, cameras: list[int], policy: str, budget: float, interval: int, options: dict):
        if policy not in self.policies:
            raise ValueError(f"Unknown policy {policy}")
        if not 0 < budget <= 1 or interval < 1:
            raise ValueError("budget must be in (0,1] and interval >=1")
        if policy == "full" and budget != 1:
            raise ValueError("full policy requires budget=1")
        self.cameras = sorted(cameras)
        self.policy, self.budget, self.interval = policy, budget, interval
        self.max_gap = options.get("max_detection_gap", 15)
        if self.max_gap < 1:
            raise ValueError("max_detection_gap must be >=1")
        self.uncertainty_weight = options.get("uncertainty_weight", 1.0)
        self.handoff_weight = options.get("handoff_weight", 2.0)
        self.identity_weight = options.get("identity_weight", 1.0)
        if not math.isfinite(self.identity_weight) or self.identity_weight < 0:
            raise ValueError("identity_weight must be finite and nonnegative")
        self.credit = float(len(cameras))
        self.last = {c: None for c in cameras}
        self.calls = 0

    def choose(self, tick: int, tracks: dict[int, list], coordinator: Coordinator, now: float) -> dict[int, dict]:
        n = len(self.cameras)
        if tick > 0:
            self.credit = min(float(n), self.credit + self.budget * n)
        available = int(self.credit + 1e-9)
        scores = []
        decisions = {}
        for position, camera in enumerate(self.cameras):
            age = tick - self.last[camera] if self.last[camera] is not None else self.max_gap
            uncertainty = max((t["uncertainty"] for t in tracks[camera]), default=0.0)
            local = age / self.max_gap + self.uncertainty_weight * min(uncertainty, 5.0)
            # Overdue cameras are prioritised over ordinary arrival hints, subject
            # to the shared cap. A hard guarantee is impossible below the safety floor.
            if age >= self.max_gap:
                local += 100.0
            hint = coordinator.arrival_priority(camera, now) if self.policy in ("coordinated", "identity") else 0.0
            identity_hint = (coordinator.identity_priority(camera, now)
                             if self.policy == "identity" and self.identity_weight > 0 else 0.0)
            score = local + self.handoff_weight * hint + self.identity_weight * identity_hint
            eligible = self.policy != "fixed" or tick % self.interval == 0
            decisions[camera] = {"detect": False, "age": age, "score": score, "handoff_priority": hint,
                                 "identity_priority": identity_hint,
                                 "overdue": age >= self.max_gap}
            if eligible:
                # Deterministic rotating tie-break avoids privileging camera IDs.
                tie = (position - tick) % n
                scores.append((-score, tie, camera))
        scores.sort()
        for _, _, camera in scores[:available]:
            decisions[camera]["detect"] = True
            self.last[camera] = tick
            self.credit -= 1
            self.calls += 1
        return decisions

