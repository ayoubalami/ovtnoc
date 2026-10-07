"""Causal, bounded confirmation of global identities from actual observations."""
from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import canonical, point_in_polygon, unit_vector
from .coordination import Coordinator


CANDIDATE_REJECTION_GATES = (
    "used_message", "other_destination", "delivered_after_deadline", "active_identity",
    "no_entry_observation", "travel_window", "source_descriptor_incompatible",
    "source_descriptor_missing", "appearance_distance",
)


def validate_options(options: dict):
    allowed = {"kind", "max_cosine_distance", "ambiguity_margin", "confirmation_frames",
               "minimum_observations", "gallery_size", "time_weight", "feature_mode", "source_pooling",
               "source_retention_frames", "diagnostics"}
    if set(options) - allowed:
        raise ValueError(f"Unknown bounded association options: {sorted(set(options) - allowed)}")
    confirmation = options.get("confirmation_frames", 15)
    if type(confirmation) is not int or confirmation < 0:
        raise ValueError("confirmation_frames must be a nonnegative integer")
    if "source_retention_frames" in options:
        retention = options["source_retention_frames"]
        if type(retention) is not int or retention < 0:
            raise ValueError("source_retention_frames must be a nonnegative integer")
    if type(options.get("diagnostics", False)) is not bool:
        raise ValueError("diagnostics must be a boolean")
    for name, default in (("minimum_observations", 3), ("gallery_size", 5)):
        value = options.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if options.get("minimum_observations", 3) > options.get("gallery_size", 5):
        raise ValueError("minimum_observations must not exceed gallery_size")
    if options.get("feature_mode", "raw") not in ("raw", "ema_observed"):
        raise ValueError("feature_mode must be raw or ema_observed")
    if options.get("source_pooling", "nearest_gallery") not in ("nearest_gallery", "exit_prototype"):
        raise ValueError("source_pooling must be nearest_gallery or exit_prototype")
    for name, default, upper in (("max_cosine_distance", .8, 2), ("ambiguity_margin", .03, 2), ("time_weight", .1, 2)):
        value = options.get(name, default)
        if not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name} must be finite and in [0,{upper}]")


class BoundedCoordinator(Coordinator):
    """Hold entrant IDs until observed evidence resolves them or their deadline expires.

    Galleries contain only detector observations, never Kalman predictions. The
    gallery affinity is the median, across destination observations, of nearest
    source-gallery cosine distance. Route/time gates precede affinity scoring.
    Assignment uses private null columns and one shared column per global ID.
    Published mappings are permanent; buffered rows are emitted once at commit.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        options = cfg["association"]
        validate_options(options)
        self.window = options.get("confirmation_frames", 15) / self.fps
        # A fixed retention horizon isolates destination waiting from source
        # message expiry in a common-kernel ablation. Omission retains v0.3.
        self.retention_window = options.get("source_retention_frames",
                                            options.get("confirmation_frames", 15)) / self.fps
        self.diagnostics = options.get("diagnostics", False)
        self.minimum = options.get("minimum_observations", 3)
        self.gallery_size = options.get("gallery_size", 5)
        self.time_weight = options.get("time_weight", .1)
        self.feature_mode = options.get("feature_mode", "raw")
        self.source_pooling = options.get("source_pooling", "nearest_gallery")
        self.states = {}
        self.galleries = {}
        self.exit_events = {}
        self.commits = {}

    def deliver(self, now):
        # An observed on-time entrant may finish confirmation after a window ends.
        while self.pending and self.pending[0][0] <= now + 1e-10:
            delivery, _, message = heapq.heappop(self.pending)
            message["delivered_at_s"] = now
            self.available.append(message)
            self.log.append({"event": "delivered", "time_s": now, "message_id": message["id"],
                             "scheduled_delivery_s": delivery})
        self.available = [m for m in self.available if m["end_s"] + self.retention_window >= now - 1e-10]

    def _commit_diagnostics(self, key, now, reason):
        """Describe recorded decisions, without claiming GT correctness or causes.

        Candidate counts/gaps refer to the last evaluation of the entrant. If a
        caller flushes before any resolve, unknown candidate counts stay null.
        """
        state = self.states.get(key)
        observations = len(state["gallery"]) if state else len(self.galleries.get(key, []))
        details = dict(state.get("diagnostics", {})) if state else {}
        if reason == "confirmed_handoff":
            outcome = "early_accepted" if now < state["deadline"] - 1e-10 else "deadline_accepted"
        else:
            outcome = {"confirmation_deadline": "deadline_fresh", "end_of_segment": "interval_end_fresh",
                       "outside_entry_zone": "outside_entry"}[reason]
        details.update({"confirmation_outcome": outcome, "observations": observations,
                        "minimum_observations": self.minimum,
                        "evidence_shortfall": max(0, self.minimum - observations),
                        "assignment_eligible": state is not None and observations >= self.minimum,
                        "legal_candidate_count": details.get("legal_candidate_count"),
                        "last_evaluated_at_s": details.get("last_evaluated_at_s")})
        if state is None:
            details["decision_status"] = "outside_entry_zone"
        elif observations < self.minimum:
            details["decision_status"] = "insufficient_observations"
        else:
            details.setdefault("decision_status", "not_evaluated")
        details.setdefault("blocking_checks", [])
        return details

    def _record_resolution_diagnostics(self, unresolved, now):
        for key, state in unresolved.items():
            candidates = state["candidates"]
            count = len(state["gallery"])
            if count < self.minimum:
                status = "insufficient_observations"
            elif not candidates:
                status = "no_legal_candidate"
            else:
                status = "unassigned_after_batch"
            details = {"decision_status": status, "legal_candidate_count": len(candidates),
                       "candidate_global_ids": [c["global_id"] for c in candidates],
                       "last_evaluated_at_s": now, "blocking_checks": []}
            details["candidate_gates"] = state["candidate_gates"]
            if candidates:
                best = candidates[0]
                competitors = [c["cost"] for other_key, other_state in unresolved.items()
                               if other_key != key for c in other_state["candidates"]
                               if c["global_id"] == best["global_id"]]
                details.update({"best_candidate_global_id": best["global_id"],
                                "best_candidate_cost": best["cost"],
                                "competing_entrants": len(competitors),
                                "best_row_gap": (candidates[1]["cost"] - best["cost"]
                                                 if len(candidates) > 1 else None),
                                "best_column_gap": (min(competitors) - best["cost"]
                                                    if competitors else None)})
            state["diagnostics"] = details

    def _new(self, key, now, reason):
        gid = self.next_id
        self.next_id += 1
        self._commit(key, gid, now, reason)

    def _commit(self, key, gid, now, reason, candidate=None):
        if key in self.mapping:
            raise RuntimeError("A committed identity cannot be rewritten")
        self.mapping[key] = gid
        record = {"event": "identity_committed", "time_s": now, "camera": key[0],
                  "local_id": key[1], "global_id": gid, "reason": reason}
        if key in self.states:
            record["first_time_s"] = self.states[key]["first"]
            record["deadline_s"] = self.states[key]["deadline"]
            record["observations"] = len(self.states[key]["gallery"])
            self.states[key]["resolved"] = True
        if candidate:
            record.update({k: candidate[k] for k in ("message_id", "cosine_distance", "cost", "arrival_s")})
            self.used.update(m["id"] for m in self.available if m["global_id"] == gid)
        if self.diagnostics:
            record["diagnostics"] = self._commit_diagnostics(key, now, reason)
        self.commits[key] = record
        self.log.append(record)

    def update(self, camera, tracks, now):
        self.set_active(camera, tracks)
        incoming = [r for r in self.routes if r["destination"] == camera]
        for track in tracks:
            key = (camera, track["local_id"])
            actual = track.get("age_since_detection", 0) == 0
            raw = unit_vector(track.get("observation_feature", track["feature"])) if actual else None
            valid = actual and np.linalg.norm(raw) > 0
            descriptor = unit_vector(track["feature"]) if valid and self.feature_mode == "ema_observed" else raw
            gallery = self.galleries.setdefault(key, deque(maxlen=self.gallery_size))
            # Repeated update calls in a tick cannot manufacture extra evidence.
            if valid and (not gallery or gallery[-1][0] < now - 1e-10):
                gallery.append((now, descriptor.tolist()))
            if key in self.mapping:
                continue
            if key not in self.states:
                if not any(point_in_polygon(track["point"], r["entry_polygon"]) for r in incoming):
                    self._new(key, now, "outside_entry_zone")
                    continue
                self.states[key] = {"first": now, "deadline": now + self.window,
                                    "gallery": deque(maxlen=self.gallery_size), "entries": [],
                                    "resolved": False, "candidates": []}
            state = self.states[key]
            if now <= state["deadline"] + 1e-10 and actual:
                if not state["entries"] or state["entries"][-1][0] < now - 1e-10:
                    state["entries"].append((now, list(track["point"])))
                if valid and (not state["gallery"] or state["gallery"][-1][0] < now - 1e-10):
                    state["gallery"].append((now, descriptor.tolist()))

    def assign(self, camera, track, now):
        return self.mapping.get((camera, track["local_id"]))

    def _candidates(self, key, state):
        gallery = np.asarray([f for _, f in state["gallery"]])
        gates = None
        if self.diagnostics:
            gates = {"scanned_messages": 0, "legal_message_edges": 0, "eligible_distinct_ids": 0,
                     "destination_gallery_observations": len(gallery),
                     "first_rejection_counts": dict.fromkeys(CANDIDATE_REJECTION_GATES, 0)}
            state["candidate_gates"] = gates

        def reject(name):
            if gates is not None:
                gates["first_rejection_counts"][name] += 1

        if len(gallery) == 0:
            return []
        distinct = {}
        for message in self.available:
            if gates is not None:
                gates["scanned_messages"] += 1
            gid = message["global_id"]
            if message["id"] in self.used:
                reject("used_message")
                continue
            if message["destination"] != key[0]:
                reject("other_destination")
                continue
            if message.get("delivered_at_s", state["first"]) > state["deadline"] + 1e-10:
                reject("delivered_after_deadline")
                continue
            if any((camera, local) != key and self.mapping.get((camera, local)) == gid
                   for camera, active in self.active_local.items() for local in active):
                reject("active_identity")
                continue
            entries = [t for t, point in state["entries"] if point_in_polygon(point, message["entry_polygon"])]
            if not entries:
                reject("no_entry_observation")
                continue
            arrival = min(entries)
            if not message["begin_s"] - 1e-10 <= arrival <= message["end_s"] + 1e-10:
                reject("travel_window")
                continue
            source = np.asarray([message["feature"]] if self.source_pooling == "exit_prototype"
                                else message.get("gallery", [message["feature"]]), dtype=float)
            if source.ndim != 2 or source.shape[1] != gallery.shape[1]:
                reject("source_descriptor_incompatible")
                continue
            source = np.asarray([unit_vector(f) for f in source if np.linalg.norm(f) > 0])
            if len(source) == 0:
                reject("source_descriptor_missing")
                continue
            distance = float(np.median(np.min(np.clip(1 - gallery @ source.T, 0, 2), axis=1)))
            if distance > self.distance_limit:
                reject("appearance_distance")
                continue
            if gates is not None:
                gates["legal_message_edges"] += 1
            middle = (message["begin_s"] + message["end_s"]) / 2
            width = max((message["end_s"] - message["begin_s"]) / 2, 1 / self.fps)
            cost = distance + self.time_weight * ((arrival - middle) / width) ** 2
            candidate = {"global_id": gid, "message_id": message["id"], "cosine_distance": distance,
                         "cost": cost, "arrival_s": arrival}
            if gid not in distinct or (cost, message["id"]) < (distinct[gid]["cost"], distinct[gid]["message_id"]):
                distinct[gid] = candidate
        if gates is not None:
            gates["eligible_distinct_ids"] = len(distinct)
        return sorted(distinct.values(), key=lambda c: (c["cost"], c["global_id"]))

    def resolve(self, now):
        for key, state in self.states.items():
            if key not in self.mapping and now > state["deadline"] + 1e-10:
                self._new(key, state["deadline"], "confirmation_deadline")
        unresolved = {k: s for k, s in self.states.items() if k not in self.mapping}
        for key, state in unresolved.items():
            state["candidates"] = self._candidates(key, state)
        if self.diagnostics:
            self._record_resolution_diagnostics(unresolved, now)
        keys = sorted(k for k, state in unresolved.items() if len(state["gallery"]) >= self.minimum)
        gids = sorted({c["global_id"] for k in keys for c in unresolved[k]["candidates"]})
        if keys and gids:
            # Null cost is the largest admitted combined affinity, not an extra
            # threshold confounded with the appearance-only gate.
            null = self.distance_limit + self.time_weight + 1e-6
            cost = np.full((len(keys), len(gids) + len(keys)), 1e6)
            edges = {}
            for i, key in enumerate(keys):
                cost[i, len(gids) + i] = null
                candidates = unresolved[key]["candidates"]
                for candidate in candidates:
                    j = gids.index(candidate["global_id"])
                    cost[i, j] = candidate["cost"]
                    edges[(i, j)] = candidate
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if j >= len(gids) or (i, j) not in edges:
                    continue
                # Both alternatives for this entrant and competing entrants for
                # this identity must leave the selected edge unambiguous.
                alternative = min((cost[i, q] for q in range(len(gids)) if q != j), default=1e6)
                competitor = min((candidate["cost"] for other_key, state in unresolved.items()
                                  if other_key != keys[i] for candidate in state["candidates"]
                                  if candidate["global_id"] == gids[j]), default=1e6)
                if self.diagnostics:
                    details = unresolved[keys[i]]["diagnostics"]
                    details.update({"selected_global_id": gids[j], "selected_cost": float(cost[i, j]),
                                    "row_gap": (float(alternative - cost[i, j]) if alternative < 1e6 else None),
                                    "column_gap": (float(competitor - cost[i, j]) if competitor < 1e6 else None),
                                    "blocking_checks": [name for name, gap in
                                                        (("row_margin", alternative - cost[i, j]),
                                                         ("column_margin", competitor - cost[i, j]))
                                                        if gap + 1e-10 < self.margin]})
                if min(alternative, competitor) - cost[i, j] + 1e-10 < self.margin:
                    if self.diagnostics:
                        details["decision_status"] = "competition_rejected"
                    continue
                if self.diagnostics:
                    details["decision_status"] = "accepted"
                self._commit(keys[i], gids[j], now, "confirmed_handoff", edges[(i, j)])
        # Even K=0 first considers this tick's actual observation through the
        # same assignment/screening kernel, then creates a fresh fallback ID.
        for key, state in unresolved.items():
            if key not in self.mapping and now >= state["deadline"] - 1e-10:
                self._new(key, state["deadline"], "confirmation_deadline")

    def identity_priority(self, camera, now):
        scores = []
        for key, state in self.states.items():
            if key[0] != camera or key in self.mapping or now > state["deadline"] + 1e-10:
                continue
            if key[1] not in self.active_local.get(camera, set()):
                continue
            candidates = state["candidates"]
            if not candidates:
                continue
            evidence_deficit = max(0., 1 - len(state["gallery"]) / self.minimum)
            ambiguity = 0.
            if len(candidates) > 1:
                gap = candidates[1]["cost"] - candidates[0]["cost"]
                ambiguity = math.exp(-gap / max(self.margin, .05))
            best = candidates[0]
            for other_key, other_state in self.states.items():
                if other_key == key or other_key in self.mapping:
                    continue
                for other in other_state["candidates"]:
                    if other["global_id"] == best["global_id"]:
                        gap = max(0., other["cost"] - best["cost"])
                        ambiguity = max(ambiguity, math.exp(-gap / max(self.margin, .05)))
            # Deadline urgency is a heuristic; it is not a calibrated probability
            # or an estimate of expected information gain.
            urgency = (1. if self.window == 0
                       else min(1., max(0., (now - state["first"]) / self.window)))
            scores.append(min(1., (.25 + .75 * urgency) * max(evidence_deficit, ambiguity)))
        return max(scores, default=0.)

    def observe(self, camera, tracks, now):
        for track in tracks:
            if track.get("age_since_detection", 0) != 0:
                continue
            key = (camera, track["local_id"])
            for index, route in enumerate(self.routes):
                token = (*key, index)
                if route["source"] != camera or token in self.sent or token in self.exit_events:
                    continue
                if not point_in_polygon(track["point"], route["exit_polygon"]):
                    continue
                direction = np.asarray(route.get("direction", [0, 0]))
                if np.linalg.norm(direction) > 0 and np.dot(direction, track["velocity"]) <= 0:
                    continue
                gallery = [f for _, f in self.galleries.get(key, [])]
                self.exit_events[token] = {"track": track, "exit_time_s": now, "gallery": gallery}
        # Pending source identities publish only after their own commit. Travel
        # times use the original observed exit; delivery uses publication time.
        for token, event in list(self.exit_events.items()):
            if token[0] != camera or token[:2] not in self.mapping:
                continue
            route = self.routes[token[2]]
            self.sent.add(token)
            self.serial += 1
            lo, hi = route["travel_time_s"]
            track, exit_time = event["track"], event["exit_time_s"]
            message = {"id": self.serial, "source": camera, "destination": route["destination"],
                       "exit_time_s": exit_time, "begin_s": exit_time + lo, "end_s": exit_time + hi,
                       "global_id": self.mapping[token[:2]], "feature": track["feature"],
                       "confidence": track["score"],
                       "entry_polygon": route["entry_polygon"]}
            if self.source_pooling == "nearest_gallery":
                message["gallery"] = event["gallery"]
            self.messages += 1
            self.bytes += len(canonical(message).encode())
            dropped = self.random.random() < self.loss
            self.log.append({"event": "sent", "time_s": now, "message": message, "dropped": dropped})
            if dropped:
                self.dropped += 1
            else:
                heapq.heappush(self.pending, (now + max(self.delay, 1 / self.fps), self.serial, message))
            del self.exit_events[token]

    def flush(self, now):
        # Terminal rows become available at their deadline, with no future
        # observations/messages consumed. This is explicitly recorded by runner.
        for key, state in sorted(self.states.items()):
            if key not in self.mapping:
                self._new(key, max(now, state["deadline"]), "end_of_segment")
        return self.mapping
