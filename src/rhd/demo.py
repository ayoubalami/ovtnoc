from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .cache import DetectionCache
from .common import load_config, write_json
from .runner import sweep
from .vision import color_descriptor


def generate(output: Path) -> tuple[dict, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"cameras": [1, 2], "fps": 15.0, "start_frame": 0, "stop_frame": 150, "seed": 7,
           "routes": [{"source": 1, "destination": 2,
                       "exit_polygon": [[0.78, 0], [1, 0], [1, 1], [0.78, 1]],
                       "entry_polygon": [[0, 0], [0.35, 0], [0.35, 1], [0, 1]],
                       "direction": [1, 0], "travel_time_s": [0.7, 1.8]}],
           "tracker": {"match_thresh": 0.9}, "scheduler": {"max_detection_gap": 15, "handoff_weight": 3.0},
           "association": {"max_cosine_distance": 0.2, "ambiguity_margin": 0.05},
           "communication": {"delay_s": 0.0, "loss_probability": 0.0}}
    write_json(output / "config.json", cfg)
    cache_path = output / "synthetic.sqlite"
    gt_path = output / "ground_truth.txt"
    if cache_path.exists() or gt_path.exists():
        raise ValueError("Synthetic input output already exists; use a new demo folder")
    random = np.random.default_rng(7)
    with DetectionCache(cache_path) as cache, gt_path.open("w", encoding="utf-8") as gt:
        cache.ensure_meta("detector", {"kind": "synthetic_detector_responses", "generator": "two_camera_v1", "seed": 7})
        for camera in cfg["cameras"]:
            cache.ensure_meta(f"camera:{camera}", {"fps": 15.0, "frames": 150, "width": 320, "height": 180})
            for frame in range(150):
                image = np.full((180, 320, 3), 30, dtype=np.uint8)
                detections = []
                for identity, shift, color, y in ((1, 0.0, (20, 30, 220), 60), (2, 1.4, (220, 30, 20), 120)):
                    time = frame / 15 - shift - (3.0 if camera == 2 else 0.0)
                    if not 0 <= time < 2.5:
                        continue
                    x = 8 + 105 * time
                    box = [x, y, x + 40, y + 24]
                    cv2.rectangle(image, (int(x), y), (int(x + 40), y + 24), color, -1)
                    gt.write(f"{camera} {identity} {frame} {x:.6f} {y} 40 24 -1 -1\n")
                    noisy = np.asarray(box) + random.normal(0, 0.35, 4)
                    detections.append({"box": noisy.tolist(), "score": 0.92, "class_id": 2,
                                       "feature": color_descriptor(image, noisy)})
                cache.add_frame(camera, frame, detections, detector_ms=0.0, appearance_ms=0.0)
            cache.commit()
    return load_config(output / "config.json"), cache_path, gt_path


def demo(output: Path):
    cfg, cache, ground_truth = generate(output / "inputs")
    return sweep(cfg, cache, ground_truth, output / "experiments", budgets=(0.5,), intervals=(3,))

