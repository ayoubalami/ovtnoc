"""GT-free, resumable YOLO11m/vehicle-ReID extraction on native CityFlow frames.

The immutable scene.json is the input contract. Raw detector outputs after the
declared confidence/NMS/class filters are retained separately from ROI-filtered
inference observations. No identity annotations or tracker outputs are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from cityflow.data import load_roi, roi_accepts
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json
from vehicle_reid_support import MODEL_SHA256, VehicleReID


VEHICLE_CLASSES = (2, 3, 5, 7)
ROI_RULE = "official corner rule; x,y,width,height truncated separately; exact 255 white; out-of-bounds corners ignored"


def frame_range(metadata: dict, start_s: float, stop_s: float) -> tuple[int, int]:
    """Same rounded microsecond interval convention as the frozen native replay."""
    if not math.isfinite(start_s) or not math.isfinite(stop_s) or not 0 <= start_s < stop_s:
        raise ValueError("Need a finite nonnegative half-open interval")
    fps = Fraction(str(metadata["fps"]))
    offset = Fraction(str(metadata["timestamp_offset_s"]))
    if fps <= 0 or offset < 0 or type(metadata["frames"]) is not int or metadata["frames"] < 0:
        raise ValueError("Invalid native camera clock")

    def lower_bound(seconds):
        target = round(Fraction(str(seconds)) * 1_000_000)
        left, right = 0, metadata["frames"]
        while left < right:
            middle = (left + right) // 2
            if round((offset + middle / fps) * 1_000_000) < target:
                left = middle + 1
            else:
                right = middle
        return left

    return lower_bound(start_s), lower_bound(stop_s)


def next_interval_frame(cache: DetectionCache, camera: int, start: int, stop: int) -> int:
    """Resume a contiguous committed prefix, including intervals starting after 0."""
    retained = [r[0] for r in cache.db.execute(
        "SELECT frame FROM frames WHERE camera=? ORDER BY frame", (camera,))]
    raw = [r[0] for r in cache.db.execute(
        "SELECT frame FROM raw_frames WHERE camera=? ORDER BY frame", (camera,))]
    expected = list(range(start, start + len(retained)))
    if retained != expected or raw != retained or start + len(retained) > stop:
        raise ValueError(f"Cache has gaps, extra frames, or unmatched raw rows for camera {camera}; cannot resume")
    return start + len(retained)


def retained_observations(raw: list[dict], roi: np.ndarray) -> tuple[list[dict], dict]:
    """Keep original detector boxes/scores/classes; apply image-only official ROI."""
    result = []
    for item in raw:
        box = item["box"]
        if (len(box) != 4 or not np.isfinite(box).all() or
                box[2] <= box[0] or box[3] <= box[1] or
                not math.isfinite(item["score"]) or not 0 <= item["score"] <= 1 or
                item["class_id"] not in VEHICLE_CLASSES):
            raise ValueError("Invalid raw vehicle detection")
        if roi_accepts(box, roi):
            result.append({"box": list(box), "score": item["score"], "class_id": item["class_id"]})
    return result, {"raw_post_confidence_nms_class_filter": len(raw),
                    "outside_roi": len(raw) - len(result), "retained": len(result)}


class LocalYOLO:
    """Local weights only, deterministic CPU float32 batch-one inference."""
    def __init__(self, model: Path, threads: int, settings: dict):
        if not model.exists():
            raise FileNotFoundError(f"Local detector weights required; downloads are disabled: {model}")
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        torch.manual_seed(settings["seed"])
        self.model = YOLO(str(model.resolve()), task="detect")
        self.settings = settings
        self.device_argument = "intel:cpu" if model.is_dir() else "cpu"
        # Predictor initialization changes torch's CPU thread count. Initialize
        # it on a synthetic image, then enforce settings before dataset frames.
        self(np.zeros((settings["image_size"], settings["image_size"], 3), dtype=np.uint8))
        torch.set_num_threads(threads)
        self.runtime = {"torch_intraop_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "backend": "openvino" if model.is_dir() else "pytorch",
                        "execution_device": "CPU", "precision": "float32"}
        if model.is_dir():
            import openvino as ov
            backend = self.model.predictor.model
            backend.ov_compiled_model = ov.Core().compile_model(backend.ov_model, "CPU", {
                "INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f32"})
            backend.inference_mode = "LATENCY"
            self.runtime.update({"openvino_inference_threads": int(backend.ov_compiled_model.get_property("INFERENCE_NUM_THREADS")),
                "openvino_precision_hint": str(backend.ov_compiled_model.get_property("INFERENCE_PRECISION_HINT")),
                "openvino_execution_devices": list(backend.ov_compiled_model.get_property("EXECUTION_DEVICES")),
                "openvino_performance_hint": str(backend.ov_compiled_model.get_property("PERFORMANCE_HINT"))})
            if self.runtime["openvino_execution_devices"] != ["CPU"]:
                raise ValueError("Detector must execute on CPU only")
        if torch.get_num_threads() != threads:
            raise ValueError("Unable to enforce requested torch CPU threads")

    def __call__(self, image) -> list[dict]:
        results = self.model.predict(source=image, device=self.device_argument, half=False, batch=1,
            imgsz=self.settings["image_size"], conf=self.settings["confidence"],
            iou=self.settings["nms_iou"], classes=list(VEHICLE_CLASSES),
            max_det=self.settings["maximum_detections"], augment=False,
            agnostic_nms=False, rect=False, verbose=False, save=False, stream=False)
        if len(results) != 1:
            raise ValueError("Expected one detector result for one native video frame")
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        return [{"box": list(map(float, box)), "score": float(score), "class_id": int(class_id)}
                for box, score, class_id in zip(xyxy, confidence, classes)]


def verify_scene_assets(scene: dict):
    """Content hashes and dimensions, without opening any annotation file."""
    if not isinstance(scene.get("cameras"), dict) or not scene["cameras"]:
        raise ValueError("Need full scene.json camera dictionary")
    root = Path(scene["dataset_root"])
    for key, folder, suffix in (("timestamps_sha256", "cam_timestamp", ".txt"),
                                ("framecounts_sha256", "cam_framenum", ".txt")):
        if file_hash(root / folder / (scene["scenario"] + suffix)) != scene[key]:
            raise ValueError(f"Scene source changed: {key}")
    for camera, meta in scene["cameras"].items():
        if int(camera) != meta["camera"]:
            raise ValueError("Scene camera key does not match native metadata")
        for path_key, hash_key in (("video", "video_sha256"), ("roi", "roi_sha256")):
            if file_hash(Path(meta[path_key])) != meta[hash_key]:
                raise ValueError(f"Scene {path_key} changed for camera {camera}")
        roi = load_roi(Path(meta["roi"]))
        if roi.shape != (meta["height"], meta["width"]):
            raise ValueError(f"ROI/video dimension mismatch for camera {camera}")


def detector_assets(model: Path, source_checkpoint: Path | None = None) -> dict:
    if model.is_file() and model.suffix.lower() == ".pt":
        return {"format": "pytorch", "weights_sha256": file_hash(model),
                "source_checkpoint_sha256": file_hash(model)}
    if model.is_dir():
        xmls = list(model.glob("*.xml"))
        if len(xmls) != 1 or not xmls[0].with_suffix(".bin").is_file() or not (model / "metadata.yaml").is_file():
            raise ValueError("Need one local OpenVINO XML/BIN pair and metadata.yaml")
        if source_checkpoint is None or not source_checkpoint.is_file() or source_checkpoint.suffix.lower() != ".pt":
            raise ValueError("OpenVINO detector requires its local source .pt checkpoint for provenance")
        files = [xmls[0], xmls[0].with_suffix(".bin"), model / "metadata.yaml"]
        files += sorted(model.glob("*.json"))
        return {"format": "openvino", "file_sha256": {p.name: file_hash(p) for p in files},
                "source_checkpoint_sha256": file_hash(source_checkpoint)}
    raise FileNotFoundError("A local .pt detector or OpenVINO model directory is required; no silent download")


def provenance(scene_path: Path, scene: dict, model: Path, reid_model: Path,
               start_s: float, stop_s: float, settings: dict, source_checkpoint=None) -> dict:
    assets = detector_assets(model, source_checkpoint)
    if not reid_model.is_file() or file_hash(reid_model) != MODEL_SHA256:
        raise ValueError("ReID weights differ from verified official vehicle-reid-0001 ONNX")
    root = Path(__file__).resolve().parents[1]
    dependencies = [Path(__file__).resolve(), Path(__file__).with_name("vehicle_reid_support.py"),
                    root / "src/cityflow/data.py", root / "src/rhd/cache.py", root / "src/rhd/common.py"]
    return {"kind": "native_cityflow_local_pretrained_yolo_vehicle_reid_v0_6",
        "model": model.name, "detector_assets": assets,
        "source_checkpoint_sha256": assets["source_checkpoint_sha256"],
        "detector_timing_available": True, "descriptor": "vehicle-reid-0001_512_v1",
        "reid_model_sha256": MODEL_SHA256, "scene_json_sha256": file_hash(scene_path),
        "scene_signature": digest(scene), "settings": settings,
        "interval": {"start_s": start_s, "stop_s": stop_s,
                     "native_zero_based_ranges": {c: list(frame_range(m, start_s, stop_s))
                                                   for c, m in scene["cameras"].items()}},
        "roi_filter": ROI_RULE,
        "raw_detection_scope": "All vehicle detections returned after confidence, class filtering, and NMS; pre-threshold model candidates unavailable",
        "feature_scope": "All ROI-retained detections, cropped from their own exact native video frame",
        "source_sha256": {str(p.relative_to(root)): file_hash(p) for p in dependencies},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("ultralytics", "torch", "openvino", "numpy", "opencv-python", "Pillow")},
        "note": "No dataset training or GT-based inference filtering. Video decoding excluded from detector and feature inference timings."}


def cache_statistics(cache: DetectionCache, ranges: dict) -> dict:
    result = {}
    for camera, (start, stop) in ranges.items():
        next_frame = next_interval_frame(cache, int(camera), start, stop)
        totals = Counter()
        for (payload,) in cache.db.execute("SELECT payload FROM raw_frames WHERE camera=?", (int(camera),)):
            raw = json.loads(payload)
            totals["raw_post_confidence_nms_class_filter"] += len(raw)
        for (payload,) in cache.db.execute("SELECT payload FROM frames WHERE camera=?", (int(camera),)):
            ds = json.loads(payload)
            totals["retained"] += len(ds)
            totals["nonzero_descriptors"] += sum(int(np.linalg.norm(d["feature"]) > 0) for d in ds)
        totals["outside_roi"] = totals["raw_post_confidence_nms_class_filter"] - totals["retained"]
        result[camera] = {"start_local_frame": start, "stop_local_frame": stop,
            "next_local_frame": next_frame, "cached_frames": next_frame-start,
            "complete": next_frame == stop, "counts": dict(totals)}
    return result


def extract(scene: dict, cache: DetectionCache, detector, reid, ranges: dict,
            selected: list[int], output: Path, progress_every=50, max_frames_per_camera=None):
    """Injection points support exact-frame and interrupted-resume verification."""
    cache.db.execute("CREATE TABLE IF NOT EXISTS raw_frames (camera INTEGER NOT NULL, frame INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(camera,frame))")
    known = set(map(int, scene["cameras"]))
    stored = {r[0] for r in cache.db.execute("SELECT DISTINCT camera FROM frames UNION SELECT DISTINCT camera FROM raw_frames")}
    if not stored <= known:
        raise ValueError("Cache contains undeclared cameras")
    for c, (start, stop) in ranges.items():
        next_interval_frame(cache, int(c), start, stop)
    cache.commit()
    for camera in selected:
        meta = scene["cameras"][str(camera)]
        start, stop = ranges[str(camera)]
        resume = next_interval_frame(cache, camera, start, stop)
        if resume == stop:
            print(f"camera {camera}: complete [{start},{stop})", flush=True)
            continue
        roi = load_roi(Path(meta["roi"]))
        cap = cv2.VideoCapture(meta["video"])
        begin_session = time.perf_counter()
        calls = 0
        try:
            if not cap.isOpened():
                raise ValueError(f"Unreadable native video: {meta['video']}")
            actual = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            expected = (meta["fps"], meta["frames"], meta["width"], meta["height"])
            if any(abs(a-b) > .001 for a, b in zip(actual, expected)):
                raise ValueError(f"Native video metadata changed for camera {camera}")
            # Seeking compressed video can land on a nearby keyframe. Decode from
            # zero and skip via grab so the inference frame index is exact.
            for _ in range(resume):
                if not cap.grab():
                    raise ValueError(f"Video ended before resume camera {camera}/{resume}")
            for frame in range(resume, stop):
                ok, image = cap.read()
                if not ok or image is None or image.shape[:2] != (meta["height"], meta["width"]):
                    raise ValueError(f"Missing or invalid native frame {camera}/{frame}")
                begin = time.perf_counter()
                raw = detector(image)
                detector_ms = (time.perf_counter()-begin) * 1000
                observations, _ = retained_observations(raw, roi)
                begin = time.perf_counter()
                for item in observations:
                    item["feature"] = reid.descriptor(image, item["box"])
                    vector = np.asarray(item["feature"])
                    if vector.shape != (512,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid native-frame vehicle descriptor")
                    norm = float(np.linalg.norm(vector))
                    if not (norm == 0 or abs(norm-1.) < 1e-5):
                        raise ValueError("Vehicle descriptor must be zero or unit normalized")
                    item["reid_computed"] = True
                appearance_ms = (time.perf_counter()-begin) * 1000
                cache.db.execute("INSERT INTO raw_frames VALUES (?,?,?)", (camera, frame, canonical(raw)))
                cache.add_frame(camera, frame, observations, detector_ms, appearance_ms)
                calls += 1
                if calls % progress_every == 0 or frame == stop-1 or calls == max_frames_per_camera:
                    cache.commit()
                    elapsed = time.perf_counter()-begin_session
                    stats = cache_statistics(cache, ranges)
                    write_json(output.with_suffix(".progress.json"), {"status": "extracting",
                        "camera": camera, "session_detector_calls": calls,
                        "session_elapsed_s": elapsed, "camera_statistics": stats})
                    print(f"camera {camera}: {frame+1}/{stop} native frames; {calls} session calls; elapsed {elapsed:.1f}s", flush=True)
                if max_frames_per_camera is not None and calls >= max_frames_per_camera:
                    break
            cache.commit()
        except Exception:
            cache.db.rollback()
            raise
        finally:
            cap.release()
    statistics = cache_statistics(cache, ranges)
    report = {"status": "complete" if all(v["complete"] for v in statistics.values()) else "partial",
              "camera_statistics": statistics, "cache": str(output),
              "note": "Committed SQLite rows are authoritative; restarting safely skips them. Raw counts are post-threshold/NMS vehicle outputs."}
    write_json(output.with_suffix(".progress.json"), report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--start-s", type=float, required=True)
    p.add_argument("--stop-s", type=float, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reid-model", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, help="Original local .pt checkpoint for an exported OpenVINO detector")
    p.add_argument("--image-size", type=int, default=960)
    p.add_argument("--confidence", type=float, default=.1)
    p.add_argument("--nms-iou", type=float, default=.7)
    p.add_argument("--maximum-detections", type=int, default=300)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cameras", type=int, nargs="+")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--max-frames-per-camera", type=int, help="Resumable invocation limit for profiling; does not change signed inference settings")
    args = p.parse_args()
    if (args.image_size < 32 or args.image_size % 32 or not 0 <= args.confidence <= 1 or
            not 0 < args.nms_iou <= 1 or args.maximum_detections < 1 or args.threads < 1 or
            args.progress_every < 1 or (args.max_frames_per_camera is not None and args.max_frames_per_camera < 1)):
        p.error("Invalid inference or progress settings")
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    selected = sorted(map(int, scene["cameras"])) if args.cameras is None else sorted(args.cameras)
    if len(set(selected)) != len(selected) or not set(selected) <= set(map(int, scene["cameras"])):
        p.error("Unknown or repeated camera")
    ranges = {c: frame_range(m, args.start_s, args.stop_s) for c, m in scene["cameras"].items()}
    verify_scene_assets(scene)
    settings = {"image_size": args.image_size, "confidence": args.confidence,
                "nms_iou": args.nms_iou, "maximum_detections": args.maximum_detections,
                "vehicle_classes": list(VEHICLE_CLASSES), "device": "cpu", "precision": "float32",
                "batch_size": 1, "augment": False, "agnostic_nms": False, "rect": False, "seed": 7,
                "torch_threads": args.threads, "openvino_threads": args.threads}
    signature = provenance(args.scene, scene, args.model, args.reid_model,
                           args.start_s, args.stop_s, settings, args.source_checkpoint)
    with DetectionCache(args.output) as cache:
        cache.ensure_meta("scene", scene)
        cache.ensure_meta("detector", signature)
        for camera, meta in scene["cameras"].items():
            cache.ensure_meta(f"camera:{camera}", meta)
        detector = LocalYOLO(args.model, args.threads, settings)
        cache.ensure_meta("detector_runtime", detector.runtime)
        reid = VehicleReID(args.reid_model, args.threads)
        report = extract(scene, cache, detector, reid, ranges, selected, args.output,
                         args.progress_every, args.max_frames_per_camera)
    print(canonical(report), flush=True)


if __name__ == "__main__":
    main()
