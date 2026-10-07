from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .cache import DetectionCache
from .common import digest, file_hash, unit_vector, write_json
from .dataset import find_ground_truth


def color_descriptor(image: np.ndarray, box) -> list[float]:
    """Bootstrap descriptor, not a learned vehicle ReID model."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return [0.0] * 48
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    return unit_vector(hist).tolist()


def discover(dataset: Path) -> list[tuple[int, Path]]:
    dataset = dataset.resolve()
    found = []
    for directory in sorted(dataset.glob("imagesc*")):
        suffix = directory.name.removeprefix("imagesc")
        video = directory / "video.mp4"
        if suffix.isdigit() and video.is_file():
            found.append((int(suffix), video))
    if not found:
        raise FileNotFoundError(f"No imagesc001/video.mp4 layout under {dataset}; pass the extracted RoundaboutHD root")
    return found


def audit(dataset: Path, output: Path | None = None) -> dict:
    videos = []
    for camera, video in discover(dataset):
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open {video}")
            info = {"camera": camera, "path": str(video), "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "sct_present": any((video.parent / "SCT").glob("*.txt")),
                    "geometry_present": (video.parent / "geo-mapping").is_dir()}
            videos.append(info)
        finally:
            cap.release()
    try:
        ground_truth = find_ground_truth(dataset)
    except FileNotFoundError:
        ground_truth = None
    result = {"dataset": str(dataset.resolve()), "videos": videos,
              "ground_truth": str(ground_truth.resolve()) if ground_truth else None,
              "ground_truth_present": ground_truth is not None,
              "timing_note": "Video frame indices are aligned under the dataset's simultaneous-start assumption; inspect any clock offsets separately."}
    if output:
        write_json(output, result)
    return result


def extract(dataset: Path, output: Path, model_path: Path, stop: int | None, device: str, image_size: int, confidence: float, cameras=None):
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}. Supply an existing .pt file; no silent downloads")
    videos = discover(dataset)
    if cameras is not None:
        missing = set(cameras) - {c for c, _ in videos}
        if missing:
            raise ValueError(f"Requested camera videos missing: {sorted(missing)}")
        videos = [(c, p) for c, p in videos if c in cameras]
    settings = {"model_sha256": file_hash(model_path), "model": model_path.name, "image_size": image_size,
                "confidence": confidence, "device": device, "classes": [2, 3, 5, 7], "iou": 0.7,
                "descriptor": "hsv_12x4_v1", "ultralytics": importlib.metadata.version("ultralytics"),
                "opencv": cv2.__version__, "kind": "real_video_detector_cache"}
    with DetectionCache(output) as cache:
        cache.ensure_meta("detector", settings)
        detector = YOLO(str(model_path))
        for camera, video in videos:
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Cannot open {video}")
                meta = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "video_sha256": file_hash(video)}
                cache.ensure_meta(f"camera:{camera}", meta)
                end = min(stop, meta["frames"]) if stop is not None else meta["frames"]
                start = cache.next_frame(camera)
                if start >= end:
                    print(f"camera {camera}: already cached through frame {start-1}", flush=True)
                    continue
                # Decode from zero so resume does not rely on codec-specific random seek accuracy.
                for _ in range(start):
                    if not cap.grab():
                        raise ValueError(f"Video ended before resume point {start}")
                for frame in range(start, end):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Video ended at camera {camera}, frame {frame}")
                    begin = time.perf_counter()
                    result = detector.predict(image, imgsz=image_size, conf=confidence, classes=settings["classes"],
                                              iou=0.7, device=device, verbose=False, rect=False)[0]
                    boxes = result.boxes.data.cpu().numpy()
                    detector_ms = (time.perf_counter() - begin) * 1000
                    begin = time.perf_counter()
                    detections = [{"box": row[:4].tolist(), "score": float(row[4]), "class_id": int(row[5]),
                                   "feature": color_descriptor(image, row[:4])} for row in boxes]
                    cache.add_frame(camera, frame, detections, detector_ms, (time.perf_counter() - begin) * 1000)
                    if (frame + 1) % 200 == 0 or frame == end - 1:
                        cache.commit()
                        print(f"camera {camera}: cached {frame+1}/{end} frames", flush=True)
            finally:
                cap.release()


class CameraTracker:
    """Use upstream ByteTrack on detector frames; predict without fake misses on skipped frames."""

    def __init__(self, fps: float, width: int, height: int, options: dict, output_filter=None):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from .quality import MovingVehicleFilter

        defaults = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                        track_buffer=30, match_thresh=0.8, fuse_score=True)
        unknown = set(options) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown tracker options: {sorted(unknown)}")
        defaults.update(options)
        self.tracker = BYTETracker(SimpleNamespace(**defaults), frame_rate=fps)
        self.width, self.height = width, height
        self.features: dict[int, np.ndarray] = {}
        self.observation_features: dict[int, np.ndarray] = {}
        self.last_observed: dict[int, int] = {}
        self.output_filter = MovingVehicleFilter(output_filter, width, height)

    def step(self, detections: list | None, frame: int) -> list[dict]:
        from ultralytics.engine.results import Boxes

        if detections is None:
            self.tracker.frame_id += 1
            pool = self.tracker.joint_stracks(self.tracker.tracked_stracks, self.tracker.lost_stracks)
            self.tracker.multi_predict(pool)
        else:
            array = np.asarray([d["box"] + [d["score"], d["class_id"]] for d in detections], dtype=np.float32).reshape(-1, 6)
            self.tracker.update(Boxes(array, (self.height, self.width)))
            # ByteTrack stores the original detection index in each updated STrack.
            for track in self.tracker.tracked_stracks:
                if not track.is_activated:
                    continue
                index = int(track.idx)
                if 0 <= index < len(detections):
                    current = unit_vector(detections[index]["feature"])
                    self.observation_features[track.track_id] = current
                    old = self.features.get(track.track_id)
                    self.features[track.track_id] = unit_vector(0.8 * old + 0.2 * current) if old is not None else current
                    self.last_observed[track.track_id] = frame
        snapshots = []
        for track in self.tracker.tracked_stracks:
            if not track.is_activated:
                continue
            box = np.asarray(track.xyxy, dtype=float)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, self.width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, self.height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            age = frame - self.last_observed.get(track.track_id, frame)
            mean = track.mean
            snapshots.append({"local_id": int(track.track_id), "box": box.tolist(), "score": float(track.score),
                              "feature": self.features.get(track.track_id, np.zeros(48)).tolist(),
                              "observation_feature": (self.observation_features.get(track.track_id, np.zeros_like(self.features.get(track.track_id, np.zeros(48))))
                                                      if age == 0 else np.zeros_like(self.features.get(track.track_id, np.zeros(48)))).tolist(),
                              "point": [(box[0] + box[2]) / (2 * self.width), box[3] / self.height],
                              "velocity": [float(mean[4] / self.width), float(mean[5] / self.height)],
                              "uncertainty": float(np.trace(track.covariance[:2, :2]) / max((box[2] - box[0]) ** 2, 1)),
                              "age_since_detection": age})
        live_ids = {t.track_id for t in self.tracker.tracked_stracks + self.tracker.lost_stracks}
        self.features = {k: v for k, v in self.features.items() if k in live_ids}
        self.observation_features = {k: v for k, v in self.observation_features.items() if k in live_ids}
        self.last_observed = {k: v for k, v in self.last_observed.items() if k in live_ids}
        return self.output_filter.apply(snapshots, frame, live_ids)


class LiveSource:
    """Sequential desktop video source; run inference only when the policy requests it."""

    def __init__(self, dataset: Path, model_path: Path, device="cpu", image_size=640, confidence=0.1):
        self.dataset, self.model_path = Path(dataset), Path(model_path)
        self.device, self.image_size, self.confidence = device, image_size, confidence
        self.caps, self.metadata, self.images, self.indices = {}, {}, {}, {}
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing pretrained model: {model_path}")
        if image_size < 32 or not 0 < confidence <= 1:
            raise ValueError("invalid live image size/confidence")

    def __enter__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        self.settings = {"kind": "real_video_live", "model_sha256": file_hash(self.model_path),
                         "device": self.device, "image_size": self.image_size, "confidence": self.confidence,
                         "classes": [2, 3, 5, 7], "descriptor": "hsv_12x4_v1", "iou": 0.7,
                         "ultralytics": importlib.metadata.version("ultralytics"), "opencv": cv2.__version__}
        return self

    def __exit__(self, *args):
        for cap in self.caps.values():
            cap.release()

    def validate_range(self, cameras, start, stop, fps):
        paths = dict(discover(self.dataset))
        for camera in cameras:
            if camera not in paths:
                raise ValueError(f"Camera {camera} video missing")
            video = paths[camera]
            cap = cv2.VideoCapture(str(video))
            self.caps[camera] = cap
            if not cap.isOpened():
                raise ValueError(f"Cannot open {video}")
            meta = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "video_sha256": file_hash(video)}
            if abs(meta["fps"] - fps) > 0.01 or stop > meta["frames"]:
                raise ValueError(f"Camera {camera} FPS/range does not match config")
            self.metadata[camera] = meta
            for _ in range(start):
                if not cap.grab():
                    raise ValueError(f"Cannot advance camera {camera} to start frame")
            self.indices[camera] = start - 1

    def get_meta(self, key):
        return self.settings if key == "detector" else self.metadata[int(key.split(":")[1])]

    def signature(self, cameras, start, stop):
        return digest({"detector": self.settings, "cameras": self.metadata, "range": [start, stop]})

    def advance(self, camera, frame):
        if frame != self.indices[camera] + 1:
            raise ValueError("Live replay must advance sequentially")
        ok, image = self.caps[camera].read()
        if not ok:
            raise ValueError(f"Camera {camera} ended at frame {frame}")
        self.indices[camera] = frame
        self.images[camera] = image

    def frame(self, camera, frame):
        if self.indices[camera] != frame:
            raise ValueError("Requested live detector frame has not been decoded")
        image = self.images[camera]
        begin = time.perf_counter()
        result = self.model.predict(image, imgsz=self.image_size, conf=self.confidence, classes=[2, 3, 5, 7],
                                    iou=0.7, device=self.device, verbose=False, rect=False)[0]
        boxes = result.boxes.data.cpu().numpy()
        detector_ms = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        detections = [{"box": b[:4].tolist(), "score": float(b[4]), "class_id": int(b[5]),
                       "feature": color_descriptor(image, b[:4])} for b in boxes]
        return detections, detector_ms, (time.perf_counter() - begin) * 1000
