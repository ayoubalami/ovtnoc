"""Separate resumable appearance cache for a declared later frame interval."""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
import time

import cv2
from rhd.cache import DetectionCache
from rhd.common import file_hash, load_config, write_json
from rhd.vision import discover
from benchmark_support import ArchiveSource
from vehicle_reid_support import VehicleReID, route_regions, in_regions, MODEL_SHA256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/RoundaboutHD"))
    p.add_argument("--config", type=Path, default=Path("configs/handoff_pair_development_fitted.json"))
    p.add_argument("--model", type=Path, default=Path("models/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx"))
    p.add_argument("--stop-frame", type=int, default=1800)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--padding", type=float, default=.025)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("caches/author_yolov12x_vehicle_reid_pair_v0_1.sqlite"))
    args = p.parse_args()
    if args.start_frame < 0 or args.stop_frame <= args.start_frame:
        raise ValueError("Invalid explicit cache interval")
    cfg = load_config(args.config)
    if cfg.get("phase") != "development":
        raise ValueError("Appearance regions must be defined from development geometry")
    regions = route_regions(cfg, args.padding)
    videos = dict(discover(args.dataset))
    source = ArchiveSource(args.dataset, {"cameras": cfg["cameras"], "start_frame": args.start_frame, "stop_frame": args.stop_frame})
    model = VehicleReID(args.model, args.threads)
    provenance = {"kind": "provided_raw_detector_cache", "model": "author supplied YOLOv12x detections",
        "source": source.provenance, "descriptor": "vehicle-reid-0001_512_v1", "reid_model_sha256": MODEL_SHA256,
        "detector_timing_available": False, "opencv": cv2.__version__, "pillow": importlib.metadata.version("pillow"),
        "openvino": importlib.metadata.version("openvino"), "threads": args.threads,
        "builder_sha256": file_hash(Path(__file__)), "appearance_source_sha256": file_hash(Path(__file__).with_name("vehicle_reid_support.py")),
        "appearance_regions": regions, "region_padding": args.padding,
        "region_definition": "padded development-zone bounding rectangles, tested on normalized detection bottom center",
        "preprocessing": "clipped rounded crop; BGR to RGB; Pillow bilinear 208x208; NCHW float32 0..255; no external normalization; output L2 normalization",
        "note": "VeRi-776-trained vehicle model; no RoundaboutHD training. Raw labels_xy only. Each feature uses only its corresponding video frame. Outside regions: zero 512-vector; tracker retains earlier nonzero descriptor. Detection weights/settings/timing unavailable."}
    try:
        with DetectionCache(args.output) as cache:
            cache.ensure_meta("detector", provenance)
            cache.ensure_meta("declared_interval", [args.start_frame, args.stop_frame])
            for camera in cfg["cameras"]:
                video = videos[camera]
                cap = cv2.VideoCapture(str(video))
                try:
                    if not cap.isOpened():
                        raise ValueError(f"Cannot open {video}")
                    meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS), "video_sha256": file_hash(video)}
                    cache.ensure_meta(f"camera:{camera}", meta)
                    count, first, last = cache.db.execute("SELECT COUNT(*), MIN(frame), MAX(frame) FROM frames WHERE camera=?", (camera,)).fetchone()
                    if count and (first != args.start_frame or count != last-first+1):
                        raise ValueError("Range cache is not contiguous from its declared start")
                    start = args.start_frame + count
                    if args.stop_frame > meta["frames"]:
                        raise ValueError("Requested cache exceeds video length")
                    if start >= args.stop_frame:
                        print(f"camera {camera}: verified complete through {start-1}", flush=True)
                        continue
                    for _ in range(start):
                        if not cap.grab():
                            raise ValueError("Video ended before resume point")
                    calls = 0
                    for frame in range(start, args.stop_frame):
                        detections = source.frame(camera, frame)
                        selected = [i for i, d in enumerate(detections) if in_regions(d["box"], meta["width"], meta["height"], regions[camera])]
                        if selected:
                            ok, image = cap.read()
                        else:
                            ok, image = cap.grab(), None
                        if not ok:
                            raise ValueError(f"Missing exact video frame {camera}/{frame}")
                        begin = time.perf_counter()
                        for d in detections:
                            d["feature"] = [0.] * 512
                            d["reid_computed"] = False
                        for i in selected:
                            detections[i]["feature"] = model.descriptor(image, detections[i]["box"])
                            detections[i]["reid_computed"] = True
                        calls += len(selected)
                        cache.add_frame(camera, frame, detections, detector_ms=0., appearance_ms=(time.perf_counter()-begin)*1000)
                        if (frame+1) % 200 == 0 or frame == args.stop_frame-1:
                            cache.commit()
                            write_json(args.output.with_suffix(".progress.json"), {"last_camera": camera, "next_frame": frame+1,
                                "session_appearance_calls": calls, "provenance": provenance,
                                "note": "Cache is authoritative for resume; session count excludes any earlier invocation."})
                            print(f"camera {camera}: cached {frame+1}/{args.stop_frame}; session vehicle embeddings={calls}", flush=True)
                finally:
                    cap.release()
    finally:
        source.close()


if __name__ == "__main__":
    main()
