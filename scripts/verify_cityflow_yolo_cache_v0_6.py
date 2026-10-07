"""Verify native YOLO cache, raw/ROI preservation, and exact-video ReID samples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from cache_cityflow_yolo_v0_6 import detector_assets, frame_range, retained_observations, verify_scene_assets
from cityflow.data import load_roi
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json
from vehicle_reid_support import MODEL_SHA256, VehicleReID


def validate_frame(raw, observations, roi):
    expected, counts = retained_observations(raw, roi)
    if len(expected) != len(observations):
        raise ValueError('Raw-to-ROI detection counts differ')
    for original, actual in zip(expected, observations):
        if any(actual.get(key) != original[key] for key in ('box','score','class_id')):
            raise ValueError('Retained raw detector box/score/class changed')
        feature = np.asarray(actual.get('feature'))
        if feature.shape != (512,) or not np.isfinite(feature).all() or actual.get('reid_computed') is not True:
            raise ValueError('Malformed actual observation descriptor')
        if abs(float(np.linalg.norm(feature))-1.) > 1e-5:
            raise ValueError('Descriptor must be nonzero and L2 normalized')
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene','cache','reid-model','detector-model','source-checkpoint','output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    before = file_hash(args.cache)
    scene = json.loads(args.scene.read_text())
    verify_scene_assets(scene)
    reid = VehicleReID(args.reid_model, 4)
    root = Path(__file__).resolve().parents[1]
    records = []
    with DetectionCache(args.cache, readonly=True) as cache:
        detector = cache.get_meta('detector')
        if detector.get('kind') != 'native_cityflow_local_pretrained_yolo_vehicle_reid_v0_6':
            raise ValueError('Unexpected detector cache')
        if digest(cache.get_meta('scene')) != digest(scene) or detector['scene_json_sha256'] != file_hash(args.scene):
            raise ValueError('Scene metadata changed')
        if detector['reid_model_sha256'] != MODEL_SHA256 or file_hash(args.reid_model) != MODEL_SHA256:
            raise ValueError('Unexpected ReID model')
        if detector_assets(args.detector_model, args.source_checkpoint) != detector['detector_assets']:
            raise ValueError('Detector checkpoint/export changed')
        for relative, expected in detector['source_sha256'].items():
            if file_hash(root/relative) != expected:
                raise ValueError(f'Extraction source changed: {relative}')
        expected_cameras = set(map(int, scene['cameras']))
        for table in ('frames','raw_frames'):
            actual = {row[0] for row in cache.db.execute(f'SELECT DISTINCT camera FROM {table}')}
            if actual != expected_cameras:
                raise ValueError('Cache contains missing/undeclared cameras')
        interval = detector['interval']
        for camera_text, meta in scene['cameras'].items():
            camera = int(camera_text)
            if canonical(cache.get_meta(f'camera:{camera}')) != canonical(meta):
                raise ValueError('Camera metadata changed')
            start, stop = frame_range(meta, interval['start_s'], interval['stop_s'])
            if interval['native_zero_based_ranges'][camera_text] != [start,stop]:
                raise ValueError('Recorded native range changed')
            for table in ('frames','raw_frames'):
                actual = [row[0] for row in cache.db.execute(f'SELECT frame FROM {table} WHERE camera=? ORDER BY frame',(camera,))]
                if actual != list(range(start,stop)):
                    raise ValueError('Missing/extra native frame or raw observation')
            roi = load_roi(Path(meta['roi']))
            samples, totals = [], dict(raw=0,retained=0,outside_roi=0)
            targets = [start,(start+stop-1)//2,stop-1]
            eligible = []
            for frame in range(start,stop):
                raw = json.loads(cache.db.execute('SELECT payload FROM raw_frames WHERE camera=? AND frame=?',(camera,frame)).fetchone()[0])
                observations, detector_ms, appearance_ms = cache.frame(camera,frame)
                if not np.isfinite([detector_ms,appearance_ms]).all() or min(detector_ms,appearance_ms) < 0:
                    raise ValueError('Invalid recorded inference time')
                counts = validate_frame(raw,observations,roi)
                totals['raw'] += counts['raw_post_confidence_nms_class_filter']
                totals['retained'] += counts['retained']; totals['outside_roi'] += counts['outside_roi']
                if observations:
                    eligible.append((frame,observations[0]))
            used = set()
            for target in targets:
                candidates = [item for item in eligible if item[0] not in used]
                if candidates:
                    selected = min(candidates,key=lambda item:(abs(item[0]-target),item[0]))
                    samples.append(selected); used.add(selected[0])
            sample_map = dict(samples)
            cap = cv2.VideoCapture(meta['video']); errors = []
            try:
                for frame in range(max(sample_map,default=-1)+1):
                    ok,image = cap.read() if frame in sample_map else (cap.grab(),None)
                    if not ok:
                        raise ValueError('Native video frame unavailable for sample')
                    if frame in sample_map:
                        observed = sample_map[frame]
                        recomputed = np.asarray(reid.descriptor(image,observed['box']))
                        error = float(np.max(np.abs(recomputed-np.asarray(observed['feature']))))
                        if error > 1e-4:
                            raise ValueError('Descriptor disagrees with exact native video observation')
                        errors.append(dict(local_frame_zero_based=frame,native_frame1=frame+1,
                                           maximum_component_error=error))
            finally:
                cap.release()
            records.append(dict(camera=camera,frames=stop-start,native_interval=[start,stop],
                                counts=totals,nonzero_descriptors=totals['retained'],exact_video_samples=errors))
    after = file_hash(args.cache)
    if before != after:
        raise ValueError('Cache changed during read-only verification')
    report = dict(status='complete',cache_sha256=after,scene_signature=digest(scene),
                  verifier_sha256=file_hash(Path(__file__)),reid_model_sha256=MODEL_SHA256,
                  detector_assets=detector['detector_assets'],camera_statistics=records,
                  checks=['Every declared native frame/raw row explicit once','All raw ROI-retained boxes/scores/classes preserved',
                          'All descriptors finite,512D,nonzero,unit normalized','Actual descriptors recomputed on deterministic native video samples',
                          'Video/ROI/checkpoint/export/extraction sources unchanged','Read-only cache hash unchanged'],
                  note='No identity annotations read. Raw values are preserved outputs after detector confidence/class/NMS filters; detector itself is not rerun here.')
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:
            raise ValueError('Existing verification differs; use new output')
    else:
        write_json(args.output,report)
    print(canonical(report),flush=True)


if __name__ == '__main__':
    main()
