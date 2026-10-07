"""Verify every supplied detection and actual ReID sample on a native interval."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from cache_cityflow_mask_interval_v0_6 import next_frame
from cache_cityflow_yolo_v0_6 import frame_range,verify_scene_assets
from cityflow.data import load_roi,raw_detections
from rhd.cache import DetectionCache
from rhd.common import canonical,digest,file_hash,write_json
from vehicle_reid_support import MODEL_SHA256,VehicleReID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene','cache','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args();before = file_hash(args.cache)
    scene = json.loads(args.scene.read_text());verify_scene_assets(scene)
    model = VehicleReID(args.model,4);root = Path(__file__).resolve().parents[1]
    records = []
    with DetectionCache(args.cache,readonly=True) as cache:
        detector = cache.get_meta('detector');interval = cache.get_meta('interval')
        if canonical(cache.get_meta('scene')) != canonical(scene) or interval['scene_sha256'] != file_hash(args.scene):
            raise ValueError('Scene metadata changed')
        if detector['model'] != 'CityFlowV2 supplied mask_rcnn' or detector['reid_model_sha256'] != MODEL_SHA256:
            raise ValueError('Unexpected perception model')
        for relative,expected in {'scripts/cache_cityflow_mask_interval_v0_6.py':detector['builder_sha256'],
                'src/cityflow/data.py':detector['data_reader_sha256'],
                'scripts/vehicle_reid_support.py':detector['appearance_source_sha256'],
                'scripts/cache_cityflow_yolo_v0_6.py':interval['clock_reader_sha256']}.items():
            if file_hash(root/relative) != expected:raise ValueError('Extraction code changed')
        if {row[0] for row in cache.db.execute('SELECT DISTINCT camera FROM frames')} != set(map(int,scene['cameras'])):
            raise ValueError('Cache camera cohort changed')
        for c,meta in scene['cameras'].items():
            camera = int(c);path = Path(meta['video']).parent/'det/det_mask_rcnn.txt'
            if file_hash(path) != detector['raw_file_hashes'][c]:raise ValueError('Supplied raw detections changed')
            if canonical(cache.get_meta(f'camera:{camera}')) != canonical(meta):raise ValueError('Native metadata changed')
            start,stop = frame_range(meta,interval['start_s'],interval['stop_s'])
            if list(interval['native_zero_based_ranges'][c]) != [start,stop] or next_frame(cache,camera,start,stop) != stop:
                raise ValueError('Native interval incomplete')
            raw,_ = raw_detections(path,meta['frames'],load_roi(Path(meta['roi'])),detector['minimum_confidence'])
            targets = [start,(start+stop-1)//2,stop-1];eligible = [];count = 0
            for frame in range(start,stop):
                observed,detector_ms,appearance_ms = cache.frame(camera,frame)
                expected = raw.get(frame,[])
                if len(expected) != len(observed):raise ValueError('Raw detection counts changed')
                if not np.isfinite([detector_ms,appearance_ms]).all() or detector_ms != 0 or appearance_ms < 0:
                    raise ValueError('Invalid supplied-detector inference timing')
                for actual,original in zip(observed,expected):
                    if any(actual[key] != original[key] for key in ('box','score','class_id')):
                        raise ValueError('Raw box/score/class changed')
                    feature = np.asarray(actual['feature'])
                    if feature.shape != (512,) or not np.isfinite(feature).all() or actual.get('reid_computed') is not True or abs(float(np.linalg.norm(feature))-1.) > 1e-5:
                        raise ValueError('Malformed actual observation descriptor')
                    count += 1
                if observed:eligible.append((frame,observed[0]))
            samples = {};used = set()
            for target in targets:
                choices = [item for item in eligible if item[0] not in used]
                if choices:
                    frame,item = min(choices,key=lambda item:(abs(item[0]-target),item[0]));samples[frame] = item;used.add(frame)
            cap = cv2.VideoCapture(meta['video']);errors = []
            try:
                for frame in range(max(samples,default=-1)+1):
                    ok,image = cap.read() if frame in samples else (cap.grab(),None)
                    if not ok:raise ValueError('Sample native frame missing')
                    if frame in samples:
                        item = samples[frame];value = np.asarray(model.descriptor(image,item['box']))
                        error = float(np.max(np.abs(value-np.asarray(item['feature']))))
                        if error > 1e-4:raise ValueError('Descriptor differs from exact native video observation')
                        errors.append(dict(native_frame1=frame+1,local_frame_zero_based=frame,maximum_component_error=error))
            finally:cap.release()
            records.append(dict(camera=camera,native_interval=[start,stop],frames=stop-start,
                                nonzero_descriptors=count,exact_video_samples=errors))
    after = file_hash(args.cache)
    if before != after:raise ValueError('Read-only cache changed during verification')
    report = dict(status='complete',cache_sha256=after,scene_signature=digest(scene),
                  verifier_sha256=file_hash(Path(__file__)),camera_statistics=records,
                  checks=['Every native interval frame explicit once','All supplied ROI-retained boxes/scores/classes unchanged',
                          'Actual 512D descriptors finite/nonzero/unit normalized','Deterministic exact-native-video descriptors recomputed',
                          'Selected video/ROI/raw detector/ReID weights/extraction code unchanged','Cache unchanged during verification'],
                  note='No identity annotations read. Supplied Mask R-CNN inference runtime is unavailable.')
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:raise ValueError('Existing verification differs')
    else:write_json(args.output,report)
    print(canonical(report),flush=True)


if __name__ == '__main__':main()
