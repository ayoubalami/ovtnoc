"""Cache supplied Mask R-CNN and actual ReID on an explicit native interval."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
from cache_cityflow_yolo_v0_6 import frame_range, verify_scene_assets
from cityflow.data import load_roi, raw_detections
from rhd.cache import DetectionCache
from rhd.common import canonical,digest,file_hash,write_json
from vehicle_reid_support import VehicleReID,MODEL_SHA256


def next_frame(cache,camera,start,stop):
    frames = [row[0] for row in cache.db.execute('SELECT frame FROM frames WHERE camera=? ORDER BY frame',(camera,))]
    if frames != list(range(start,start+len(frames))) or start+len(frames) > stop:
        raise ValueError('Native interval cache has missing/extra frames')
    return start+len(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start-s',type=float,required=True);p.add_argument('--stop-s',type=float,required=True)
    p.add_argument('--minimum-score',type=float,default=.1);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-frames-per-camera',type=int)
    args = p.parse_args()
    if not 0 < args.minimum_score <= 1 or args.threads < 1:
        p.error('Invalid confidence or thread count')
    scene = json.loads(args.scene.read_text());verify_scene_assets(scene)
    inputs = {int(c):Path(m['video']).parent/'det/det_mask_rcnn.txt' for c,m in scene['cameras'].items()}
    root = Path(__file__).resolve().parents[1]
    provenance = dict(kind='provided_raw_cityflow_detector_cache',model='CityFlowV2 supplied mask_rcnn',
        detector_timing_available=False,minimum_confidence=args.minimum_score,
        roi_filter='official corner rule with separately truncated x,y,width,height; exact 255 white; no identity filtering',
        raw_file_hashes={str(c):file_hash(path) for c,path in inputs.items()},
        descriptor='vehicle-reid-0001_512_v1',reid_model_sha256=MODEL_SHA256,
        feature_scope='every retained detection on its own native video frame; no region/cadence restriction',
        builder_sha256=file_hash(Path(__file__)),data_reader_sha256=file_hash(root/'src/cityflow/data.py'),
        appearance_source_sha256=file_hash(Path(__file__).with_name('vehicle_reid_support.py')),
        openvino=importlib.metadata.version('openvino'),opencv=cv2.__version__,pillow=importlib.metadata.version('pillow'),
        threads=args.threads,
        note='No detector or ReID training on CityFlowV2; supplied raw detections, not GT/MTSC outputs. Cache timing excludes supplied detector inference.')
    ranges = {c:frame_range(m,args.start_s,args.stop_s) for c,m in scene['cameras'].items()}
    interval = dict(start_s=args.start_s,stop_s=args.stop_s,native_zero_based_ranges=ranges,
                    scene_sha256=file_hash(args.scene),clock_reader_sha256=file_hash(Path(__file__).with_name('cache_cityflow_yolo_v0_6.py')))
    model = VehicleReID(args.model,args.threads)
    with DetectionCache(args.output) as cache:
        cache.ensure_meta('scene',scene);cache.ensure_meta('detector',provenance);cache.ensure_meta('interval',interval)
        for c,meta in scene['cameras'].items():
            cache.ensure_meta(f'camera:{c}',meta)
        stored = {row[0] for row in cache.db.execute('SELECT DISTINCT camera FROM frames')}
        if not stored <= set(map(int,scene['cameras'])):
            raise ValueError('Undeclared cached camera')
        for c,meta in scene['cameras'].items():
            camera = int(c); start,stop = ranges[c]
            resume = next_frame(cache,camera,start,stop)
            if resume == stop:
                print(f'camera {camera}: complete [{start},{stop})',flush=True);continue
            ds,counts = raw_detections(inputs[camera],meta['frames'],load_roi(Path(meta['roi'])),args.minimum_score)
            cache.ensure_meta(f'detection_counts:{camera}',counts)
            cap = cv2.VideoCapture(meta['video']);begin_session = time.perf_counter();calls = 0
            try:
                actual = (cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                expected = (meta['fps'],meta['frames'],meta['width'],meta['height'])
                if not cap.isOpened() or any(abs(a-b)>.001 for a,b in zip(actual,expected)):
                    raise ValueError('Native video metadata changed')
                for _ in range(resume):
                    if not cap.grab():raise ValueError('Video ended before native resume')
                for frame in range(resume,stop):
                    observed = ds.get(frame,[])
                    ok,image = cap.read() if observed else (cap.grab(),None)
                    if not ok:raise ValueError('Native frame unavailable')
                    begin = time.perf_counter()
                    for item in observed:
                        item['feature'] = model.descriptor(image,item['box']);item['reid_computed'] = True
                        norm = float(np.linalg.norm(item['feature']))
                        if abs(norm-1.) > 1e-5:raise ValueError('Malformed observation feature')
                    cache.add_frame(camera,frame,observed,0.,1000*(time.perf_counter()-begin));calls += 1
                    if calls%100 == 0 or frame == stop-1 or calls == args.max_frames_per_camera:
                        cache.commit()
                        status = dict(status='extracting',camera=camera,next_local_frame=frame+1,
                            start_local_frame=start,stop_local_frame=stop,elapsed_s=time.perf_counter()-begin_session)
                        write_json(args.output.with_suffix('.progress.json'),status)
                        print(canonical(status),flush=True)
                    if calls == args.max_frames_per_camera:break
            except Exception:
                cache.db.rollback();raise
            finally:cap.release()
        records = {c:dict(start=start,stop=stop,next_frame=next_frame(cache,int(c),start,stop))
                   for c,(start,stop) in ranges.items()}
        complete = all(m['next_frame'] == m['stop'] for m in records.values())
        write_json(args.output.with_suffix('.progress.json'),dict(status='complete' if complete else 'partial',cameras=records))
        print(f'Mask R-CNN native interval cache complete={complete}: {args.output}',flush=True)


if __name__ == '__main__':main()
