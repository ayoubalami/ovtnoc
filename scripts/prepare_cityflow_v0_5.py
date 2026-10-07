"""Freeze a disclosed prefix-calibrated independent CityFlowV2 scene protocol."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cityflow.data import scene_metadata, annotation_rows, contiguous_spans
from rhd.common import canonical, digest, file_hash, point_in_polygon, write_json


def point(row, scene):
    meta=scene['cameras'][str(row['camera'])];box=row['box']
    return np.array([(box[0]+box[2])/(2*meta['width']),box[3]/meta['height']])


def zone(points,padding=.04):
    points=np.asarray(points);lo=np.maximum(points.min(0)-padding,0);hi=np.minimum(points.max(0)+padding,1)
    if np.any(hi<=lo):raise ValueError('Degenerate endpoint zone')
    return [[float(lo[0]),float(lo[1])],[float(hi[0]),float(lo[1])],[float(hi[0]),float(hi[1])],[float(lo[0]),float(hi[1])]]


def fit_prefix(scene,stop_s=60.,minimum_samples=5):
    rows=annotation_rows(scene,0.,stop_s)
    spans=contiguous_spans(rows)
    pairs=defaultdict(list);exclusions=defaultdict(int)
    for identity,sequence in spans.items():
        for source,destination in zip(sequence,sequence[1:]):
            if source[0]['camera']==destination[0]['camera']:
                exclusions['same_camera']+=1;continue
            if destination[0]['time_s']<=source[-1]['time_s']:
                exclusions['overlap']+=1;continue
            # At least five observed points in each contiguous span; reject
            # trajectories censored by the recording/prefix boundaries.
            if min(len(source),len(destination))<5:
                exclusions['short_span']+=1;continue
            if source[0]['frame']==0 or destination[0]['frame']==0 or destination[-1]['time_s']>=stop_s-.1 or source[-1]['time_s']>=stop_s-.1:
                exclusions['prefix_censored']+=1;continue
            pairs[(source[0]['camera'],destination[0]['camera'])].append((identity,source,destination))
    routes=[];statistics=[]
    for (source_camera,destination_camera),samples in sorted(pairs.items()):
        stats={'source':source_camera,'destination':destination_camera,'prefix_transitions':len(samples)}
        if len(samples)<minimum_samples:
            stats['status']='unsupported_fewer_than_five_prefix_transitions';statistics.append(stats);continue
        exit_polygon=zone([point(s[-1],scene) for _,s,_ in samples])
        entry_polygon=zone([point(d[0],scene) for _,_,d in samples])
        velocity=[]
        for _,s,_ in samples:
            prior=s[max(0,len(s)-6)];last=s[-1]
            velocity.append((point(last,scene)-point(prior,scene))/(last['time_s']-prior['time_s']))
        direction=np.median(velocity,axis=0)
        direction=direction/np.linalg.norm(direction) if np.linalg.norm(direction)>1e-12 else np.zeros(2)
        travel=[];crossings=[]
        for identity,s,d in samples:
            crossing=None
            for i,row in enumerate(s):
                if i==0:continue
                delta=point(row,scene)-point(s[max(0,i-5)],scene)
                if point_in_polygon(point(row,scene),exit_polygon) and (np.linalg.norm(direction)==0 or float(delta@direction)>0):
                    crossing=row;break
            if crossing is not None and point_in_polygon(point(d[0],scene),entry_polygon):
                dt=d[0]['time_s']-crossing['time_s']
                if dt>0:
                    travel.append(dt);crossings.append({'id':identity,'source_frame1':crossing['frame']+1,'destination_frame1':d[0]['frame']+1,'travel_s':dt})
        stats['fitted_crossings']=len(travel)
        if len(travel)<minimum_samples:
            stats['status']='unsupported_fewer_than_five_crossings';statistics.append(stats);continue
        lo,hi=np.quantile(travel,[.05,.95]);lo=max(0.,float(lo)-.1);hi=float(hi)+.1
        routes.append({'source':source_camera,'destination':destination_camera,'exit_polygon':exit_polygon,
                       'entry_polygon':entry_polygon,'direction':direction.tolist(),'travel_time_s':[lo,hi]})
        stats.update({'status':'frozen','median_s':float(np.median(travel)),'travel_time_s':[lo,hi],'crossings':crossings})
        statistics.append(stats)
    if not routes:raise ValueError('No route has sufficient prefix support')
    return routes,{'start_s':0.,'stop_s':stop_s,'minimum_transition_samples':minimum_samples,
        'minimum_contiguous_span_observations':5,'zone_padding_normalized':.04,
        'rule':'adjacent contiguous GT camera spans; positive visibility gap; no boundary-censored spans; first directional zone crossing; 5th/95th travel quantiles +/-0.1s',
        'selected_route_statistics':statistics,'excluded_prefix_spans':dict(exclusions),
        'ground_truth_sha256':{c:file_hash(Path(m['video']).parent/'gt'/'gt.txt') for c,m in scene['cameras'].items()}}


def render_audit(scene,routes,output):
    cameras=sorted(map(int,scene['cameras']));canvas=Image.new('RGB',(1280,390*((len(cameras)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
    for position,camera in enumerate(cameras):
        meta=scene['cameras'][str(camera)];index=max(0,int((30.-meta['timestamp_offset_s'])*meta['fps']))
        cap=cv2.VideoCapture(meta['video'])
        try:
            for _ in range(index):
                if not cap.grab():raise ValueError('Audit frame unavailable')
            ok,image=cap.read()
            if not ok:raise ValueError('Audit frame unavailable')
        finally:cap.release()
        for route in routes:
            for camera_key,polygon_key,color in [('source','exit_polygon',(0,0,255)),('destination','entry_polygon',(0,255,0))]:
                if route[camera_key]!=camera:continue
                poly=np.rint(np.asarray(route[polygon_key])*[meta['width'],meta['height']]).astype(np.int32)
                cv2.polylines(image,[poly],True,color,5)
                cv2.putText(image,f"{route['source']}->{route['destination']} {polygon_key}",tuple(poly[0]),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
        x=(position%2)*640;y=(position//2)*390
        canvas.paste(Image.fromarray(image[:,:,::-1]).resize((640,360)),(x,y+25));draw.text((x+10,y+6),f'c{camera:03d}: prefix frame {index+1}; RED exit / GREEN entry',fill='black')
    canvas.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    scene=scene_metadata(args.dataset,'validation','S02')
    routes,calibration=fit_prefix(scene)
    cfg={'version':'cityflow_v0_5','dataset':scene['dataset_root'],'split':'validation','scenario':'S02',
       'cameras':sorted(map(int,scene['cameras'])),'fps':10.,'seed':7,'replay_start_s':60.,'score_start_s':120.,'stop_s':212.,
       'tracker':{'track_high_thresh':.6,'track_low_thresh':.1,'new_track_thresh':.6,'track_buffer':30,'match_thresh':.9,'fuse_score':True},
       'association':{'kind':'bounded','feature_mode':'ema_observed','source_pooling':'exit_prototype',
                      'max_cosine_distance':.8,'ambiguity_margin':0.,'confirmation_frames':8,'source_retention_frames':8,
                      'minimum_observations':1,'gallery_size':5,'time_weight':0.,'diagnostics':True},
       'communication':{'delay_s':0.,'loss_probability':0.},'output_filter':{'kind':'none'},'routes':routes,
       'route_calibration':calibration,
       'scope':'Independent location from RoundaboutHD with disclosed labeled target-scene prefix calibration. Non-overlap route handoffs only; not zero-shot generalization or full CityFlowV2 benchmark.',
       'perception':'Provided raw Mask R-CNN detections >=0.1, official image-only ROI, vehicle-reid-0001 on every retained native observation. No model training.',
       'controls':{'confirmation_frames':[0,2,4,8],'minimum_observations':1,'source_retention_frames':8,
                   'note':'Only confirmation deadline changes; four methods share detections, descriptors, local tracker, gates, and assignment.'},
       'clock_limit':scene['timing_limit'],'scene_signature':digest(scene)}
    if max(r['travel_time_s'][1] for r in routes)+3.8 > cfg['score_start_s']-cfg['replay_start_s']:
        raise ValueError('Warmup insufficient for fitted route window plus source buffer and deadline')
    args.output.mkdir(parents=True,exist_ok=True);config_path=args.output/'protocol.json'
    if config_path.exists():
        if canonical(json.loads(config_path.read_text())) != canonical(cfg):raise ValueError('Frozen protocol mismatch; use new version directory')
        freeze=json.loads((args.output/'freeze.json').read_text())
        for name,key in [('protocol.json','protocol_sha256'),('scene.json','scene_sha256'),('prefix_zones.png','zones_sha256')]:
            if not (args.output/name).is_file() or file_hash(args.output/name)!=freeze[key]:raise ValueError(f'Frozen protocol artifact changed: {name}')
        if canonical(json.loads((args.output/'scene.json').read_text()))!=canonical(scene):raise ValueError('Frozen scene metadata changed')
        print(f'Frozen protocol verified: {config_path}',flush=True);return
    if any(args.output.iterdir()):raise ValueError('Protocol output must be empty')
    write_json(config_path,cfg);write_json(args.output/'scene.json',scene)
    render_audit(scene,routes,args.output/'prefix_zones.png')
    write_json(args.output/'freeze.json',{'status':'frozen_before_method_scores','protocol_sha256':file_hash(config_path),
      'scene_sha256':file_hash(args.output/'scene.json'),'zones_sha256':file_hash(args.output/'prefix_zones.png'),
      'builder_sha256':file_hash(Path(__file__)),'ground_truth_prefix_only':True,
      'note':'Feasibility annotation audit precedes freeze; detector/tracker/association scores have not been inspected. No tuning on suffix.'})
    print(json.dumps({'protocol':str(config_path),'routes':[(r['source'],r['destination']) for r in routes],
                      'prefix_samples':[(s['source'],s['destination'],s.get('fitted_crossings')) for s in calibration['selected_route_statistics'] if s['status']=='frozen']}),flush=True)


if __name__=='__main__':main()
