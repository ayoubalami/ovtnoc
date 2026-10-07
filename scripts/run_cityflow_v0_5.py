"""Execute all frozen S02 controls and score the held-out suffix, with honest scope."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter
from statistics import median

from cityflow.data import annotation_rows
from cityflow.evaluation import evaluate_predictions, export_official_predictions, prepare_rois
from cityflow.handoffs import evaluate_handoffs
from cityflow.replay import run, source_signature
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json


def attempt_directory(root,tag):
    directory=root/tag;attempt=1
    while (directory/'manifest.json').is_file():
        state=json.loads((directory/'manifest.json').read_text())
        if state.get('status')=='complete':return directory
        attempt+=1;directory=root/f'{tag}_attempt{attempt}'
    return directory


def pending_statistics(directory,score_start,score_stop):
    records=[json.loads(line) for line in (directory/'messages.jsonl').read_text().splitlines()]
    commits=[r for r in records if r.get('event')=='identity_committed' and 'first_time_s' in r
             and score_start<=r['first_time_s']<score_stop]
    waits=sorted(r['time_s']-r['first_time_s'] for r in commits)
    reasons=Counter(r.get('diagnostics',{}).get('confirmation_outcome','unreported') for r in commits)
    return {'scored_pending_tracks':len(commits),'confirmation_outcomes':dict(reasons),
            'median_wait_s':median(waits) if waits else None,
            'maximum_wait_s':max(waits) if waits else None,
            'note':'Actual decisions for entrants first observed in the score interval; acceptance is not GT correctness.'}


def score(cfg,scene,directory,ground_truth,gt_rows,rois):
    source=directory/'predictions.txt';official=directory/'official_predictions.txt'
    if official.exists():
        check=directory/'official_predictions.verify.tmp'
        if check.exists():check.unlink()
        try:
            export_official_predictions(source,check,cameras=cfg['cameras'])
            if file_hash(check)!=file_hash(official):raise ValueError('Previously exported official predictions changed')
        finally:
            if check.exists():check.unlink()
    else:export_official_predictions(source,official,cameras=cfg['cameras'])
    clocks={int(c):{'fps':m['fps'],'offset_s':m['timestamp_offset_s']} for c,m in scene['cameras'].items()}
    metrics=evaluate_predictions(official_eval_path=Path(cfg['dataset'])/'eval'/'eval.py',
      ground_truth_path=ground_truth,prediction_path=official,roi_directory=rois,split=cfg['split'],
      cameras=cfg['cameras'],score_interval_s=[cfg['score_start_s'],cfg['stop_s']],camera_clocks=clocks)
    metrics['handoffs']=evaluate_handoffs(scene,gt_rows,source,routes=cfg['routes'],
      replay_start_s=cfg['replay_start_s'],score_start_s=cfg['score_start_s'],score_stop_s=cfg['stop_s'],
      availability=directory/'availability.jsonl' if (directory/'availability.jsonl').is_file() else None)
    write_json(directory/'metrics.json',metrics)
    return metrics


def write_local_control(source,output):
    output.mkdir(parents=True,exist_ok=True)
    lines=[]
    for line in source.read_text().splitlines():
        a=line.split();a[1]=str(int(a[0])*1_000_000_000+int(a[1]));lines.append(' '.join(a))
    text='\n'.join(lines)+('\n' if lines else '')
    path=output/'predictions.txt'
    if path.exists() and path.read_text()!=text:raise ValueError('Local control changed')
    path.write_text(text)
    write_json(output/'summary.json',{'method':'camera_local_ID_control','shared_local_source_sha256':file_hash(source),
               'note':'No cross-camera association. Official singleton filtering may remove every row; unfiltered score retains them.'})


def build_report(records,cfg,output):
    keys=['method','confirmation_frames','deadline_s','official_IDF1','roi_only_IDF1','unfiltered_IDF1',
          'reference_handoffs','confirmed_handoffs','handoff_recall','detector_calls','median_pending_wait_s']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    lines=['# CityFlowV2 S02: calibrated independent-scene experiment (v0.5)','',
      cfg['scope'],'',f"Calibration [0,60) s; replay [{cfg['replay_start_s']:g},{cfg['stop_s']:g}) s; score [{cfg['score_start_s']:g},{cfg['stop_s']:g}) s. Camera-local frame numbers and offsets are preserved.",'',
      f"Frozen non-overlap routes: {', '.join(str(r['source'])+' -> '+str(r['destination']) for r in cfg['routes'])}.",'',
      cfg['perception'],'',
      '| Method | Deadline (s) | Official IDF1 (%) | ROI-only IDF1 (%) | Unfiltered IDF1 (%) | Correct/reference route handoffs |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        deadline='--' if r['deadline_s'] is None else f"{r['deadline_s']:.1f}"
        lines.append(f"| {r['method']} | {deadline} | {100*r['official_IDF1']:.2f} | {100*r['roi_only_IDF1']:.2f} | {100*r['unfiltered_IDF1']:.2f} | {r['confirmed_handoffs']}/{r['reference_handoffs']} |")
    lines+=['','## Interpretation limits','',
      '- Official evaluator rules remove predicted IDs that appear in only one camera. ROI-only and unfiltered identity scores expose fresh-ID failures hidden by this filter.',
      '- Custom handoff scores use 15 native frames at each endpoint and positive-gap GT transitions on the declared routes. Overlaps and unsupported routes are counted separately in each metrics file.',
      '- The K=0/2/4/8 controls use the same bounded assignment kernel, one required observation, and a fixed eight-frame source retention; only the deadline changes. At 10 FPS the deadlines are 0/0.2/0.4/0.8 s.',
      '- This is a suffix study with labeled target-scene prefix calibration, not a full official validation score or zero-shot scene generalization.',
      '- Published association results, when present, reuse local tracks and appearance inputs on a desktop adaptation. They do not reproduce the complete detector/tracker/ReID pipeline of that publication.',
      '- Supplied detector runtime is unavailable. Cache replay speed does not establish detector throughput, real-time latency, or edge deployment.',
      '- No suffix score is used to select geometry, thresholds, or a preferred deadline. Retain all outcomes, including negative ones.','']
    (output/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        controls=[r for r in records if r['confirmation_frames'] is not None]
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for key,label in [('official_IDF1','Official'),('roi_only_IDF1','ROI only'),('unfiltered_IDF1','Unfiltered')]:
            axes[0].plot([r['deadline_s'] for r in controls],[100*r[key] for r in controls],'o-',label=label)
        axes[0].set(xlabel='Confirmation deadline (s)',ylabel='Global IDF1 (%)');axes[0].legend();axes[0].grid(alpha=.25)
        axes[1].plot([r['deadline_s'] for r in controls],[100*r['handoff_recall'] if r['handoff_recall'] is not None else 0 for r in controls],'o-')
        axes[1].set(xlabel='Confirmation deadline (s)',ylabel='Declared-route handoff recall (%)');axes[1].grid(alpha=.25)
        fig.savefig(output/'deadline_comparison.png',dpi=180);fig.savefig(output/'deadline_comparison.svg');plt.close(fig)
    except ImportError:pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-validation',type=Path)
    p.add_argument('--published-baseline',action='store_true');args=p.parse_args()
    cfg=json.loads(args.protocol.read_text());scene=json.loads((args.protocol.parent/'scene.json').read_text())
    freeze=json.loads((args.protocol.parent/'freeze.json').read_text())
    if freeze['protocol_sha256']!=file_hash(args.protocol) or freeze['scene_sha256']!=file_hash(args.protocol.parent/'scene.json'):
        raise ValueError('Frozen protocol or scene changed')
    if cfg['route_calibration']['stop_s']>cfg['replay_start_s'] or cfg['score_start_s']<cfg['replay_start_s']:
        raise ValueError('Calibration and evaluation overlap')
    ground_truth=Path(cfg['dataset'])/'eval'/f"ground_truth_{cfg['split']}.txt"
    baseline_script=Path(__file__).with_name('run_cityflow_published_v0_5.py')
    if args.published_baseline and not baseline_script.is_file():raise FileNotFoundError(baseline_script)
    with DetectionCache(args.cache,readonly=True) as cache:
        detector=cache.get_meta('detector')
        expected={'kind':'provided_raw_cityflow_detector_cache','model':'CityFlowV2 supplied mask_rcnn',
                  'minimum_confidence':.1,'descriptor':'vehicle-reid-0001_512_v1',
                  'feature_scope':'every retained detection on its own native video frame; no region/cadence restriction'}
        if any(detector.get(key)!=value for key,value in expected.items()):raise ValueError('Cache does not match the frozen perception protocol')
        if digest(cache.get_meta('scene'))!=cfg['scene_signature']:raise ValueError('Cached scene does not match the frozen protocol')
    if args.cache_validation:
        validation=json.loads(args.cache_validation.read_text())
        if validation.get('status')!='complete' or validation.get('cache_sha256')!=file_hash(args.cache):raise ValueError('Cache verification missing or changed')
    cohort_text='\n'.join(line for line in ground_truth.read_text().splitlines() if int(line.split()[0]) in cfg['cameras'])+'\n'
    signed={'config':cfg,'source_signature':source_signature(),'cache_sha256':file_hash(args.cache),
            'driver_sha256':file_hash(Path(__file__)),'ground_truth_sha256':file_hash(ground_truth),
            'published_baseline':args.published_baseline,
            'baseline_script_sha256':file_hash(baseline_script) if args.published_baseline else None}
    signed['cache_validation_sha256']=file_hash(args.cache_validation) if args.cache_validation else None
    signature=digest(signed);args.output.mkdir(parents=True,exist_ok=True);state_path=args.output/'progress.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous['signature']!=signature or digest(previous['signed_inputs'])!=signature:raise ValueError('Study signature changed; use new output folder')
        for name,expected in previous.get('artifact_hashes',{}).items():
            if not (args.output/name).is_file() or file_hash(args.output/name)!=expected:raise ValueError(f'Study artifact changed: {name}')
        if previous['status']=='complete':
            print(f'Verified completed CityFlowV2 study reused: {args.output}',flush=True);return
    elif any(args.output.iterdir()):raise ValueError('Study output must be empty')
    write_json(state_path,{'status':'running','signature':signature,'signed_inputs':signed,'completed':[]})
    # Validate all selected GT keys before producing scores. Labels remain outside replay.
    gt_rows=annotation_rows(scene,0.,cfg['stop_s'])
    cohort_path=args.output/'ground_truth_S02_cohort.txt'
    if cohort_path.exists() and cohort_path.read_text()!=cohort_text:raise ValueError('Scenario GT cohort changed')
    cohort_path.write_text(cohort_text,encoding='utf-8')
    rois=args.output/'ROIs'
    roi_provenance=prepare_rois(cfg['dataset'],cfg['split'],cfg['scenario'],cfg['cameras'],rois)
    write_json(args.output/'evaluation_inputs.json',{'original_gt_sha256':file_hash(ground_truth),
       'scenario_cohort_sha256':file_hash(cohort_path),'roi_provenance':roi_provenance,
       'note':'Camera cohort selected offline for evaluation only; inference receives no GT.'})
    records=[];local_hash=None;trace_hash=None;reference=None
    for k in cfg['controls']['confirmation_frames']:
        candidate=copy.deepcopy(cfg);candidate['association']['confirmation_frames']=k
        directory=attempt_directory(args.output,f'K{k}_n1');summary=run(candidate,args.cache,directory)
        this_local=file_hash(directory/'local_predictions.txt');this_trace=file_hash(directory/'decisions.jsonl')
        if local_hash is None:local_hash=this_local;trace_hash=this_trace;reference=directory
        if this_local!=local_hash or this_trace!=trace_hash:raise RuntimeError('Delay controls did not share the exact local stream and detector call trace')
        metrics=score(cfg,scene,directory,cohort_path,gt_rows,rois)
        pending=pending_statistics(directory,cfg['score_start_s'],cfg['stop_s']);write_json(directory/'pending_diagnostics.json',pending)
        handoffs=metrics['handoffs']
        row={'method':f'Common kernel K={k}','confirmation_frames':k,'deadline_s':k/cfg['fps'],
             'official_IDF1':metrics['official_benchmark']['idf1'],'roi_only_IDF1':metrics['roi_only_identity']['idf1'],
             'unfiltered_IDF1':metrics['unfiltered_identity']['idf1'],
             'reference_handoffs':handoffs['reference_handoffs'],'confirmed_handoffs':handoffs['confirmed_handoffs'],
             'handoff_recall':handoffs['handoff_recall'],'detector_calls':summary['detector_calls'],
             'median_pending_wait_s':pending['median_wait_s'],'run':str(directory)}
        records.append(row);build_report(records,cfg,args.output)
        # Global reports are refreshed after each control. Only completed-run
        # files and immutable evaluation inputs belong to partial checkpoints.
        artifacts={str(p.relative_to(args.output)):file_hash(p) for p in sorted(args.output.rglob('*'))
                   if p.is_file() and (p.parent!=args.output or p.name in ('ground_truth_S02_cohort.txt','evaluation_inputs.json'))}
        write_json(state_path,{'status':'running','signature':signature,'signed_inputs':signed,'completed':records,'artifact_hashes':artifacts})
        print(canonical(row),flush=True)
    local_directory=args.output/'local_ID_control';write_local_control(reference/'local_predictions.txt',local_directory)
    extra=[('Camera-local ID control',local_directory)]
    if args.published_baseline:
        for retention in ('native','development_route'):
            directory=args.output/f'published_association_{retention}'
            subprocess.run([sys.executable,str(baseline_script),'--config',str(args.protocol),'--cache',str(args.cache),
                            '--local-run',str(reference),'--output',str(directory),'--retention',retention],check=True)
            extra.append((f'Published association ({retention} retention)',directory))
    for name,directory in extra:
        metrics=score(cfg,scene,directory,cohort_path,gt_rows,rois);h=metrics['handoffs']
        records.append({'method':name,'confirmation_frames':None,'deadline_s':None,
             'official_IDF1':metrics['official_benchmark']['idf1'],'roi_only_IDF1':metrics['roi_only_identity']['idf1'],
             'unfiltered_IDF1':metrics['unfiltered_identity']['idf1'],
             'reference_handoffs':h['reference_handoffs'],'confirmed_handoffs':h['confirmed_handoffs'],
             'handoff_recall':h['handoff_recall'],'detector_calls':summary['detector_calls'],
             'median_pending_wait_s':None,'run':str(directory)})
    build_report(records,cfg,args.output)
    write_json(args.output/'shared_stream_proof.json',{'local_predictions_sha256':local_hash,'decisions_sha256':trace_hash,
       'all_deadline_controls_identical':True,'note':'Source retention/minimum observation/gates/assignment/perception held fixed.'})
    artifacts={str(p.relative_to(args.output)):file_hash(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='progress.json'}
    write_json(state_path,{'status':'complete','signature':signature,'signed_inputs':signed,'completed':records,'artifact_hashes':artifacts})
    print(f'CityFlowV2 study complete: {args.output}',flush=True)


if __name__=='__main__':main()
