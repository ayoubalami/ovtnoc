"""Post-test identity/track diagnostics; never changes inference or selection."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from cityflow.data import annotation_rows,contiguous_spans
from cityflow_dev_v0_6.diagnostics import match_rows,read_prediction_rows
from rhd.common import file_hash,write_json


def main():
    root = Path(__file__).resolve().parents[1];study = root/'runs/cityflow_S05_suffix_v0_6'
    if json.loads((study/'progress.json').read_text())['status'] != 'complete':
        raise ValueError('Post-test diagnosis requires the completed frozen study')
    scene = json.loads((root/'protocols/cityflow_development_v0_6/holdout_scene_metadata.json').read_text())
    full_stop = max(meta['timestamp_offset_s']+meta['frames']/meta['fps'] for meta in scene['cameras'].values())
    full_gt = annotation_rows(scene,0.,full_stop)
    cohort = {29,33,34,35};gt = [r for r in full_gt if r['camera'] in cohort and 160 <= r['time_s'] < 340]
    local_path = study/'K0_n1/local_predictions.txt'
    local = read_prediction_rows(local_path,scene,160.,340.)
    matches = match_rows(gt,local,.5);spans = contiguous_spans(full_gt);fragmentation = []
    for visits in spans.values():
        for visit in visits:
            if visit[0]['camera'] not in cohort:continue
            selected = [r for r in visit if 160 <= r['time_s'] < 340]
            observed = [matches[(r['camera'],r['frame'],r['id'])]['prediction']['id'] for r in selected
                        if (r['camera'],r['frame'],r['id']) in matches]
            if len(observed) >= 10:
                fragmentation.append(dict(gt_id=visit[0]['id'],camera=visit[0]['camera'],
                    matched_observations=len(observed),distinct_local_ids=len(set(observed))))
    directory = study/'K8_n1';metrics = json.loads((directory/'metrics.json').read_text())
    commits = {(r['camera'],r['local_id']):r for line in (directory/'messages.jsonl').read_text().splitlines()
               if (r:=json.loads(line)).get('event') == 'identity_committed'}
    availability = {(r['camera'],r['frame'],r['global_id']):r for line in (directory/'availability.jsonl').read_text().splitlines()
                    if (r:=json.loads(line))['camera'] in cohort}
    reasons = Counter();examples = []
    for ref in metrics['handoffs']['reference_outcomes']:
        if ref['outcome'] == 'missed_handoffs':
            category = 'spatial_endpoint_missing'
        elif ref['outcome'] == 'confirmed_handoffs':
            category = 'confirmed_endpoint_continuity'
        else:
            key = (ref['destination_camera'],ref['destination_matched_native_frame_1based'],ref['destination_predicted_id'])
            entry = availability[key];commit = commits[(entry['camera'],entry['local_id'])]
            if commit['reason'] == 'outside_entry_zone':
                category = 'destination_first_activated_outside_entry_zone'
            else:
                legal = commit.get('diagnostics',{}).get('candidate_global_ids',[])
                if ref['source_predicted_id'] not in legal:
                    category = 'source_endpoint_identity_not_in_legal_candidates_at_commit'
                else:
                    category = 'source_endpoint_identity_legal_but_different_identity_committed'
            examples.append(dict(gt_id=ref['gt_id'],route=[ref['source_camera'],ref['destination_camera']],
                category=category,source_endpoint_identity=ref['source_predicted_id'],
                destination_identity=ref['destination_predicted_id'],commit_reason=commit['reason'],
                decision_status=commit.get('diagnostics',{}).get('decision_status'),
                legal_candidate_count=commit.get('diagnostics',{}).get('legal_candidate_count')))
        reasons[category] += 1
    counts = [r['distinct_local_ids'] for r in fragmentation]
    report = dict(status='complete',phase='post_test_explanatory_only',score_interval_s=[160,340],
        local_annotation_recall=len(matches)/len(gt),matched_annotation_rows=len(matches),annotation_rows=len(gt),
        contiguous_gt_spans_with_at_least_10_matches=len(counts),
        spans_with_multiple_local_ids=sum(value>1 for value in counts),
        median_local_ids_per_span=float(np.median(counts)) if counts else None,
        maximum_local_ids_per_span=max(counts,default=None),span_fragmentation=fragmentation,
        K8_reference_outcome_diagnosis=dict(reasons),K8_identity_break_evidence=examples,
        source_sha256=dict(metrics=file_hash(directory/'metrics.json'),local_predictions=file_hash(local_path),
                           availability=file_hash(directory/'availability.jsonl'),messages=file_hash(directory/'messages.jsonl')),
        analyzer_sha256=file_hash(Path(__file__)),
        limits=['No inference setting changed after test exposure.',
                'Legal-candidate diagnostics are decision-time evidence, not a complete physical-source causal diagnosis.',
                'Fragmentation uses contiguous full-scene GT visits with >=10 spatially matched annotated observations; multiple visits are not merged.',
                'GT labels multi-camera vehicles only. Unmatched predictions are not all false positives.'])
    output = root/'paper/cityflow_v0_6/post_test_diagnosis.json'
    if output.exists() and json.loads(output.read_text()) != report:raise ValueError('Existing diagnosis differs')
    write_json(output,report)
    print(json.dumps({key:report[key] for key in ('local_annotation_recall','contiguous_gt_spans_with_at_least_10_matches',
        'spans_with_multiple_local_ids','median_local_ids_per_span','maximum_local_ids_per_span','K8_reference_outcome_diagnosis')}),flush=True)


if __name__ == '__main__':main()
