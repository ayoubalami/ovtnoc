"""Descriptive paired deadline differences, resampling whole GT vehicle IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
from rhd.common import file_hash,write_json

FIELDS = ('gt_id','source_camera','destination_camera','source_last_native_frame_1based',
          'destination_first_native_frame_1based')


def compare(base,other,repetitions=2000,seed=19):
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row[field] for field in FIELDS)
            if key in result:raise ValueError('Repeated handoff reference key')
            result[key] = row['outcome'] == 'confirmed_handoffs'
        return result
    a,b = index(base),index(other)
    if a.keys() != b.keys():raise ValueError('Methods have different reference cohorts')
    if not a:return dict(reference_handoffs=0,vehicle_clusters=0,delta_recall_pp=None,interval_95_pp=None)
    vehicles = sorted({key[0] for key in a})
    totals = np.asarray([sum(key[0] == identity for key in a) for identity in vehicles])
    differences = np.asarray([sum(int(b[key])-int(a[key]) for key in a if key[0] == identity) for identity in vehicles])
    interval = None
    if len(vehicles) >= 2:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0,len(vehicles),size=(repetitions,len(vehicles)))
        sampled = 100*differences[draws].sum(axis=1)/totals[draws].sum(axis=1)
        interval = np.quantile(sampled,[.025,.975]).tolist()
    return dict(reference_handoffs=len(a),vehicle_clusters=len(vehicles),
        base_confirmed=sum(a.values()),other_confirmed=sum(b.values()),
        newly_confirmed=sum(b[key] and not a[key] for key in a),
        lost_confirmations=sum(a[key] and not b[key] for key in a),
        delta_recall_pp=100*(sum(b.values())-sum(a.values()))/len(a),interval_95_pp=interval,
        bootstrap_repetitions=repetitions,seed=seed,bootstrap_unit='whole GT vehicle identity',
        note='Descriptive percentile interval conditional on this recording and declared routes; no across-scene interval or significance claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    paths = {k:args.study/f'K{k}_n1/metrics.json' for k in (0,2,4,8)}
    metrics = {k:json.loads(path.read_text()) for k,path in paths.items()}
    report = dict(status='complete',comparisons={str(k):compare(metrics[0]['handoffs']['reference_outcomes'],
        metrics[k]['handoffs']['reference_outcomes']) for k in (2,4,8)},
        input_sha256={str(path):file_hash(path) for path in paths.values()},
        analysis_sha256=file_hash(Path(__file__)),note='All deadlines are retained. No winning deadline is selected from this analysis.')
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:raise ValueError('Existing paired analysis changed')
    else:write_json(args.output,report)
    print(json.dumps(report['comparisons']),flush=True)


if __name__ == '__main__':main()
