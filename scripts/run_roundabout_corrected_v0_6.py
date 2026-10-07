"""Rerun the observed Roundabout protocols with corrected observation indices.

All detector inputs, local geometry, motion filter, gates and deadlines remain
the frozen v0.4 choices. Reanalyses do not become untouched validation evidence.
"""
from cityflow_dev_v0_6.roundabout_replay import run,source_signature
import run_deadline_ablation_v0_4 as study


if __name__ == '__main__':
    study.run = run;study.source_signature = source_signature
    study.main()
