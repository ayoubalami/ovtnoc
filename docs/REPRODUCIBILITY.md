# Reproducing the manuscript experiments

## Method and evaluation scope

The contribution is bounded, immutable global identity commitment. Perception, ByteTrack and the vehicle ReID model are existing components. Buffered prediction rows are emitted with explicit publication times. Ground truth enters offline evaluation and the disclosed prefix calibration, not online candidate scoring or identity assignment.

The primary ablation changes only `confirmation_frames`: 0, 2, 4 and 8. Candidate gates, assignment, source retention (8 frames), and the minimum observation requirement (1) are fixed. Drivers require identical local prediction and detector-decision hashes across these controls. RoundaboutHD additionally reports a separate K=8/minimum-two-observations control, excluded from the delay-only comparison.

| Experiment | Cameras | Replay interval | Scored interval | Native clock |
| --- | --- | --- | --- | --- |
| RoundaboutHD validation | 1, 2 | [1800,3000) frames | [1800,3000) | 15 FPS |
| RoundaboutHD test | 1, 2 | [7800,9000) frames | [7800,9000) | 15 FPS |
| RoundaboutHD expanded | 1–4 | [3900,7500) frames | [5100,7500) | 15 FPS |
| CityFlowV2 S05 | 29, 33, 34, 35 | [120,340) seconds | [160,340) | 10 FPS |

RoundaboutHD intervals are within-scene reanalyses of previously exposed data. CityFlow perception was selected on S02 [0,60) seconds. S05 routes and geometry were calibrated on its labeled [0,120) prefix. S05 is a separate location relative to S02/RoundaboutHD, with target-scene calibration; the experiment does not establish zero-shot scene transfer.

CityFlow uses supplied Mask R-CNN detections at confidence >=0.1, the official image-only ROI rule, indexed observed descriptors and observed boxes. RoundaboutHD uses supplied raw YOLOv12x detections, with vehicle descriptors in the frozen development regions. Both use the pretrained `vehicle-reid-0001` model. Corrected adapters preserve original detection indices through ByteTrack confidence partitions. No detector/ReID training is performed.

## Runtime and setup

The manuscript was developed on a Windows desktop using Python 3.12. The pinned SciPy version requires Python >=3.12; this release targets Python 3.12. `pyproject.toml` pins the relevant installed package versions; `provenance/original_runtime.txt` preserves the full historical environment snapshot and is **not** an install requirements file. It includes unrelated packages and the former local distribution name. Create a new virtual environment and install this repository with `pip install -e ".[test]"`.

The provided launchers use Python and filesystem paths and contain no machine-specific dataset root. Paths in templates are rebound from `--dataset`. Installation in a completely empty environment and Linux execution should be checked on the intended publication platform; local verification details are in `provenance/validation_report.json`.

## Dataset layout

RoundaboutHD can be extracted using the included dataset utility:

```powershell
.venv\Scripts\python.exe -m rhd prepare --archive "D:\downloads\RoundaboutHD.zip" --output "D:\datasets\RoundaboutHD"
```

Keep the original raw detection archives and annotations. The publication launcher converts the declared **one-based** source annotation frames to zero-based derivative files under `work/labels/roundabout/`, retaining annotation conflicts in an audit. Conflicting selected labels fail evaluation; no arbitrary deduplication is used.

For CityFlowV2, keep the original AICity22 folder structure and its supplied `eval/eval.py`. The official evaluator is not copied into this repository. The complete S05 camera annotations are necessary for transition topology and evaluation, even though inference uses four cameras. The launcher validates frozen asset checksums before replay; a modified/transcoded dataset requires a separate protocol rather than silently replacing the reference inputs.

## Preparation, extraction and replay

The README commands use three stages:

1. `--stage prepare`: bind paths, verify inputs and sign the relocated protocol. No evaluation labels are opened.
2. `--stage cache`: read raw supplied detections and their native video frames, compute vehicle descriptors, and verify observation contracts.
3. `--stage run`: perform causal replay, compare deadlines, and evaluate predictions offline. CityFlow includes both published-association retention variants and the camera-local identity control; RoundaboutHD adds the published association controls with `--baseline`.

`--stage all` performs stages 2 and 3 after preparation. Extraction can be slow on CPU and caches can be large. Changing input paths, code, parameters or model versions changes run signatures. Use a new `--work` folder for changed experiments; retain earlier results.

The portable protocol freeze records relocated paths and regenerated checksums. It reproduces the previously frozen scientific settings; its creation date does not constitute a new prospective holdout or replace the original pre-evaluation freeze evidence.

## Outputs and comparisons

Study output includes global/local predictions, decisions, identity messages, availability logs, summaries, per-method metrics and comparison tables. Availability-aware scoring counts a row only after its publication time. Inspect early commitment, deadline matching, fresh-ID fallback and unmatched endpoints alongside aggregate identity scores.

CityFlow reports official, ROI-only and unfiltered scores. The official evaluator removes singleton-camera predicted identities; its IDF1 is therefore not interchangeable with the other variants. Full 19-camera transition topology is constructed before route/interval filtering. Hand-off endpoint metrics are custom mechanism diagnostics, not official leaderboard metrics.

The published association controls use unchanged upstream association logic with shared local tracks/descriptors. Report them as **adapted association baselines**. Their detector/tracker/feature pipeline differs from the original paper, so published headline scores are not directly comparable to these runs.

Replay timing excludes supplied detector inference and does not establish live throughput or edge-device performance. Model extraction timing and replay timing must be identified separately.

## Source preservation and cleaning

All 24 modules under `src/` retain the exact bytes from the frozen source archive shipped with manuscript v0.7. Some legacy utilities remain because experiment source signatures hash every module in their package. Relevant drivers and their transitive helpers are retained. Experimental algorithm and evaluation code were not refactored during packaging.

Excluded: datasets and image crops, weights, virtual environments, caches, generated tracking videos, old run folders, manuscripts/slides, installer scripts, unused BoxMOT code, and artifact-building scripts. The benchmark manifest now verifies only the retained TrackEval and RoundaboutHD evaluator files. Fast Online MTMCT retains its separate pinned manifest.

Three tests were changed to use a relative default dataset location instead of a private Windows user path. Optional private development-artifact links were removed from the selection template; reference selection metadata and numerical summaries remain available. Original and release checksums are recorded under `provenance/`.

Run `python scripts/verify_release.py` to verify the distributed file inventory and frozen source. It intentionally fails after local source edits; preserve the release and record a new revision for further work.
