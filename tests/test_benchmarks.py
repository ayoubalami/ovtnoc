from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_support import (ArchiveSource, MotionGate, legacy_numpy, official_sct_audit,
                               metric_data, sct_metrics, upstream_metrics, verify_vendor)
from rhd.evaluation import identity_metrics


def row(frame, identity=1, camera=1, box=None):
    return {"frame": frame, "id": identity, "camera": camera, "box": box or [10, 10, 40, 30]}


def test_upstream_metrics_perfect_and_switch():
    verify_vendor()
    gt = [row(f) for f in range(3)]
    perfect = sct_metrics(gt, gt, [1], 0, 3)["combined_counts"]
    assert perfect["IDF1"] == perfect["HOTA"] == perfect["MOTA"] == 1
    pred = [row(0), row(1), row(2, 2)]
    result = sct_metrics(gt, pred, [1], 0, 3)
    assert result["custom_identity_parity"]
    assert result["combined_counts"]["IDF1"] == pytest.approx(2 / 3)
    assert result["combined_counts"]["IDSW"] == 1


def test_empty_ground_truth_frames_still_count_false_positives():
    gt = [row(1)]
    pred = [row(0), row(1), row(2)]
    measured = sct_metrics(gt, pred, [1], 0, 3)["combined_counts"]
    assert measured["IDFP"] == measured["CLR_FP"] == 2
    assert measured["MOTA"] == -1
    assert measured["IDF1"] == .5


def test_local_namespaces_do_not_merge_same_id_between_cameras():
    gt = [row(0, 1, 1), row(0, 1, 2)]
    pred = [row(0, 7, 1), row(0, 8, 2)]
    measured = sct_metrics(gt, pred, [1, 2], 0, 2)
    assert measured["combined_counts"]["IDF1"] == 1


def test_global_identity_assignment_matches_upstream_across_camera_handoffs():
    gt = [row(0, 1, 1), row(1, 1, 1), row(2, 1, 2), row(3, 1, 2)]
    pred = [row(0, 7, 1), row(1, 7, 1), row(2, 8, 2), row(3, 8, 2)]
    metric = upstream_metrics()[0]
    with legacy_numpy():
        standard = metric.eval_sequence(metric_data(gt, pred, [1, 2], 0, 4, global_ids=True))
    project = identity_metrics(gt, pred, .5)
    for key in ["IDF1", "IDTP", "IDFP", "IDFN"]:
        assert project[key] == standard[key]
    assert standard["IDF1"] == .5


def test_motion_gate_is_causal_and_keeps_confirmed_stopped_vehicle():
    settings = {"kind": "causal_motion", "window_frames": 10,
                "minimum_observations": 3, "displacement_fraction": .01}
    gate = MotionGate(settings, 1000, 1000)
    assert not gate.accept(1, [10, 10, 30, 30], 0)
    assert not gate.accept(1, [10, 10, 30, 30], 1)
    assert not gate.accept(1, [10, 10, 30, 30], 2)
    assert gate.accept(1, [30, 10, 50, 30], 3)
    assert gate.accept(1, [30, 10, 50, 30], 100)
    # Another parked identity cannot inherit the confirmation.
    assert not gate.accept(2, [30, 10, 50, 30], 100)


def test_raw_archive_uses_explicit_frame_names_and_rejects_missing_frames(tmp_path):
    folder = tmp_path / "imagesc001/detection"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "labels_xy.zip", "w") as archive:
        archive.writestr("labels_xy/img000001.txt", "2 1 2 11 12 .9\n")
        archive.writestr("labels_xy/img000000.txt", "")
    cfg = {"cameras": [1], "start_frame": 0, "stop_frame": 2}
    source = ArchiveSource(tmp_path, cfg)
    try:
        assert source.frame(1, 0) == []
        assert source.frame(1, 1)[0]["box"] == [1, 2, 11, 12]
    finally:
        source.close()
    cfg["stop_frame"] = 3
    with pytest.raises(ValueError, match="Missing raw detection"):
        ArchiveSource(tmp_path, cfg)


def test_corrected_and_sct_derived_archives_cannot_be_substituted(tmp_path):
    folder = tmp_path / "imagesc001/detection"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "labels_test.zip", "w") as archive:
        archive.writestr("img000000.txt", "2 1 2 11 12 1\n")
    with pytest.raises(FileNotFoundError):
        ArchiveSource(tmp_path, {"cameras": [1], "start_frame": 0, "stop_frame": 1})


def test_official_sct_audit_keeps_author_threshold_separate():
    result = official_sct_audit([row(0)], [row(0, box=[11, 10, 41, 30])], [1], 0, 1)
    scores = result["per_camera"]["1"]
    assert scores["author_unmodified_iou_distance_0"]["idf1"] == 0
    assert scores["standard_iou_0.5"]["idf1"] == 1


def test_numpy_compatibility_does_not_persist():
    before = {name: np.__dict__.get(name) for name in ["int", "float", "asfarray"]}
    with legacy_numpy():
        assert np.asfarray([1]).dtype == float
    assert {name: np.__dict__.get(name) for name in before} == before
