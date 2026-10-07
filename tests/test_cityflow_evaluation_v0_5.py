"""Synthetic parity and boundary tests; no research-method scores are run."""
from pathlib import Path
import os

import numpy as np
from PIL import Image
import pytest

from cityflow.evaluation import (_motmetrics_compatibility, evaluate_predictions,
    export_official_predictions, frame_ranges_for_interval, load_official_evaluator,
    read_submission)


@pytest.fixture
def official_path():
    root = Path(os.environ.get("CITYFLOW_DATASET_ROOT", "data/CityFlowV2"))
    path = root / "eval" / "eval.py"
    if not path.is_file():
        pytest.skip("Set CITYFLOW_DATASET_ROOT to the licensed AIC22 dataset")
    return path


@pytest.fixture
def cohort(tmp_path, official_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("".join(f"{camera} 1 {frame} 2 2 2 2 -1 -1\n"
                          for camera in (6, 7) for frame in (1, 2)))
    roi = tmp_path / "ROIs"
    for camera in (6, 7):
        path = roi / "validation" / f"c{camera:03d}" / "roi.jpg"
        path.parent.mkdir(parents=True)
        Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(path, format="PNG")
    return {"official_eval_path": official_path, "ground_truth_path": gt,
            "roi_directory": roi, "split": "validation", "cameras": [6, 7]}


def test_exact_native_parity_and_perfect_identity(cohort, tmp_path):
    pred = tmp_path / "pred.txt"
    pred.write_text("".join(f"{camera} 10 {frame} 2 2 2 2 -1 -1\n"
                           for camera in (6, 7) for frame in (1, 2)))
    result = evaluate_predictions(**cohort, prediction_path=pred)
    namespace = load_official_evaluator(cohort["official_eval_path"])
    with _motmetrics_compatibility():
        original = namespace["eval"](namespace["readData"](str(cohort["ground_truth_path"])),
                                     namespace["readData"](str(pred)), dstype="validation",
                                     roidir=str(cohort["roi_directory"]))
    assert result["official_benchmark"]["idf1"] == original.iloc[0].idf1 == 1.0
    assert result["official_benchmark"]["idtp"] == original.iloc[0].idtp == 4
    assert result["roi_only_identity"]["idf1"] == 1.0
    assert result["unfiltered_identity"]["idf1"] == 1.0


def test_singleton_filter_is_visible(cohort, tmp_path):
    pred = tmp_path / "pred.txt"
    pred.write_text("".join(f"{camera} {camera} {frame} 2 2 2 2 -1 -1\n"
                           for camera in (6, 7) for frame in (1, 2)))
    result = evaluate_predictions(**cohort, prediction_path=pred)
    assert result["filter_counts"]["rows_removed_by_singleton_filter"] == 4
    assert result["official_benchmark"]["idf1"] == 0
    assert result["official_benchmark"]["idfn"] == 4
    assert result["roi_only_identity"]["idf1"] == .5
    assert result["unfiltered_identity"]["idf1"] == .5


def test_roi_uses_corners_not_bottom_center(cohort, tmp_path):
    path = cohort["roi_directory"] / "validation" / "c006" / "roi.jpg"
    pixels = np.full((10, 10), 255, dtype=np.uint8)
    pixels[2, 2] = 254
    Image.fromarray(pixels).save(path, format="PNG")
    pred = tmp_path / "pred.txt"
    pred.write_text("".join(f"{camera} 10 {frame} 2 2 2 2 -1 -1\n"
                           for camera in (6, 7) for frame in (1, 2)))
    result = evaluate_predictions(**cohort, prediction_path=pred)
    assert result["filter_counts"]["rows_removed_by_roi"] == 2
    assert result["filter_counts"]["rows_removed_by_singleton_filter"] == 2
    assert result["roi_only_identity"]["idf1"] == pytest.approx(2 / 3)
    assert result["unfiltered_identity"]["idf1"] == 1


def test_empty_predictions_have_defined_metrics(cohort, tmp_path):
    pred = tmp_path / "pred.txt"
    pred.write_text("")
    result = evaluate_predictions(**cohort, prediction_path=pred)
    for key in ("official_benchmark", "roi_only_identity", "unfiltered_identity"):
        assert result[key]["idtp"] == 0
        assert result[key]["idfn"] == 4
        assert result[key]["idfp"] == 0
        assert result[key]["idf1"] == 0


def test_common_clock_ranges_use_actual_offsets():
    clocks = {6: {"fps": 10, "offset_s": 0},
              7: {"fps": 10, "offset_s": .061},
              8: {"fps": 10, "offset_s": .421},
              9: {"fps": 10, "offset_s": .660}}
    assert frame_ranges_for_interval(clocks, 120, 212) == {
        6: [1201, 2121], 7: [1201, 2121], 8: [1197, 2117], 9: [1195, 2115]}
    assert frame_ranges_for_interval({6: {"fps": 8, "offset_s": .125}}, .125, .5) == {6: [1, 4]}


def test_scoring_range_applies_to_both_inputs(cohort, tmp_path):
    pred = tmp_path / "pred.txt"
    pred.write_text("".join(f"{camera} 10 {frame} 2 2 2 2 -1 -1\n"
                           for camera in (6, 7) for frame in (1, 2)))
    result = evaluate_predictions(**cohort, prediction_path=pred, score_interval_s=[.1, .2],
                                  camera_clocks={6: {"fps": 10, "offset_s": 0},
                                                 7: {"fps": 10, "offset_s": 0}})
    assert result["filter_counts"]["ground_truth_rows"] == 2
    assert result["filter_counts"]["input_prediction_rows"] == 2
    assert result["official_benchmark"]["idf1"] == 1


def test_float_export_and_strict_input(tmp_path):
    source, destination = tmp_path / "float.txt", tmp_path / "integer.txt"
    source.write_text("6 1 1 -1.9 2.9 4.9 5.9 -1 -1\n")
    with pytest.raises(ValueError, match="integral"):
        read_submission(source)
    export_official_predictions(source, destination, cameras=[6])
    assert destination.read_text() == "6 1 1 -1 2 4 5 -1 -1\n"
    with pytest.raises(ValueError, match="already exists"):
        export_official_predictions(source, destination)


@pytest.mark.parametrize("row,pattern", [
    ("6 1 0 2 2 2 2 -1 -1\n", "indices"),
    ("6 1 1 2 2 0 2 -1 -1\n", "box size"),
    ("6 1 1 2 2 2 2 -1\n", "nine columns"),
    ("6 1 1 2 2 2 2 -1 -1\n6 1 1 3 3 2 2 -1 -1\n", "Duplicate")])
def test_invalid_rows_are_rejected(tmp_path, row, pattern):
    path = tmp_path / "invalid.txt"
    path.write_text(row)
    with pytest.raises(ValueError, match=pattern):
        read_submission(path)


def test_compatibility_is_restored_on_exception():
    import motmetrics as mm
    original = mm.distances.iou_matrix
    with pytest.raises(RuntimeError):
        with _motmetrics_compatibility():
            assert mm.distances.iou_matrix is not original
            raise RuntimeError("test")
    assert mm.distances.iou_matrix is original
