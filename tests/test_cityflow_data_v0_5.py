"""Synthetic input/clock contracts; no method performance measurements."""
from pathlib import Path
import os

import numpy as np
from PIL import Image
import pytest

from cityflow.data import (annotation_rows, contiguous_spans, event_time_us,
                           raw_detections, roi_accepts)


def _det_file(tmp_path, text, filename="det_mask_rcnn.txt"):
    folder = tmp_path / "det"
    folder.mkdir(exist_ok=True)
    path = folder / filename
    path.write_text(text)
    return path


@pytest.mark.parametrize("folder,name", [("gt", "gt.txt"), ("mtsc", "mtsc_mask_rcnn.txt"),
                                         ("det", "labels_test.txt")])
def test_inference_rejects_gt_and_tracking_file_paths(tmp_path, folder, name):
    path = tmp_path / folder / name
    path.parent.mkdir()
    path.write_text("1,-1,1,1,2,2,.9,-1,-1,-1\n")
    with pytest.raises(ValueError, match="raw detector"):
        raw_detections(path, 2, np.full((8, 8), 255, np.uint8))


def test_raw_reader_rejects_identity_annotations_disguised_as_detections(tmp_path):
    path = _det_file(tmp_path, "1,42,1,1,2,2,1,-1,-1,-1\n")
    with pytest.raises(ValueError, match="Unexpected raw"):
        raw_detections(path, 2, np.full((8, 8), 255, np.uint8))


def test_detector_filters_and_first_frame_to_zero_based_index(tmp_path):
    path = _det_file(tmp_path, "1,-1,1,1,2,2,.9,-1,-1,-1\n"
                              "2,-1,1,1,2,2,.05,-1,-1,-1\n"
                              "3,-1,4,4,2,2,.8,-1,-1,-1\n")
    roi = np.full((8, 8), 255, np.uint8); roi[4, 4] = 254
    rows, counts = raw_detections(path, 3, roi, minimum_score=.1)
    assert list(rows) == [0]
    assert rows[0][0]["box"] == [1., 1., 3., 3.]
    assert counts == {"raw": 3, "below_confidence": 1, "outside_roi": 1, "retained": 1}


@pytest.mark.parametrize("line", ["0,-1,1,1,2,2,.9,-1,-1,-1\n",
                                  "3,-1,1,1,2,2,.9,-1,-1,-1\n",
                                  "1,-1,nan,1,2,2,.9,-1,-1,-1\n",
                                  "1,-1,1,1,0,2,.9,-1,-1,-1\n"])
def test_invalid_raw_frames_and_boxes_fail_before_inference(tmp_path, line):
    with pytest.raises(ValueError, match="Invalid detection"):
        raw_detections(_det_file(tmp_path, line), 2, np.full((8, 8), 255, np.uint8))


def test_roi_tests_all_corners_exact_white_and_independent_width_truncation():
    roi = np.full((8, 8), 255, np.uint8)
    roi[3, 3] = 254
    # int(x)=0, int(width)=3; truncating x2 itself to 2 is a different rule.
    assert not roi_accepts([.8, .8, 3.9, 3.9], roi)
    assert roi_accepts([.8, .8, 2.9, 2.9], roi)
    roi[3, 3] = 255; roi[1, 1] = 0
    assert roi_accepts([0, 0, 3, 3], roi)  # a black center does not reject the box
    roi[0, 3] = 0
    assert not roi_accepts([0, 0, 3, 3], roi)


def test_roi_ignores_out_of_bounds_corners_as_official_code_does():
    roi = np.zeros((8, 8), np.uint8)
    assert roi_accepts([-5, -5, -2, -2], roi)
    assert roi_accepts([9, 9, 12, 12], roi)
    assert not roi_accepts([-2, -2, 2, 2], roi)


def test_roi_parity_with_frozen_official_corner_filter(tmp_path):
    from cityflow.evaluation import COLUMNS, load_official_evaluator
    import pandas as pd
    root = Path(os.environ.get("CITYFLOW_DATASET_ROOT", "data/CityFlowV2"))
    official = root / "eval/eval.py"
    if not official.exists():
        pytest.skip("Licensed dataset evaluator unavailable")
    roi = np.full((12, 16), 255, np.uint8); roi[3, 3] = 254; roi[5, 10] = 0
    folder = tmp_path / "validation/c006"; folder.mkdir(parents=True)
    Image.fromarray(roi).save(folder / "roi.jpg", format="PNG")
    boxes = [[.8,.8,3.9,3.9], [1.1,1.1,3.9,3.9], [-4,-4,-1,-1], [10,5,13,8], [14,10,18,14]]
    entries = []
    for identity, (x,y,x2,y2) in enumerate(boxes):
        entries.append([6, identity, 1, int(x), int(y), int(x2-x), int(y2-y), -1, -1])
    table = pd.DataFrame(entries, columns=COLUMNS)
    filtered = load_official_evaluator(official)["removeOutliersROI"](table, dstype="validation", roidir=str(tmp_path))
    assert set(filtered.Id) == {i for i, box in enumerate(boxes) if roi_accepts(box, roi)}


def test_local_first_frame_and_native_offset_event_clock():
    meta = {"frames": 12, "fps": 10., "timestamp_offset_s": .421}
    assert event_time_us(meta, 0) == 421000
    assert event_time_us(meta, 1) == 521000
    assert event_time_us(meta, 11) == 1521000
    for invalid in (-1, 12, True, 1.):
        with pytest.raises(ValueError, match="Local frame"):
            event_time_us(meta, invalid)


def test_annotation_reader_converts_frame_one_and_applies_clock_interval(tmp_path):
    folder = tmp_path / "c006"; (folder / "gt").mkdir(parents=True)
    (folder / "gt/gt.txt").write_text("1,17,1,2,3,4,1,-1,-1,-1\n2,17,1,2,3,4,1,-1,-1,-1\n")
    scene = {"cameras":{"6":{"video":str(folder/"vdo.avi"), "fps":10., "frames":2,
                                    "timestamp_offset_s":.421}}}
    rows = annotation_rows(scene, .421, .521)
    assert len(rows) == 1 and rows[0]["frame"] == 0 and rows[0]["time_s"] == .421


def test_gaps_separate_visits_and_camera_clock_orders_them():
    rows = [{"id":17, "camera":6, "frame":f, "time_s":120+f/10} for f in (0,1,3)]
    rows += [{"id":17, "camera":7, "frame":0, "time_s":121}]
    spans = contiguous_spans(list(reversed(rows)))[17]
    assert [[(r["camera"],r["frame"]) for r in span] for span in spans] == [
        [(6,0),(6,1)], [(6,3)], [(7,0)]]
