"""Illustration schema and exact native-video indexing tests, without method runs."""
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from export_cityflow_examples_v0_5 import (_panels,export_examples,read_exact_frames,select_examples)
from rhd.common import file_hash


def reference(**changes):
    result = {"gt_id":17,"source_camera":6,"destination_camera":7,
              "source_last_native_frame_1based":2,"destination_first_native_frame_1based":1,
              "source_end_time_s":120.1,"destination_start_time_s":120.2,"travel_gap_s":.1,
              "outcome":"confirmed_handoffs","source_predicted_id":10,"destination_predicted_id":10,
              "source_matched_native_frame_1based":1,"destination_matched_native_frame_1based":1,
              "destination_publication_wait_s":None}
    result.update(changes)
    return result


def test_selection_is_deterministic_and_absent_categories_remain_explicit():
    early = reference();late = reference(destination_start_time_s=121.)
    result = select_examples({"handoffs":{"reference_outcomes":[late,early]}})
    assert result[0]["reference"] == early
    assert [r["present"] for r in result] == [True,False,False]
    assert result[1]["reason"].startswith("No measured")


def test_metric_category_must_agree_with_endpoint_ids():
    with pytest.raises(ValueError,match="contradicts"):
        select_examples({"handoffs":{"reference_outcomes":[reference(destination_predicted_id=20)]}})


def test_panels_retain_literal_endpoint_and_exact_matched_frame():
    example = {"reference":reference()}
    panels = _panels(example)
    assert panels[0]["gt_endpoint_frame_1based"] == 2
    assert panels[0]["rendered_frame_1based"] == panels[0]["matched_frame_1based"] == 1
    with pytest.raises(ValueError,match="window"):
        _panels({"reference":reference(source_last_native_frame_1based=16)})


@pytest.fixture
def synthetic_scene(tmp_path):
    scene = {"scenario":"synthetic","cameras":{}}
    for camera,offset in ((6,120.),(7,120.2)):
        folder = tmp_path/f"c{camera:03d}";folder.mkdir()
        video = folder/"vdo.avi"
        writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10.,(64,48))
        if not writer.isOpened():
            pytest.skip("MJPG video writer unavailable")
        try:
            for level in (30,90,160):
                image = np.full((48,64,3),level,np.uint8);image[8:20,8:20,2]=220
                writer.write(image)
        finally:
            writer.release()
        (folder/"gt").mkdir()
        (folder/"gt/gt.txt").write_text("1,17,8,8,12,12,1,-1,-1,-1\n2,17,8,8,12,12,1,-1,-1,-1\n")
        scene["cameras"][str(camera)] = {"fps":10.,"frames":3,"timestamp_offset_s":offset,
                                          "video":str(video),"video_sha256":file_hash(video),"width":64,"height":48}
    return scene


def test_sequential_video_read_retains_exact_native_index_and_checks_hash(synthetic_scene):
    frames,_ = read_exact_frames(synthetic_scene,{(6,1),(6,3)})
    capture = cv2.VideoCapture(synthetic_scene["cameras"]["6"]["video"])
    expected = []
    try:
        for _ in range(3):
            ok,pixels = capture.read();assert ok;expected.append(pixels)
    finally:
        capture.release()
    assert np.array_equal(frames[(6,1)],expected[0])
    assert np.array_equal(frames[(6,3)],expected[2])
    corrupted = copy.deepcopy(synthetic_scene);corrupted["cameras"]["6"]["video_sha256"]="0"*64
    with pytest.raises(ValueError,match="differs"):
        read_exact_frames(corrupted,{(6,1)})


def test_export_preserves_source_frames_caption_and_absent_states(synthetic_scene,tmp_path):
    run = tmp_path/"K8_n1";run.mkdir()
    (run/"predictions.txt").write_text("6 10 1 8 8 12 12 -1 -1\n7 10 1 8 8 12 12 -1 -1\n")
    metrics = {"handoffs":{"scope":{"scenario":"synthetic","iou_threshold":.5,"endpoint_native_frames":15},
                           "reference_outcomes":[reference()]}}
    metrics_path = run/"metrics.json";metrics_path.write_text(json.dumps(metrics))
    scene_path = tmp_path/"scene.json";scene_path.write_text(json.dumps(synthetic_scene))
    output = tmp_path/"examples"
    report = export_examples(metrics_path,scene_path,output)
    assert report["dataset_id"] == "CityFlowV2/AIC22 Track1/synthetic"
    assert report["method"] == "K8_n1"
    assert (output/"confirmed.png").is_file()
    assert not (output/"identity_break.png").exists()
    assert "literal GT endpoint 2" in report["examples"][0]["caption"]
    assert len(report["source_frames"]) == 2
    for source in report["source_frames"]:
        assert file_hash(output/source["file"]) == source["png_sha256"]
    with pytest.raises(ValueError,match="new output"):
        export_examples(metrics_path,scene_path,output)
