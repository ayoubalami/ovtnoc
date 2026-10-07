"""Topology/causality regression tests on synthetic handoff endpoints."""
import pytest

from cityflow.handoffs import evaluate_handoffs


@pytest.fixture
def scene():
    return {"scenario":"synthetic", "cameras":{
        "6":{"fps":10., "frames":2000, "timestamp_offset_s":100.},
        "7":{"fps":10., "frames":2000, "timestamp_offset_s":120.},
        "8":{"fps":10., "frames":2000, "timestamp_offset_s":120.2}}}


def row(camera, frame, identity=1, box=None):
    return {"camera":camera, "frame":frame, "id":identity, "box":box or [2.25,2.25,4.25,4.25]}


def score(scene, gt, pred, **kwargs):
    return evaluate_handoffs(scene, gt, pred, routes=[(6,7)], **kwargs)


def test_event_clock_orders_cameras_with_different_native_frame_indices(scene):
    gt = [row(6,200),row(7,5)]  # times 120.0, 120.5; destination frame 5 < source 200
    pred = [row(6,200,10),row(7,5,10)]
    result = score(scene, gt, pred)
    assert result["reference_handoffs"] == result["confirmed_handoffs"] == 1
    assert result["correct_associations"] == 1 and result["handoff_precision"] == 1
    assert result["reference_outcomes"][0]["travel_gap_s"] == .5


def test_full_gt_topology_prevents_inventing_handoff_after_route_filtering(scene):
    gt = [row(6,200),row(8,0),row(7,5)]  # intervening camera 8 at 120.2s
    pred = [row(6,200,10),row(7,5,10)]
    result = score(scene, gt, pred)
    assert result["reference_handoffs"] == 0
    assert result["topology"]["excluded_reference_pairs"]["unsupported_route"] == 2
    assert result["predicted_handoffs"] == result["false_associations"] == 1
    assert result["correct_associations"] == 0
    routes = {(r["source"],r["destination"]):r for r in result["routes"]}
    assert routes[(6,8)]["eligible_nonoverlap_transitions"] == 1
    assert routes[(6,7)]["full_topology_cross_camera_pairs"] == 0


def test_time_filter_is_applied_after_topology_not_before(scene):
    # Camera 8 starts before score but remains visible past the source exit.
    scene["cameras"]["8"]["timestamp_offset_s"] = 119.
    gt = [row(6,180),row(8,0),row(8,1),row(7,0)]  # 118,119..119.1,120
    pred = [row(6,180,10),row(7,0,10)]
    result = score(scene, gt, pred)
    assert result["reference_handoffs"] == 0
    assert result["topology"]["full_gt_cross_camera_pairs"] == 2
    assert result["false_associations"] == 1


def test_overlapping_visibility_is_reported_and_excluded(scene):
    gt = [row(6,f) for f in range(200,211)] + [row(7,5)]
    result = score(scene, gt, [])
    assert result["reference_handoffs"] == 0
    assert result["topology"]["excluded_reference_pairs"]["overlap_or_nonpositive_gap"] == 1
    assert result["routes"][0]["destination_entries_in_score"] == 1


def test_source_before_replay_cannot_enter_reference_cohort(scene):
    scene["cameras"]["6"]["timestamp_offset_s"] = 0.
    gt = [row(6,599),row(7,0)]  # source 59.9s precedes replay
    result = score(scene, gt, [])
    assert result["reference_handoffs"] == 0
    assert result["topology"]["excluded_reference_pairs"]["source_outside_replay"] == 1


def test_destination_entry_cohort_boundaries_are_half_open(scene):
    gt = [row(6,190),row(7,0),row(6,1000,2),row(7,920,2)]
    result = score(scene, gt, [])  # destination at 120 included; at 212 excluded
    assert result["reference_handoffs"] == result["missed_handoffs"] == 1
    assert result["topology"]["excluded_reference_pairs"]["destination_outside_score"] == 1


def test_identity_break_and_missing_endpoint_are_distinct(scene):
    gt = [row(6,200),row(7,5)]
    broken = score(scene, gt, [row(6,200,10),row(7,5,20)])
    missing = score(scene, gt, [row(7,5,20)])
    assert broken["identity_breaks"] == 1 and broken["missed_handoffs"] == 0
    assert missing["missed_handoffs"] == 1 and missing["identity_breaks"] == 0


def test_endpoint_window_is_fifteen_native_frames_without_cross_camera_frame_arithmetic(scene):
    gt = [row(6,f) for f in range(100,120)] + [row(7,f) for f in range(21)]
    # Source endpoint window is 105..119; destination window 0..14.
    confirmed = score(scene, gt, [row(6,105,10),row(7,14,10)])
    late_destination = score(scene, gt, [row(6,105,10),row(7,15,10)])
    early_source = score(scene, gt, [row(6,104,10),row(7,14,10)])
    assert confirmed["confirmed_handoffs"] == 1
    assert confirmed["first_spatial_match_delay"]["median_s"] == pytest.approx(1.4)
    assert late_destination["missed_handoffs"] == early_source["missed_handoffs"] == 1


def test_float_prediction_file_boxes_are_used_without_official_integer_export(scene, tmp_path):
    gt = [row(6,200,box=[.9,.9,1.9,1.9]),row(7,5,box=[.9,.9,1.9,1.9])]
    pred = tmp_path / "predictions.txt"
    pred.write_text("6 10 201 .9 .9 1 1 -1 -1\n7 10 6 .9 .9 1 1 -1 -1\n")
    result = score(scene, gt, pred)
    assert result["confirmed_handoffs"] == 1


def test_matched_publication_wait_is_distinct_from_spatial_confirmation_delay(scene):
    gt = [row(6,200),row(7,5),row(7,6)]
    pred = [row(6,200,10),row(7,6,10)]
    availability = [{"camera":6,"global_id":10,"frame":201,"local_frame_zero_based":200,
                     "time_s":120.,"published_at_s":120.},
                    {"camera":7,"global_id":10,"frame":7,"local_frame_zero_based":6,
                     "time_s":120.6,"published_at_s":121.1}]
    result = score(scene, gt, pred, availability=availability)
    assert result["first_spatial_match_delay"]["median_s"] == pytest.approx(.1)
    assert result["matched_destination_publication_wait"]["median_s"] == pytest.approx(.5)
    assert result["confirmed_destination_publication_wait"]["samples"] == 1


def test_missing_availability_does_not_silently_become_zero_wait(scene):
    gt = [row(6,200),row(7,5)]
    result = score(scene, gt, [row(6,200,10),row(7,5,10)], availability=[])
    distribution = result["confirmed_destination_publication_wait"]
    assert distribution["samples"] == 0 and distribution["median_s"] is None
    assert distribution["missing_availability_rows"] == 1


def test_publication_before_observation_is_rejected(scene):
    gt = [row(6,200),row(7,5)]
    with pytest.raises(ValueError, match="precedes"):
        score(scene, gt, [row(6,200,10),row(7,5,10)], availability=[
            {"camera":7,"global_id":10,"frame":6,"published_at_s":120.4}])


def test_duplicate_prediction_or_clock_mismatch_is_rejected(scene):
    gt = [row(6,200),row(7,5)]
    with pytest.raises(ValueError, match="Repeated"):
        score(scene, gt, [row(6,200,10),row(6,200,10)])
    bad = row(6,200,10); bad["time_s"] = 119.9
    with pytest.raises(ValueError, match="timestamp"):
        score(scene, gt, [bad])
