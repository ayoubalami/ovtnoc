from rhd.quality import MovingVehicleFilter
from rhd.vision import CameraTracker


def settings():
    return {"kind": "causal_motion", "window_frames": 30,
            "minimum_observations": 3, "displacement_fraction": .005}


def track(box, age=0, identity=1):
    return {"local_id": identity, "box": box, "age_since_detection": age}


def test_prediction_drift_cannot_confirm_a_parked_track():
    gate = MovingVehicleFilter(settings(), 1000, 1000)
    assert not gate.apply([track([10, 10, 50, 50])], 0, {1})
    assert not gate.apply([track([100, 10, 140, 50], age=1)], 1, {1})
    assert not gate.apply([track([200, 10, 240, 50], age=2)], 2, {1})
    assert gate.statistics["prediction_only_points_ignored"] == 2
    assert not gate.apply([track([12, 10, 52, 50])], 3, {1})
    assert gate.apply([track([20, 10, 60, 50])], 4, {1})
    assert gate.apply([track([20, 10, 60, 50], age=8)], 12, {1})


def test_confirmation_does_not_retroactively_emit_or_transfer_identity():
    gate = MovingVehicleFilter(settings(), 1000, 1000)
    assert not gate.apply([track([10, 10, 50, 50])], 0, {1})
    assert not gate.apply([track([14, 10, 54, 50])], 1, {1})
    assert gate.apply([track([20, 10, 60, 50])], 2, {1})
    assert not gate.apply([track([20, 10, 60, 50], identity=2)], 3, {2})
    assert 1 not in gate.states


def test_full_tracking_and_intentional_skipping_use_same_eligibility():
    tracker = CameraTracker(15, 1000, 1000, {}, settings())
    detection = {"box": [100, 100, 150, 140], "score": .9, "class_id": 2, "feature": [1, 0]}
    assert not tracker.step([detection], 0)
    assert not tracker.step(None, 1)
    assert not tracker.step([{**detection, "box": [110, 100, 160, 140]}], 2)
    confirmed = tracker.step([{**detection, "box": [120, 100, 170, 140]}], 3)
    assert confirmed
    assert tracker.step(None, 4)[0]["local_id"] == confirmed[0]["local_id"]
    assert not tracker.step([], 5)
