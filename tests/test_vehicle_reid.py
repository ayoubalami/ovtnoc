import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vehicle_reid_support import prepare_crop, route_regions, in_regions


def test_original_model_preprocessing_preserves_rgb_and_raw_scale():
    bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    bgr[:] = [5, 100, 240]
    actual = prepare_crop(bgr, [-5, 0, 20, 10])
    assert actual.shape == (1, 3, 208, 208)
    assert actual.dtype == np.float32
    assert actual[0, :, 100, 100].tolist() == [240., 100., 5.]
    assert prepare_crop(bgr, [30, 0, 40, 10]) is None


def test_feature_region_uses_bottom_center_and_development_geometry_only():
    cfg = {"cameras": [1, 2], "routes": [{"source": 1, "destination": 2,
        "exit_polygon": [[0, .4], [.1, .4], [.1, .7], [0, .7]],
        "entry_polygon": [[.5, .1], [.6, .1], [.6, .3], [.5, .3]]}]}
    regions = route_regions(cfg, .025)
    assert in_regions([0, 350, 100, 600], 1000, 1000, regions[1])
    assert not in_regions([400, 350, 500, 600], 1000, 1000, regions[1])
    assert in_regions([510, 50, 590, 310], 1000, 1000, regions[2])
