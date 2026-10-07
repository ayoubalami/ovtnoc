"""Official externally trained vehicle OSNet; no RoundaboutHD training or labels."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from rhd.common import file_hash, unit_vector

MODEL_SHA256 = "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"


def prepare_crop(image, box):
    """Original ONNX expects RGB float32 0..255; its input IN is in the graph."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    rgb = np.ascontiguousarray(image[y1:y2, x1:x2, ::-1])
    resized = Image.fromarray(rgb).resize((208, 208), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None].copy()


def route_regions(config, padding=0.025):
    """Image geometry only: padded bounding rectangles of development route zones."""
    if not 0 <= padding <= .1:
        raise ValueError("Invalid image-region padding")
    regions = {c: [] for c in config["cameras"]}
    for route in config["routes"]:
        for camera, key in ((route["source"], "exit_polygon"), (route["destination"], "entry_polygon")):
            points = np.asarray(route[key])
            lo = np.maximum(points.min(axis=0) - padding, 0)
            hi = np.minimum(points.max(axis=0) + padding, 1)
            regions[camera].append([*lo.tolist(), *hi.tolist()])
    return regions


def in_regions(box, width, height, regions):
    x = (box[0] + box[2]) / (2 * width)
    y = box[3] / height
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)


class VehicleReID:
    def __init__(self, model: Path, threads=4):
        if file_hash(model) != MODEL_SHA256:
            raise ValueError("Vehicle model differs from verified official ONNX")
        import openvino as ov
        net = ov.Core().read_model(model)
        net.reshape([1, 3, 208, 208])
        self.compiled = ov.Core().compile_model(net, "CPU", {
            "INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32"})
        self.output = self.compiled.output(0)

    def descriptor(self, image, box):
        value = prepare_crop(image, box)
        if value is None:
            return [0.] * 512
        output = np.asarray(self.compiled([value])[self.output]).reshape(-1)
        if output.size != 512 or not np.isfinite(output).all():
            raise ValueError("Invalid vehicle appearance embedding")
        return unit_vector(output).tolist()
