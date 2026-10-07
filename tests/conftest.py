"""Keep dependency settings within the local, ignored work directory."""
import os
from pathlib import Path

if 'YOLO_CONFIG_DIR' not in os.environ:
    settings = Path(__file__).resolve().parents[1]/'work/ultralytics'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
