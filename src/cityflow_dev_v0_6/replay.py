"""Versioned observation-box adapter around the unchanged native replay kernel."""
from pathlib import Path
from cityflow import replay as legacy
from rhd.common import digest,file_hash
from .observed_tracker import IndexedCameraTracker, ObservedBoxTracker

_BASE_SIGNATURE = legacy.source_signature


def source_signature():
    folder = Path(__file__).parent
    return digest({'unchanged_core':_BASE_SIGNATURE(),
                   'adapter':file_hash(folder/'replay.py'),
                   'observed_tracker':file_hash(folder/'observed_tracker.py')})


def run(cfg,cache_path,output):
    mode = cfg.get('tracker_box_mode','kalman')
    if mode == 'kalman':
        return legacy.run(cfg,cache_path,output)
    if mode not in ('kalman_indexed','observed'):
        raise ValueError('tracker_box_mode must be kalman, kalman_indexed or observed')
    original_tracker,original_signature = legacy.CameraTracker,legacy.source_signature
    if original_signature is not _BASE_SIGNATURE:
        raise RuntimeError('Observation-box replay must run serially')
    # The native event implementation uses module globals. Restrict substitution
    # to this serial call, restore on failure, and bind both implementations into
    # its manifest signature. Independent processes have independent globals.
    legacy.CameraTracker = ObservedBoxTracker if mode == 'observed' else IndexedCameraTracker
    legacy.source_signature = source_signature
    try:
        return legacy.run(cfg,cache_path,output)
    finally:
        legacy.CameraTracker = original_tracker
        legacy.source_signature = original_signature
