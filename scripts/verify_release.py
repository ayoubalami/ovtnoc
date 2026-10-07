"""Verify release and frozen scientific-source integrity, without dataset access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(files, root):
    root = root.resolve()
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Missing or changed release file: {name}')


def main():
    manifest_path = ROOT/'provenance/release_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('release_manifest.json is missing')
    release = json.loads(manifest_path.read_text())
    check(release['files_sha256'], ROOT)
    frozen = json.loads((ROOT/'provenance/frozen_files.json').read_text())
    check(frozen['unchanged_archive_files_sha256'], ROOT)
    modules = {p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')}
    expected = {p for p in frozen['unchanged_archive_files_sha256'] if p.startswith('src/')}
    if modules != expected or len(modules) != 24:
        raise ValueError('Frozen scientific-module inventory changed')
    vendor = json.loads((ROOT/'third_party/benchmark_manifest.json').read_text())
    check(vendor['files'], ROOT/'third_party')
    print(f"Release verified: {len(release['files_sha256'])} files; {len(modules)} unchanged core modules.")


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
