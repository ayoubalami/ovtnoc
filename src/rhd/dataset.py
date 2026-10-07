from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from .common import file_hash, write_json


def find_ground_truth(dataset: Path) -> Path:
    """The distributed archive uses 'Turth'; accept that spelling without renaming data."""
    paths = [dataset / name for name in ("Multi_CAM_Ground_Truth.txt", "Multi_CAM_Ground_Turth.txt")]
    found = [p for p in paths if p.is_file()]
    if len(found) > 1:
        raise ValueError("Both ground-truth filename variants exist; choose the label file explicitly")
    if not found:
        raise FileNotFoundError(f"No multi-camera annotation file under {dataset}")
    return found[0]


def safe_destination(root: Path, member: str) -> Path:
    portable = PurePosixPath(member.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or any(":" in p for p in portable.parts):
        raise ValueError(f"Unsafe archive path: {member}")
    destination = (root / Path(*portable.parts)).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes output directory: {member}")
    return destination


def crc_of(path: Path):
    crc = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def prepare_archive(archive: Path, output: Path, include_reid=False) -> dict:
    archive, output = archive.resolve(), output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "extraction_manifest.json"
    archive_stat = archive.stat()
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    # Hash the archive once; reuse only if its size and mtime still match. Extracted
    # members are always checked against their actual ZIP CRC when resuming.
    archive_hash = previous.get("archive_sha256") if previous and previous.get("archive_size") == archive_stat.st_size and previous.get("archive_mtime_ns") == archive_stat.st_mtime_ns else None
    if not archive_hash:
        print("Hashing source ZIP for provenance...", flush=True)
        archive_hash = file_hash(archive)
    if previous and previous.get("archive_sha256") != archive_hash:
        raise ValueError("Output contains another archive's data; choose a new extraction directory")
    records = dict(previous.get("members", {})) if previous else {}
    with zipfile.ZipFile(archive) as z:
        selected = [i for i in z.infolist() if not i.is_dir() and (include_reid or not any("reid" in p.lower() for p in PurePosixPath(i.filename).parts))]
        for info in selected:
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive symlink is not supported: {info.filename}")
            safe_destination(output, info.filename)
        remaining = sum(i.file_size for i in selected if not safe_destination(output, i.filename).exists())
        free = shutil.disk_usage(output).free
        if free < remaining + 512 * 1024 * 1024:
            raise ValueError(f"Not enough disk space: need {remaining} bytes plus 512 MiB reserve, have {free}")
        manifest = {"archive": str(archive), "archive_sha256": archive_hash,
                    "archive_size": archive_stat.st_size, "archive_mtime_ns": archive_stat.st_mtime_ns,
                    "include_reid": include_reid, "members": records, "status": "extracting"}
        write_json(manifest_path, manifest)
        for index, info in enumerate(selected, 1):
            destination = safe_destination(output, info.filename)
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != info.file_size or crc_of(destination) != info.CRC:
                    raise ValueError(f"Existing extracted file differs; preserving it: {destination}")
                if info.filename not in records:
                    records[info.filename] = {"bytes": info.file_size, "crc32": info.CRC, "sha256": file_hash(destination)}
                print(f"verified existing [{index}/{len(selected)}] {info.filename}", flush=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            crc, h, count = 0, hashlib.sha256(), 0
            # Failed/interrupted partial files are retained, and completed data is
            # published atomically only after size and CRC validation.
            with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".partial", dir=destination.parent, delete=False) as out:
                temporary = Path(out.name)
                with z.open(info) as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        out.write(block)
                        count += len(block)
                        crc = zlib.crc32(block, crc)
                        h.update(block)
                        if count % (256 * 1024 * 1024) == 0:
                            print(f"{info.filename}: {count // (1024*1024)} MiB extracted", flush=True)
            if count != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
                raise ValueError(f"Archive integrity check failed for {info.filename}; partial file retained at {temporary}")
            temporary.replace(destination)
            records[info.filename] = {"bytes": count, "crc32": info.CRC, "sha256": h.hexdigest()}
            write_json(manifest_path, manifest)
            print(f"extracted [{index}/{len(selected)}] {info.filename}", flush=True)
        manifest["status"] = "complete"
        manifest["selected_files"] = len(selected)
        manifest["selected_bytes"] = sum(i.file_size for i in selected)
        write_json(manifest_path, manifest)
        return {"output": str(output), "files": len(selected), "bytes": manifest["selected_bytes"],
                "archive_sha256": archive_hash, "reid_extracted": include_reid}
