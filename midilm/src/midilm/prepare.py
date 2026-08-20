from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

MIDI_SUFFIXES = {".mid", ".midi"}
MAX_ARCHIVE_BYTES = 100 * 1024**3
MAX_MIDI_BYTES = 64 * 1024**2


def midi_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES)


def extract_zip(archive_path: Path, root: Path) -> None:
    destination = root / "extracted" / archive_path.relative_to(root).with_suffix("")
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if Path(member.filename).suffix.lower() in MIDI_SUFFIXES]
        if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive is too large: {archive_path}")
        for member in members:
            if member.file_size > MAX_MIDI_BYTES:
                raise ValueError(f"MIDI file is too large: {member.filename}")
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"unsafe archive path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def prepare_dataset(repo_id: str, output: Path, revision: str | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=output,
            allow_patterns=["*.mid", "*.midi", "*.MID", "*.MIDI", "*.zip", "*.ZIP"],
        )
    )
    for archive in (path for path in snapshot.rglob("*") if path.suffix.lower() == ".zip"):
        extract_zip(archive, snapshot)
    count = len(midi_files(snapshot))
    if count == 0:
        raise ValueError(
            "no MIDI files found; this dataset may store data in Parquet/Arrow and needs a schema-specific adapter"
        )
    return count
