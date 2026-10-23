"""Read-only disk accounting for a generated run."""
from pathlib import Path


def disk_snapshot(out):
    total=temporary=0
    for file in Path(out).rglob('*'):
        try:
            if not file.is_file():continue
            size=file.stat().st_size
        except FileNotFoundError:continue
        total+=size
        if '_temporary_ds_source' in file.parts:temporary+=size
    return dict(disk_bytes=total,temporary_bytes=temporary)
