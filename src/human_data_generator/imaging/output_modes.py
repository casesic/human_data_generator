"""Create and clean a frame's temporary source only when ownership matches."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

from human_data_generator.runtime.task_paths import task_ref

MARKER = ".human_data_temporary_source.json"


def validate_output_mode(value):
    if value not in ("compact", "full"):
        raise ValueError("invalid_output_mode")
    return value


def _owned_frame(task_root, output, task_id):
    task_root, output = Path(task_root).resolve(), Path(output).resolve()
    ref = task_ref(task_root, output)
    parts = Path(ref).parts
    if (
        len(parts) != 3
        or parts[0] != "rigs"
        or re.fullmatch(r"rig_[a-z0-9_]+", parts[1]) is None
        or re.fullmatch(r"frame_[0-9]{3,}", parts[2]) is None
        or re.fullmatch(r"[0-9a-f]{32}", str(task_id)) is None
    ):
        raise ValueError("invalid_owned_frame_workspace")
    return output, ref


def prepare_source_workspace(output, output_mode, *, task_root, task_id):
    """Reject existing source directories; compact mode adds a cleanup token."""
    mode = validate_output_mode(output_mode)
    output, output_ref = _owned_frame(task_root, output, task_id)
    if mode == "full":
        source = output / "source"
        source.mkdir(parents=True, exist_ok=False)
        return source

    temporary = output / "_temporary_ds_source"
    temporary.mkdir(parents=True, exist_ok=False)
    source = temporary / "source"
    source.mkdir()
    marker = dict(
        owner="human_data_generator.ds_output_modes",
        task_id=task_id,
        output_ref=output_ref,
        temporary_ref=output_ref + "/_temporary_ds_source",
        token=uuid.uuid4().hex,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    (temporary / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    return source


def cleanup_owned_source(output, source_dir, *, task_root, task_id):
    """A matching task, frame path and token are required before any deletion."""
    output, output_ref = _owned_frame(task_root, output, task_id)
    source_dir = Path(source_dir).resolve()
    temporary = output / "_temporary_ds_source"
    marker_path = temporary / MARKER
    if source_dir != temporary / "source" or not marker_path.is_file():
        raise RuntimeError(f"Refusing to clean unowned source directory: {source_dir}")
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("owner") != "human_data_generator.ds_output_modes"
        or marker.get("task_id") != task_id
        or marker.get("output_ref") != output_ref
        or marker.get("temporary_ref") != output_ref + "/_temporary_ds_source"
        or not isinstance(marker.get("token"), str)
        or len(marker["token"]) != 32
    ):
        raise RuntimeError(f"Temporary source ownership mismatch: {temporary}")
    shutil.rmtree(temporary)
    return temporary


def complete_output(output, output_mode, source_dir, *, task_root, task_id, checks=None):
    """The caller has validated output; compact buffers are now safe to clean."""
    mode = validate_output_mode(output_mode)
    if mode == "compact":
        cleanup_owned_source(output, source_dir, task_root=task_root, task_id=task_id)
    return dict(output_mode=mode, cleanup="completed" if mode == "compact" else "not_applicable",
                checks=checks or {})
