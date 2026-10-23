"""Owned source-workspace lifecycle for compact/full DS generation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shutil
import uuid

import cv2
import numpy as np
import PIL

from human_data_generator.runtime.task_paths import task_ref

def validate_output_mode(value):
    if value not in ('compact','full'):raise ValueError('invalid_output_mode')
    return value



MARKER = ".human_data_temporary_source.json"


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _owned_frame(task_root, output, task_id):
    task_root, output = Path(task_root).resolve(), Path(output).resolve()
    ref = task_ref(task_root, output)
    parts = Path(ref).parts
    if (len(parts) != 3 or parts[0] != 'rigs' or
            re.fullmatch(r'rig_[a-z0-9_]+', parts[1]) is None or
            re.fullmatch(r'frame_[0-9]{3,}', parts[2]) is None or
            re.fullmatch(r'[0-9a-f]{32}', str(task_id)) is None):
        raise ValueError('invalid_owned_frame_workspace')
    return output, ref


def prepare_source_workspace(output, output_mode, *, task_root, task_id):
    """Create a source directory and, for compact mode, an ownership marker.

    Existing directories are always rejected.  The marker binds cleanup to the
    exact output directory and a single generated token.
    """
    mode = validate_output_mode(output_mode)
    output, output_ref = _owned_frame(task_root, output, task_id)
    if mode == "full":
        source = output / "source"
        source.mkdir(parents=True, exist_ok=False)
        record_status(
            output,
            mode,
            stage="source_workspace_created",
            success=False,
            cleanup="not_applicable",
            source_dir=str(source),
        )
        return source
    temporary = output / "_temporary_ds_source"
    temporary.mkdir(parents=True, exist_ok=False)
    source = temporary / "source"
    source.mkdir()
    marker = dict(
        owner="human_data_generator.ds_output_modes",
        task_id=task_id,
        output_ref=output_ref,
        temporary_ref=output_ref+'/_temporary_ds_source',
        token=uuid.uuid4().hex,
        created_utc=_utc_now(),
    )
    _write(temporary / MARKER, marker)
    record_status(
        output,
        mode,
        stage="source_workspace_created",
        success=False,
        cleanup="not_started",
        temporary_root=str(temporary),
    )
    return source


def record_status(output, output_mode, *, stage, success, cleanup, **extra):
    value = dict(
        output_mode=validate_output_mode(output_mode),
        stage=stage,
        success=bool(success),
        cleanup=cleanup,
        updated_utc=_utc_now(),
        **extra,
    )
    _write(Path(output) / "generation_status.json", value)
    return value


def preserve_source_recipe(
    output, source_dir, *, render_command=None, source_buffers_retained=False, extra=None
):
    """Persist reproducible settings without declaring temporary files inputs."""
    output, source_dir = Path(output), Path(source_dir)
    manifest = json.loads((source_dir / "source_manifest.json").read_text())
    for row in manifest.get("sources", []):
        source = row.get("source")
        if source is not None and not source_buffers_retained:
            row.pop("source")
            row["ephemeral_filename"] = Path(source).name
            row["retained"] = False
    command = list(render_command) if render_command is not None else None
    if command is not None:
        command = [
            "<ephemeral-source-dir>" if str(item) == str(source_dir) else item
            for item in command
        ]
    revision='migration_a3_20261022'
    dirty=None
    value = dict(
        source_manifest=manifest,
        render_command=command,
        source_buffers_retained=source_buffers_retained,
        code_version=dict(git_commit=revision, working_tree_dirty=dirty),
        software_versions=dict(
            python=platform.python_version(), numpy=np.__version__,
            opencv=cv2.__version__, pillow=PIL.__version__,
            blender=manifest.get("blender_version", "not_recorded_by_legacy_renderer"),
        ),
        recompute_boundary=(
            "Recomputing DS imaging, source-raster visibility, or occlusion requires "
            "rerendering the source RGB/range/instance buffers from this recipe."
        ),
    )
    if extra:
        value.update(extra)
    _write(output / "source_recipe.json", value)
    return value


def cleanup_owned_source(output, source_dir, *, task_root, task_id):
    """Remove only a compact workspace carrying its exact ownership marker."""
    output, output_ref = _owned_frame(task_root, output, task_id)
    source_dir = Path(source_dir).resolve()
    temporary = output / '_temporary_ds_source'
    marker_path = temporary / MARKER
    if source_dir != temporary / 'source' or not marker_path.is_file():
        raise RuntimeError(f"Refusing to clean unowned source directory: {source_dir}")
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("owner") != "human_data_generator.ds_output_modes"
        or marker.get('task_id') != task_id
        or marker.get('output_ref') != output_ref
        or marker.get('temporary_ref') != output_ref+'/_temporary_ds_source'
        or not isinstance(marker.get('token'),str) or len(marker['token']) != 32
    ):
        raise RuntimeError(f"Temporary source ownership mismatch: {temporary}")
    shutil.rmtree(temporary)
    return temporary


def complete_output(output, output_mode, source_dir, *, task_root, task_id, checks=None):
    """Record successful validation, then clean compact-owned source buffers."""
    mode = validate_output_mode(output_mode)
    output, source_dir = Path(output).resolve(), Path(source_dir).resolve()
    record_status(
        output,
        mode,
        stage="validated",
        success=True,
        cleanup="pending" if mode == "compact" else "not_applicable",
        checks=checks or {},
    )
    if mode == "compact":
        try:
            cleaned = cleanup_owned_source(output, source_dir,task_root=task_root,task_id=task_id)
        except Exception as error:
            temporary = source_dir.parent
            record_status(
                output,
                mode,
                stage="validated_cleanup_failed",
                success=False,
                data_validation="passed",
                cleanup="failed",
                cleanup_error=f"{type(error).__name__}: {error}",
                residual_path=str(temporary) if temporary.exists() else None,
                residual_exists=temporary.exists(),
                checks=checks or {},
            )
            raise
        return record_status(
            output,
            mode,
            stage="completed",
            success=True,
            cleanup="completed",
            cleaned_temporary_root=str(cleaned),
            checks=checks or {},
        )
    return record_status(
        output,
        mode,
        stage="completed",
        success=True,
        cleanup="not_applicable",
        retained_source_dir=str(source_dir),
        checks=checks or {},
    )


def record_failure(output, output_mode, source_dir, stage, error):
    """Record failure without deleting any source or partial output."""
    status_path = Path(output) / "generation_status.json"
    if status_path.is_file():
        existing = json.loads(status_path.read_text())
        if (
            existing.get("stage") == "validated_cleanup_failed"
            and existing.get("data_validation") == "passed"
            and existing.get("cleanup") == "failed"
        ):
            return existing
    return record_status(
        output,
        output_mode,
        stage=stage,
        success=False,
        cleanup="preserved_after_failure",
        source_dir=str(Path(source_dir).resolve()),
        error=f"{type(error).__name__}: {error}",
    )
