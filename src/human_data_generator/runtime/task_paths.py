"""Resolve files owned by one prepared task from its root, never from cwd."""
import json
from pathlib import Path, PurePosixPath
import re


RUN_SCHEMA = 'human_run_config_v2'


def task_path(root, reference):
    root = Path(root).resolve()
    if not isinstance(reference, str) or not reference or '\\' in reference:
        raise ValueError('invalid_task_reference')
    relative = PurePosixPath(reference)
    if (relative.is_absolute() or reference != relative.as_posix() or
            any(part in ('', '.', '..') for part in relative.parts)):
        raise ValueError('invalid_task_reference')
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError('task_reference_outside_root')
    return path


def task_ref(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('task_reference_outside_root')
    return path.relative_to(root).as_posix()


def run_config(root):
    root = Path(root).resolve()
    path = task_path(root, 'run_config.json')
    if not path.is_file():
        raise ValueError('current_prepared_layout_required')
    config = json.loads(path.read_text())
    rigs = config.get('rig_ids')
    frames = config.get('selected_frames')
    if (config.get('schema') != RUN_SCHEMA or
            not isinstance(rigs, list) or not 1 <= len(rigs) <= 8 or
            any(not isinstance(rid, str) or re.fullmatch(r'rig_[a-z0-9_]+', rid) is None for rid in rigs) or
            len(rigs) != len(set(rigs)) or
            not isinstance(frames, list) or not frames or
            any(type(frame) is not int or frame < 0 for frame in frames) or
            frames != sorted(set(frames)) or
            not isinstance(config.get('task_id'), str) or
            re.fullmatch(r'[0-9a-f]{32}', config['task_id']) is None):
        raise ValueError('current_prepared_layout_required')
    if any(key in config for key in ('reprocess_units_before_render',
                                     'prior_completed_units', 'remaining_budget_seconds')):
        raise ValueError('obsolete_run_config_fields')
    for name in ('scene.json', 'recipe.json', 'scene_matrices.npy'):
        if not task_path(root, name).is_file():
            raise ValueError('current_prepared_layout_required:' + name)
    for rid in rigs:
        for name in ('rig.json', 'planned_regions.json'):
            if not task_path(root, f'rigs/{rid}/{name}').is_file():
                raise ValueError('current_prepared_layout_required:' + rid + '/' + name)
    obsolete = ('job.json', 'multirig.json', 'camera_configs',
                *(f'rigs/{rid}/camera_configs' for rid in rigs))
    if any(task_path(root, name).exists() for name in obsolete):
        raise ValueError('obsolete_prepared_layout')
    return config
