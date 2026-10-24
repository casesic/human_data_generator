"""Plan, prepare, render and export one current-format MAMMA scene."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from human_data_generator.config import PROJECT_ROOT, resource_path
from human_data_generator.io.json_file import write_json
from human_data_generator.runtime.task_paths import run_config, task_path


def _task_path(name, preview=False):
    if not name or not name.replace("_", "").isalnum() or len(name) > 64:
        raise ValueError("invalid_task_name")
    return PROJECT_ROOT / "outputs" / ("previews" if preview else "runs") / name


def plan(scene_path, frames=None, rigs=None, preview=False):
    from human_data_generator.runtime.scene import read_scene

    scene = read_scene(scene_path)
    selected = list(range(scene["frame_count"])) if frames is None else list(frames)
    rig_ids = [r["rig_id"] for r in scene["rigs"] if rigs is None or r["rig_id"] in rigs]
    if (
        not selected
        or selected != sorted(set(selected))
        or any(k < 0 or k >= scene["frame_count"] for k in selected)
    ):
        raise ValueError("invalid_selected_frames")
    if not rig_ids or (rigs is not None and set(rig_ids) != set(rigs)):
        raise ValueError("invalid_selected_rigs")
    if preview and (len(selected) != 1 or len(rig_ids) != 1):
        raise ValueError("preview_requires_one_frame_and_rig")
    from human_data_generator.runtime.scene import validate

    check = validate(scene, selected, rig_ids)
    if check["status"] != "passed":
        raise ValueError(check["errors"])
    return dict(
        scene_id=scene["scene_id"],
        fps=scene["fps"],
        frame_count=scene["frame_count"],
        selected_frames=selected,
        rig_ids=rig_ids,
        person_ids=[p["person_id"] for p in scene["people"]],
        camera_calls=len(selected) * len(rig_ids) * 2,
        scope="complete" if selected == list(range(scene["frame_count"])) else "selected",
    )


def prepare(scene_path, name, frames=None, rigs=None, preview=False):
    from human_data_generator.runtime.multirig_prepare import prepare as prepare_run

    output = _task_path(name, preview)
    return output, prepare_run(
        output, scene_path=scene_path, frame_indices=frames, selected_rig_ids=rigs, preview=preview
    )


def _check_gpu(gpu_uuid):
    blender = resource_path("blender_executable")
    if not blender.is_file():
        raise FileNotFoundError(blender)
    queries = dict(
        gpus=[
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv",
        ],
        compute=[
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
    )
    evidence = {key: subprocess.check_output(cmd, text=True) for key, cmd in queries.items()}
    gpu = next((row for row in evidence["gpus"].splitlines()[1:] if gpu_uuid in row), None)
    if (
        gpu is None
        or int(gpu.split(",")[2].split()[0]) > 512
        or int(gpu.split(",")[4].split()[0]) > 5
    ):
        raise RuntimeError("selected_GPU_unavailable")
    if any(gpu_uuid in row for row in evidence["compute"].splitlines()[1:]):
        raise RuntimeError("other_GPU_compute_process_present")
    return evidence


def _preflight_consumer(python):
    """Check the interpreter saved by prepare before spending a GPU render."""
    command = [
        python,
        "-c",
        'import OpenEXR; assert hasattr(OpenEXR, "File"); '
        "import human_data_generator.consumer; "
        "from human_data_generator.imaging.source import read_ds_source",
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        subprocess.check_output(command, env=env, stderr=subprocess.STDOUT, text=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "output", "") or str(error)
        raise RuntimeError(
            f"consumer_python_preflight_failed: {python}: {detail.strip()}"
        ) from error


def _blender_command(output):
    return [
        str(resource_path("blender_executable")), "-b", "-t", "4",
        "--python-exit-code", "1", "--python",
        str(PROJECT_ROOT / "src/human_data_generator/render/multirig.py"),
        "--", "--root", str(output),
    ]


def _run_blender(output, env, deadline):
    started = time.monotonic()
    with (output / "blender.log").open("w") as log:
        process = subprocess.Popen(
            _blender_command(output), cwd=PROJECT_ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        timed_out = False
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                timed_out = True
                break
            time.sleep(1)
        return dict(returncode=process.wait(), timed_out=timed_out,
                    wall_seconds=time.monotonic()-started)


def _run_consumer_stage(stage, rig_root, env, deadline, *, log_path=None):
    command = [sys.executable, "-m", "human_data_generator.consumer", stage,
               "--root", str(rig_root)]
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True,
                       timeout=max(.1, deadline-time.monotonic()))
    else:
        with log_path.open("w") as log:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=max(.1, deadline-time.monotonic()))


def render_prepared(output, gpu_uuid, budget_seconds, video=False):
    """The parent alone owns status.json; Blender appends diagnostic events."""
    from human_data_generator.runtime.roi_run_utils import disk_snapshot
    from human_data_generator.data.mamma_export import export_existing

    output = Path(output).resolve()
    config = run_config(output)
    recipe = json.loads(task_path(output, "recipe.json").read_text())
    if recipe["schema"] != "human_run_v1":
        raise ValueError("current_prepared_task_required")
    if any((output / name).exists() for name in
           ("status.json", "events.jsonl", "observations.json", "started.json", "failure.json")):
        raise FileExistsError("task_already_started")
    if type(budget_seconds) is not int or budget_seconds < 1:
        raise ValueError("positive_budget_required")
    _preflight_consumer(config["consumer_python"])
    _check_gpu(gpu_uuid)

    started = time.monotonic()
    deadline = started + budget_seconds
    config.update(gpu_uuid=gpu_uuid, budget_seconds=budget_seconds,
                  deadline_monotonic=deadline)
    write_json(output / "run_config.json", config)
    status = dict(schema="human_task_status_v1", state="running", stage="render",
                  data_complete=False, gpu_uuid=gpu_uuid, budget_seconds=budget_seconds)
    def save_status(state, stage, **details):
        status.update(state=state, stage=stage, **details)
        write_json(output / "status.json", status)

    save_status("running", "render")
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"),
               CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR="/tmp/human-data-generator-mpl")
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
    try:
        result = _run_blender(output, env, deadline)
        if result["returncode"] or result["timed_out"]:
            save_status("running", "render", blender=result)
            raise RuntimeError("render_or_consume_failed")
        if config["preview_only"]:
            save_status("preview_complete", "preview", blender=result,
                        frames=config["selected_frames"],
                        wall_seconds=time.monotonic()-started)
            return output
        save_status("running", "finish", blender=result)
        for rig_id in config["rig_ids"]:
            rig_root = task_path(output, f"rigs/{rig_id}")
            _run_consumer_stage("finish", rig_root, cpu_env, deadline,
                                log_path=rig_root / "finish.log")
        save_status("running", "export")
        manifest = export_existing(output)
        write_json(output / "observations.json", manifest)
        save_status("data_complete", "export", data_complete=True,
                    observations=len(manifest["observations"]),
                    observed=sum(row["status"] == "observed" for row in manifest["observations"]),
                    wall_seconds=time.monotonic()-started,
                    disk=disk_snapshot(output))
    except BaseException as error:
        save_status("failed", status["stage"],
                    error=f"{type(error).__name__}: {error}",
                    failed_site_preserved=True, wall_seconds=time.monotonic()-started)
        raise

    if video:
        errors = {}
        for rig_id in config["rig_ids"]:
            try:
                _run_consumer_stage("video", task_path(output, f"rigs/{rig_id}"),
                                    cpu_env, deadline)
            except Exception as error:
                errors[rig_id] = f"{type(error).__name__}: {error}"
        save_status("data_complete", "video", video_status="failed" if errors else "completed",
                    video_errors=errors)
    return output



def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "prepare", "render"):
        p = sub.add_parser(action)
        p.add_argument("--scene", type=Path, required=True)
        if action != "plan":
            p.add_argument("--name", required=True)
        p.add_argument("--frame", type=int, action="append")
        p.add_argument("--rig", action="append")
        p.add_argument("--preview", action="store_true")
        if action == "render":
            p.add_argument("--gpu-uuid", required=True)
            p.add_argument("--budget-seconds", type=int, default=1800)
            p.add_argument("--video", action="store_true")
    p = sub.add_parser("run-prepared")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--budget-seconds", type=int, default=1800)
    p.add_argument("--video", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps(plan(args.scene, args.frame, args.rig, args.preview), indent=2))
    elif args.action == "prepare":
        output, config = prepare(args.scene, args.name, args.frame, args.rig, args.preview)
        print(json.dumps(dict(output=str(output), camera_calls=config["max_camera_calls"])))
    elif args.action == "render":
        output, _ = prepare(args.scene, args.name, args.frame, args.rig, args.preview)
        render_prepared(output, args.gpu_uuid, args.budget_seconds, args.video)
    elif args.action == "run-prepared":
        render_prepared(args.root, args.gpu_uuid, args.budget_seconds, args.video)
    else:
        from human_data_generator.data.mamma_export import export_existing

        output = args.root.resolve()
        manifest = export_existing(output)
        destination = output / "observations.json"
        if destination.exists():
            raise FileExistsError(destination)
        write_json(destination, manifest)
        print(json.dumps(dict(output=str(destination), observations=len(manifest["observations"]))))


if __name__ == "__main__":
    main()
