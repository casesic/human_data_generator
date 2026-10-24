"""Loopback editor for current scenes; only explicit start requests launch work."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from urllib.parse import urlparse, parse_qs

from human_data_generator.config import PROJECT_ROOT
from human_data_generator.editor.core import default_scene
from human_data_generator.io.json_file import read_json, write_json
from human_data_generator.runtime.scene import map_data, validate

WEB = Path(__file__).with_name("index_current.html")
TASKS = {}
LOCK = threading.Lock()
RECIPES = PROJECT_ROOT / "outputs/recipes"


def _gpu_uuid():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for row in result.stdout.splitlines():
        fields = [x.strip() for x in row.split(",")]
        if len(fields) == 3 and int(fields[1]) <= 512 and int(fields[2]) <= 5:
            return fields[0]
    raise RuntimeError("no_idle_GPU")


def _worker(task):
    try:
        task["state"] = "rendering"
        command = [
            sys.executable,
            "-m",
            "human_data_generator.run",
            "render",
            "--scene",
            task["scene_path"],
            "--name",
            task["id"],
            "--gpu-uuid",
            task["gpu_uuid"],
            "--budget-seconds",
            str(task["budget_seconds"]),
        ]
        if task["kind"] == "preview":
            command += ["--preview", "--frame", str(task["frame"]), "--rig", task["rig_id"]]
        output = (
            PROJECT_ROOT
            / "outputs"
            / ("previews" if task["kind"] == "preview" else "runs")
            / task["id"]
        )
        task["output"] = str(output)
        log = PROJECT_ROOT / "outputs" / f"editor_{task['id']}.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=dict(
                    os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            task["pid"] = process.pid
            code = process.wait()
        if code:
            raise RuntimeError(f"run_failed_{code}: {log}")
        task["state"] = "completed"
    except Exception as error:
        task["state"] = "failed"
        task["error"] = f"{type(error).__name__}: {error}"
    finally:
        task.pop("pid", None)


def start_task(kind, scene, frame=None, rig_id=None, budget_seconds=None):
    if kind not in ("preview", "run"):
        raise ValueError("invalid_task_kind")
    chosen = [frame] if kind == "preview" else None
    rigs = [rig_id] if kind == "preview" else None
    result = validate(scene, chosen, rigs)
    if result["status"] != "passed":
        raise ValueError(result["errors"])
    budget = 300 if kind == "preview" else 2400
    if budget_seconds is not None:
        if type(budget_seconds) is not int or budget_seconds < 1:
            raise ValueError("invalid_budget")
        budget = budget_seconds
    with LOCK:
        if any(t["state"] in ("queued", "rendering") for t in TASKS.values()):
            raise RuntimeError("editor_task_already_running")
        task_id = "task_" + uuid.uuid4().hex[:12]
        gpu = _gpu_uuid()
        scene_path = PROJECT_ROOT / "outputs" / f"editor_scene_{task_id}.json"
        write_json(scene_path, scene)
        task = dict(
            id=task_id,
            kind=kind,
            state="queued",
            frame=frame,
            rig_id=rig_id,
            scene_path=str(scene_path),
            gpu_uuid=gpu,
            budget_seconds=budget,
        )
        TASKS[task_id] = task
    threading.Thread(target=_worker, args=(task,), daemon=True).start()
    return {"task_id": task_id}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request_too_large")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                payload = WEB.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/example":
                return self._reply(default_scene(query.get("environment", ["city"])[0]))
            if parsed.path == "/api/map":
                return self._reply(map_data(query.get("environment", ["city"])[0]))
            if parsed.path == "/api/components":
                from human_data_generator.components import describe

                return self._reply(describe())
            if parsed.path == "/api/task":
                task = TASKS.get(query.get("id", [""])[0])
                if task is None:
                    raise ValueError("unknown_task")
                return self._reply(task)
            if parsed.path == "/api/recipe":
                name = query.get("id", [""])[0]
                if re.fullmatch(r"[a-z0-9_]+__r\d+", name) is None:
                    raise ValueError("invalid_recipe_id")
                return self._reply(read_json(RECIPES / (name + ".json")))
            if parsed.path.startswith("/api/image/"):
                _, _, _, task_id, camera = parsed.path.split("/")
                task = TASKS.get(task_id)
                if (
                    task is None
                    or task["state"] != "completed"
                    or task["kind"] != "preview"
                    or camera not in ("cam2", "cam4")
                ):
                    raise ValueError("image_unavailable")
                image = (
                    Path(task["output"])
                    / "rigs"
                    / task["rig_id"]
                    / f"frame_{task['frame']:03d}"
                    / camera
                    / "roi.png"
                )
                payload = image.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            raise ValueError("unknown_route")
        except Exception as error:
            self._reply({"error": str(error)}, 400)

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/check":
                return self._reply(validate(body["scene"], body.get("frames"), body.get("rig_ids")))
            if self.path == "/api/save":
                scene = body["scene"]
                check = validate(scene)
                if check["status"] != "passed":
                    raise ValueError(check["errors"])
                RECIPES.mkdir(parents=True, exist_ok=True)
                revision = 1
                while (RECIPES / f"{scene['scene_id']}__r{revision}.json").exists():
                    revision += 1
                name = f"{scene['scene_id']}__r{revision}"
                write_json(RECIPES / (name + ".json"), scene)
                return self._reply({"id": name, "path": str(RECIPES / (name + ".json"))})
            if self.path == "/api/start":
                return self._reply(
                    start_task(
                        body["kind"],
                        body["scene"],
                        body.get("frame"),
                        body.get("rig_id"),
                        body.get("budget_seconds"),
                    )
                )
            raise ValueError("unknown_route")
        except Exception as error:
            self._reply({"error": str(error)}, 400)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("loopback_only")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
