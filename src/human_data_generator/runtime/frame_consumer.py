"""One synchronous CPU child: one request, one response, no queued frames.

Only stdlib here so importing the client in Blender cannot import torch/OpenCV.
The server calls the unchanged consumer and never cleans up on an exception.
"""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time
import traceback

from human_data_generator.runtime.task_paths import run_config, task_path


def serve_frames(root, consume, *, incoming=None, outgoing=None):
    root = Path(root).resolve()
    incoming = sys.stdin if incoming is None else incoming
    outgoing = sys.stdout if outgoing is None else outgoing
    config = run_config(root)
    rig_roots = {rid: task_path(root,f'rigs/{rid}') for rid in config['rig_ids']}
    completed = []

    def reply(value):
        outgoing.write(json.dumps(dict(pid=os.getpid(), **value), allow_nan=False) + '\n')
        outgoing.flush()

    reply(dict(status='ready',protocol=1))
    for line in incoming:
        request = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be an object')
            if request.get('command') == 'stop':
                reply(dict(status='stopped', completed_frames=completed))
                return 0
            if request.get('command') != 'frame':
                raise ValueError('unknown consumer command')
            frame = request.get('frame')
            rig_id=request.get('rig_id')
            if rig_id not in rig_roots:
                raise ValueError('unknown_rig_id')
            key=[frame,rig_id]
            if type(frame) is not int or frame not in config['selected_frames'] or key in completed:
                raise ValueError('invalid or duplicate frame')
            start = time.monotonic()
            # stdout is the protocol only; per-frame diagnostics remain on disk.
            task_root=rig_roots[rig_id]
            with (task_root / f'frame_{frame:03d}' / 'consume.log').open('w') as log:
                with redirect_stdout(log):
                    consume(task_root, frame)
            completed.append(key)
            reply(dict(status='ok', frame=frame, rig_id=rig_id, seconds=time.monotonic()-start,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        except Exception as error:
            failure = dict(status='error', frame=request.get('frame') if isinstance(request,dict) else None,
                           rig_id=request.get('rig_id') if isinstance(request,dict) else None,
                           error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                           completed_frames=completed, failed_site_preserved=True)
            (root / 'consumer_failure.json').write_text(json.dumps(dict(pid=os.getpid(),**failure),indent=2)+'\n')
            reply(failure)
            return 1
    # The parent disappeared without a successful close; never clean its files.
    return 1


class FrameConsumer:
    """Blocking request/reply pipe with bounded waits and owned-child cleanup."""
    def __init__(self, command, *, env, cwd, log_path, deadline):
        self.deadline = deadline
        self.buffer = b''
        self.log = Path(log_path).open('w')
        self.process = None
        self.responses = []
        self.closed = False
        start = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, env=env, cwd=cwd, bufsize=0)
            ready = self._receive()
            if ready.get('status') != 'ready' or ready.get('protocol') != 1:
                raise RuntimeError(f'consumer did not initialize: {ready}')
            self.startup_seconds = time.monotonic()-start
        except BaseException:
            self.abort()
            raise

    def _remaining(self):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('CPU consumer deadline exhausted')
        return remaining

    def _receive(self):
        while b'\n' not in self.buffer:
            if not select.select([self.process.stdout], [], [], self._remaining())[0]:
                raise TimeoutError('CPU consumer response timeout')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('CPU consumer exited without a response; inspect consumer_worker.log')
            self.buffer += chunk
            if len(self.buffer) > 1024*1024:
                raise RuntimeError('oversized CPU consumer response')
        line, self.buffer = self.buffer.split(b'\n',1)
        value = json.loads(line)
        if value.get('pid') != self.process.pid:
            raise RuntimeError('CPU consumer response PID mismatch')
        return value

    def _send(self, value):
        self._remaining()
        if self.closed:
            raise RuntimeError('CPU consumer already closed')
        self.process.stdin.write((json.dumps(value)+'\n').encode())
        self.process.stdin.flush()

    def consume(self, frame, rig_id):
        start = time.monotonic()
        request=dict(command='frame',frame=frame)
        request['rig_id']=rig_id
        self._send(request)
        response = self._receive()
        if response.get('status') != 'ok' or response.get('frame') != frame or response.get('rig_id') != rig_id:
            raise RuntimeError(f'CPU frame consumption failed: {response}')
        response['roundtrip_seconds'] = time.monotonic()-start
        self.responses.append(response)
        return response

    def close(self):
        start = time.monotonic()
        self._send(dict(command='stop'))
        response = self._receive()
        expected=[[r['frame'],r['rig_id']] for r in self.responses]
        if response.get('status') != 'stopped' or response.get('completed_frames') != expected:
            raise RuntimeError(f'CPU consumer shutdown mismatch: {response}')
        self.process.stdin.close()
        code = self.process.wait(timeout=self._remaining())
        if code:
            raise RuntimeError(f'CPU consumer exited with {code}')
        self.process.stdout.close()
        self.log.close()
        self.closed = True
        return dict(pid=self.process.pid,startup_seconds=self.startup_seconds,
                    shutdown_seconds=time.monotonic()-start,returncode=code,frames=self.responses)

    def abort(self):
        # Only our own child; no worker restart and no deletion of frame files.
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in [self.process.stdin,self.process.stdout]:
                if stream and not stream.closed:
                    stream.close()
        if not self.log.closed:
            self.log.close()
        self.closed = True
