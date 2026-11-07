"""Persistent isolated Paddle process, keeping its OpenCV/CUDA libraries separate."""
import atexit
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from threading import RLock, Thread


class PaddleWorkerDetector:
    def __init__(self, cache_dir, device=None, *, python=None, model_dir=None, timeout=180):
        self.cache_dir, self.device = Path(cache_dir), device
        self.python = python or os.environ.get('PERSIAN_PADDLE_PYTHON')
        config = Path(sys.prefix)/'paddle-worker.json'
        if config.is_file():
            settings = json.loads(config.read_text())
            self.python = self.python or settings['python']
            self.device = self.device or settings.get('device', 'cpu')
        self.device = self.device or 'cpu'
        self.model_dir, self.timeout = model_dir, timeout
        self.process = None
        self.responses = queue.Queue()
        self._lock = RLock()

    def _receive(self, timeout):
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('Paddle detector worker timed out; inspect its stderr')
        if response is None:
            self.close()
            raise RuntimeError('Paddle detector worker exited unexpectedly; inspect its stderr')
        if 'error' in response:
            raise RuntimeError('Paddle detection failed: ' + response['error'])
        return response

    def load(self):
        with self._lock:
            if self.process is not None:
                return self
            if not self.python or not Path(self.python).is_file():
                raise RuntimeError('Run scripts/setup_colab.py first, or set PERSIAN_PADDLE_PYTHON to the isolated Paddle Python executable')
            worker = Path(__file__).with_name('paddle_worker.py')
            command = [str(self.python), '-u', str(worker), '--cache', str(self.cache_dir), '--device', self.device]
            if self.model_dir is not None:
                command += ['--model-dir', str(self.model_dir)]
            self.responses = queue.Queue()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, encoding='utf-8', bufsize=1,
                                       env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))  # stderr is inherited for live diagnostics
            self.process = process
            responses = self.responses
            def consume():
                try:
                    for line in process.stdout:
                        if line.startswith('PADDLE_JSON:'):
                            try:
                                responses.put(json.loads(line[len('PADDLE_JSON:'):]))
                            except ValueError:
                                responses.put({'error': 'Malformed worker response'})
                finally:
                    responses.put(None)
            Thread(target=consume, daemon=True).start()
            try:
                if not self._receive(600).get('ready'):
                    raise RuntimeError('Paddle worker did not signal readiness')
            except Exception:
                self.close()
                raise
            atexit.register(self.close)
            return self

    def detect(self, card):
        with self._lock:
            self.load()
            buffer = BytesIO()
            card.convert('RGB').save(buffer, format='PNG')
            try:
                self.process.stdin.write(json.dumps({'image': base64.b64encode(buffer.getvalue()).decode('ascii')})+'\n')
                self.process.stdin.flush()
                return self._receive(self.timeout)['regions']
            except (BrokenPipeError, OSError) as error:
                self.close()
                raise RuntimeError('Paddle worker connection failed') from error

    def close(self):
        with self._lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
