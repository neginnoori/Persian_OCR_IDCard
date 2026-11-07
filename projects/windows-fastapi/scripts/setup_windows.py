"""Install the Windows/NVIDIA profile into two local, isolated environments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import struct
import subprocess
import sys
import venv

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
OCR_PROJECT = REPOSITORY / 'projects' / 'dino-sam-hezar-colab'
EXTRACTOR = REPOSITORY / 'packages' / 'cardid-extractor'


def run(*arguments: str | Path, env=None) -> None:
    child_env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    if env is not None:
        child_env.update(env)
    subprocess.run([str(arg) for arg in arguments], check=True, cwd=PROJECT, env=child_env)


def create_environment(path: Path) -> Path:
    if path.exists() and not (path / 'pyvenv.cfg').is_file():
        raise RuntimeError(f'Refusing to overwrite a non-virtualenv directory: {path}')
    if not path.exists():
        venv.EnvBuilder(with_pip=True).create(path)
    python = path / 'Scripts' / 'python.exe'
    if not python.is_file():
        raise RuntimeError(f'Not a Windows virtual environment: {path}')
    run(python, '-c', "import sys,struct; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Use Python 3.11 x64; create a fresh environment if the existing one differs'")
    return python


def main() -> None:
    if os.name != 'nt' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise SystemExit('This installer targets Windows x64 with an NVIDIA CUDA GPU.')
    if sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8:
        raise SystemExit('Install Python 3.11 x64, then run py -3.11 scripts/setup_windows.py')
    if not all((p / 'pyproject.toml').is_file() for p in (OCR_PROJECT, EXTRACTOR)):
        raise SystemExit('Download the complete repository; the two OCR packages are required.')
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi is None:
        raise SystemExit('NVIDIA driver/nvidia-smi not found. Install a suitable driver before setup.')
    run(nvidia_smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader')

    main_env = PROJECT / '.venv'
    paddle_env = PROJECT / '.venv-paddle'
    python = create_environment(main_env)
    paddle_python = create_environment(paddle_env)
    run(python, '-m', 'pip', 'install', '--upgrade', 'pip')
    run(paddle_python, '-m', 'pip', 'install', '--upgrade', 'pip')
    # Install CUDA wheels before the ordinary dependency profile. Do not replace
    # this index with the CPU index, or install Paddle into this environment.
    run(python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126')
    run(python, '-c', "import torch; from torch import cuda; assert cuda.is_available(), 'CUDA unavailable: check NVIDIA GPU and driver'; (torch.ones(1,device='cuda')+1).cpu(); print('CUDA runtime:',torch.version.cuda,'GPU:',cuda.get_device_name(0))")
    run(python, '-m', 'pip', 'install', '-r', PROJECT / 'requirements.txt', EXTRACTOR, OCR_PROJECT)
    run(python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', EXTRACTOR, OCR_PROJECT)
    run(python, OCR_PROJECT / 'scripts' / 'verify_install.py', OCR_PROJECT)
    run(python, '-m', 'pip', 'check')

    run(paddle_python, '-m', 'pip', 'install', '-r', PROJECT / 'requirements-paddle.txt')
    run(paddle_python, '-m', 'pip', 'check')
    probe_env = dict(os.environ, PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    run(paddle_python, '-c', "import paddle; from paddleocr import TextDetection; paddle.set_device('cpu'); (paddle.ones([1])+1).numpy(); print('Paddle CPU runtime:',paddle.__version__)", env=probe_env)
    (main_env / 'paddle-worker.json').write_text(
        json.dumps({'python': str(paddle_python), 'device': 'cpu'}, indent=2), encoding='utf-8')
    run(python, '-c', "from hezar.models import Model; from cardid.backend import require_cuda; import torch; require_cuda(torch,'cuda'); print('Hezar import and CUDA probe passed')")

    dotenv = PROJECT / '.env'
    if not dotenv.exists():
        example = (PROJECT / '.env.example').read_text(encoding='utf-8')
        # Never echo credentials to the console. Preserve an existing deployment key.
        with dotenv.open('x', encoding='utf-8') as target:
            target.write(example.replace('CHANGE_ME_TO_A_RANDOM_SECRET_AT_LEAST_32_CHARACTERS', secrets.token_urlsafe(32)))
    run(python, '-c', "from app.config import Settings; Settings(); print('API configuration validated')")
    print('Installation verified. A private API key is in .env; do not commit or share it publicly.')
    print('Run scripts/run.ps1. Model weights download on first startup; wait for readiness.')


if __name__ == '__main__':
    main()
