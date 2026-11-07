"""Install the combined pipeline in an isolated Colab/Linux environment."""
import argparse
import os
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=Path('/content/persian-id-venv'))
    parser.add_argument('--paddle-device', choices=['cpu', 'gpu:0'], default='cpu')
    args = parser.parse_args()
    if os.name != 'posix' or not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error('Use Linux/Colab with Python 3.11–3.13 and an NVIDIA GPU')
    project = Path(__file__).resolve().parents[1]
    extractor = project/'vendor/cardid-extractor'
    if not extractor.is_dir():
        extractor = project.parents[1]/'packages/cardid-extractor'
    if not (extractor/'pyproject.toml').is_file():
        parser.error('Extractor source missing: use the complete delivered ZIP or repository')
    environment = args.env.resolve()
    if environment.exists() and not (environment/'pyvenv.cfg').is_file():
        parser.error(f'Refusing to overwrite a non-virtualenv directory: {environment}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'virtualenv>=20.26,<21'], check=True)
    subprocess.run([sys.executable, '-m', 'virtualenv', str(environment)], check=True)
    python = str(environment/'bin/python')
    def pip(*arguments):
        subprocess.run([python, '-m', 'pip', *arguments], check=True)
    pip('install', '--upgrade', 'pip')
    pip('install', 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cu126')
    pip('install', '-r', str(project/'requirements-colab.txt'), str(extractor), str(project))
    # Reinstall local packages even when their version numbers have not changed.
    pip('install', '--force-reinstall', '--no-deps', str(extractor), str(project))
    subprocess.run([python, str(project/'scripts/verify_install.py'), str(project)], check=True)
    pip('check')
    # PaddleX needs opencv-contrib while EasyOCR needs headless cv2. Separate venvs
    # prevent competing cv2 distributions and separate Torch/Paddle CUDA libraries.
    paddle_env = environment.with_name(environment.name+'-paddle')
    if paddle_env.exists() and not (paddle_env/'pyvenv.cfg').is_file():
        parser.error(f'Refusing to overwrite a non-virtualenv directory: {paddle_env}')
    subprocess.run([sys.executable, '-m', 'virtualenv', str(paddle_env)], check=True)
    paddle_python = str(paddle_env/'bin/python')
    runtime = ['paddlepaddle==3.2.2'] if args.paddle_device == 'cpu' else [
        'paddlepaddle-gpu==3.2.0', '--index-url', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/']
    subprocess.run([paddle_python, '-m', 'pip', 'install', *runtime], check=True)
    subprocess.run([paddle_python, '-m', 'pip', 'install', '-r', str(project/'requirements-paddle.txt')], check=True)
    subprocess.run([paddle_python, '-m', 'pip', 'check'], check=True)
    check_device = (
        "assert paddle.is_compiled_with_cuda(), 'CUDA Paddle required'; "
        "paddle.set_device('gpu:0'); (paddle.ones([1])+1).numpy(); "
    ) if args.paddle_device == 'gpu:0' else "paddle.set_device('cpu'); (paddle.ones([1])+1).numpy(); "
    probe_env = dict(os.environ, PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    subprocess.run([paddle_python, '-c', "import paddle; from paddleocr import TextDetection; "+check_device+
                    "print('Paddle detection runtime:', paddle.__version__)"], check=True, env=probe_env)
    (environment/'paddle-worker.json').write_text(json.dumps({'python':paddle_python, 'device':args.paddle_device}))
    subprocess.run([python, '-c',
        "import cv2; from hezar.models import Model; from cardid.backend import require_cuda; "
        "import torch; require_cuda(torch,'cuda'); print('GPU:',torch.cuda.get_device_name(0)); "
        "print('Hezar import and GPU probe passed; weights load on first image.')"], check=True)
    print('Ready:', python)
    print('Paddle detector:', args.paddle_device, 'in', paddle_python)


if __name__ == '__main__':
    main()

