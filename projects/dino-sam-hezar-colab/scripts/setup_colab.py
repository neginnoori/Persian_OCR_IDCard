"""Install the combined pipeline in an isolated Colab/Linux environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=Path('/content/persian-id-venv'))
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
    pip('check')
    subprocess.run([python, '-c',
        "import cv2; from hezar.models import Model; from cardid.backend import require_cuda; "
        "import torch; require_cuda(torch,'cuda'); print('GPU:',torch.cuda.get_device_name(0)); "
        "print('Hezar import and GPU probe passed; weights load on first image.')"], check=True)
    print('Ready:', python)


if __name__ == '__main__':
    main()
