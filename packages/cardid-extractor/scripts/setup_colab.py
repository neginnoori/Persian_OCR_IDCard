"""Create an isolated Colab/Linux virtualenv without changing notebook ML packages.

Usage: python scripts/setup_colab.py [--env /content/cardid-venv]
Requires network for virtualenv, packages and CUDA wheels. Weights download at first load.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path("/content/cardid-venv"))
    args = parser.parse_args()
    if os.name == "nt":
        parser.error("This helper targets Colab/Linux. See README.fa.md for Windows installation.")
    if not (3,10) <= sys.version_info[:2] < (3,14):
        parser.error("Use Python 3.10-3.13")
    project = Path(__file__).resolve().parents[1]
    environment = args.env.resolve()
    if environment.exists() and not (environment/"pyvenv.cfg").exists():
        parser.error(f"Refusing to use an existing non-virtualenv directory: {environment}")
    # virtualenv avoids Debian/Colab ensurepip availability problems. It does not install ML packages globally.
    subprocess.run([sys.executable,"-m","pip","install","virtualenv>=20.26,<21"],check=True)
    subprocess.run([sys.executable,"-m","virtualenv",str(environment)],check=True)
    python = environment/"bin/python"
    subprocess.run([str(python),"-m","pip","install","--upgrade","pip"],check=True)
    subprocess.run([str(python),"-m","pip","install","torch==2.7.1","torchvision==0.22.1",
                    "--index-url","https://download.pytorch.org/whl/cu126"],check=True)
    subprocess.run([str(python),"-m","pip","install","-r",str(project/"requirements-colab.txt"),str(project)],check=True)
    subprocess.run([str(python),"-m","pip","check"],check=True)
    subprocess.run([str(python),"-c",
        "from cardid.backend import require_cuda; import torch; require_cuda(torch,'cuda'); "
        "print('CUDA GPU:',torch.cuda.get_device_name(0))"],check=True)
    print("Ready. Run examples with:",python)


if __name__ == "__main__":
    main()
