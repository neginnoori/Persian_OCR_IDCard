"""Create a Python 3.12 runtime for Colab or a local terminal without changing its kernel.

Run with any existing Python: python scripts/setup_runtime.py --root .ocr-runtime
The last line is JSON containing the worker Python path.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".persian-ocr-runtime")
    parser.add_argument("--torch-index", choices=["cpu", "cu126"], default="cpu")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    package = Path(__file__).resolve().parents[1]
    bootstrap = root / "bootstrap"
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", MPLBACKEND="Agg")

    def run(command, **kwargs):
        return subprocess.run([str(x) for x in command], env=env, check=True, **kwargs)

    if not (bootstrap / "uv" / "__main__.py").is_file():
        run([sys.executable, "-m", "pip", "install", "--target", bootstrap, "uv>=0.9,<1"])
    uv_env = dict(env, PYTHONPATH=str(bootstrap), UV_PYTHON_INSTALL_DIR=str(root / "python"),
                  UV_CACHE_DIR=str(root / "uv-cache"), UV_NO_CONFIG="1")

    def uv(arguments, capture=False):
        return subprocess.run([sys.executable, "-m", "uv", *map(str, arguments)], env=uv_env,
                              check=True, capture_output=capture, text=True)

    uv(["python", "install", "3.12", "--no-bin"])
    managed = uv(["python", "find", "--managed-python", "3.12"], capture=True).stdout.strip()
    venv = root / "env"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        uv(["venv", "--python", managed, "--seed", venv])
    if not args.core_only:
        run([python, "-m", "pip", "install", "torch==2.7.1", "torchvision==0.22.1",
             "--index-url", "https://download.pytorch.org/whl/" + args.torch_index])
    target = str(package) + ("" if args.core_only else "[models]")
    run([python, "-m", "pip", "install", "-e", target])
    run([python, "-m", "pip", "check"])
    smoke = "import persian_ocr; print('persian_ocr', persian_ocr.__version__)"
    if not args.core_only:
        smoke += "; from paddleocr import TextDetection; from hezar.models import Model; print('Model imports OK')"
    run([python, "-c", smoke])
    print(json.dumps({"python": str(python), "package": str(package)}))


if __name__ == "__main__":
    main()
