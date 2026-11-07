"""Private JSON-lines worker; no Torch, Hezar or cardid imports in this process."""
import argparse
import base64
from contextlib import redirect_stdout
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from PIL import Image


def emit(payload):
    print('PADDLE_JSON:' + json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('FLAGS_allocator_strategy', 'auto_growth')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--device', choices=['cpu','gpu:0'], required=True)
    parser.add_argument('--model-dir')
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location('paddle_detection_impl', Path(__file__).with_name('detection.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            detector = module.PaddleTextDetector(args.cache, device=args.device, model_dir=args.model_dir).load()
        emit({'ready': True})
    except Exception as error:
        emit({'error': f'{type(error).__name__}: {error}'})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('close'):
                return 0
            with Image.open(BytesIO(base64.b64decode(request['image']))) as image:
                with redirect_stdout(sys.stderr):
                    regions = detector.detect(image.convert('RGB'))
            emit({'regions': regions})
        except Exception as error:
            emit({'error': f'{type(error).__name__}: {error}'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
