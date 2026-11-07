"""Real Paddle/Hezar smoke test on already-upright cards; does not run extraction."""
import argparse
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw
from persian_id_ocr.reader import HezarReader
from persian_id_ocr.fields import assign_fields
from persian_id_ocr.paddle_client import PaddleWorkerDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('./models'))
    parser.add_argument('--paddle-python')
    parser.add_argument('--paddle-model-dir')
    parser.add_argument('--hezar-model-dir')
    parser.add_argument('--hezar-device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--paddle-device', default='cpu', choices=['cpu','gpu:0'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reader = HezarReader(args.cache/'hezar', device=args.hezar_device)
    reader.detector = PaddleWorkerDetector(args.cache/'paddle', device=args.paddle_device,
                                         python=args.paddle_python, model_dir=args.paddle_model_dir)
    if args.hezar_model_dir:
        from hezar.models import Model
        reader.recognizer = Model.load(args.hezar_model_dir, device=args.hezar_device)
        reader.recognizer.eval()
    else:
        # Load recognizer only; load() retains the explicitly configured detector.
        from hezar.models import Model
        reader.recognizer = Model.load('hezarai/crnn-base-fa-v2', device=args.hezar_device,
                                      cache_dir=str(args.cache/'hezar'))
        reader.recognizer.eval()
    summary, process_id = [], None
    try:
        for i, path in enumerate(args.images):
            with Image.open(path) as original:
                card = original.convert('RGB')
            started = time.perf_counter()
            regions = reader.read(card)
            if process_id is None:
                process_id = reader.detector.process.pid
            assert reader.detector.process.pid == process_id
            fields = assign_fields(regions, card.size)
            report = {'regions': regions, 'fields': fields, 'elapsed_seconds': time.perf_counter()-started}
            (args.output/f'sample_{i}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            annotated = card.copy()
            draw = ImageDraw.Draw(annotated)
            for region in regions:
                for polygon in region['source_polygons']:
                    draw.polygon([tuple(p) for p in polygon], outline='red', width=2)
            annotated.save(args.output/f'sample_{i}_boxes.png')
            summary.append({'sample': i, 'regions':len(regions),
                            'fields_with_values':sum(bool(f['value']) for f in fields),
                            'numeric_10digit_regions':sum(r['text'].isdigit() and len(r['text'])==10 for r in regions),
                            'seconds':round(report['elapsed_seconds'],3)})
        blank_empty = reader.detector.detect(Image.new('RGB',(856,540),'white')) == []
        if not blank_empty:
            raise RuntimeError('Blank-image detector smoke test failed')
        report = {'samples':summary, 'persistent_worker_reused':True, 'blank_image_empty':blank_empty,
                  'extraction_tested':False, 'accuracy_evaluated':False}
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        reader.close()


if __name__ == '__main__':
    main()
