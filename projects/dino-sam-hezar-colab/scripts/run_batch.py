"""One process reuses the extractor and Hezar models for all uploaded photos."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from PIL import ImageDraw
from persian_id_ocr import PersianCardOCR


def run_batch(inputs, output_root, cache_dir, *, pipeline=None, orientation_override=None,
              detector_backend='paddle', detector_device=None):
    run = Path(output_root)/datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    pipeline = pipeline if pipeline is not None else PersianCardOCR(
        cache_dir, detector_backend=detector_backend, detector_device=detector_device)
    summary = []
    for index, source in enumerate(inputs):
        name = re.sub(r'[^\w.-]', '_', Path(source).stem)[:80] or 'image'
        folder = run/f'{index:03d}_{name}'
        folder.mkdir()
        try:
            result = pipeline.process(source, orientation_override=orientation_override)
            report = result.to_dict()
            if result.image is not None:
                accepted = result.status in ('ocr_candidate', 'no_text')
                result.image.save(folder/('card.png' if accepted else 'card_needs_review.png'))
                if result.regions:
                    marked = result.image.copy()
                    draw = ImageDraw.Draw(marked)
                    for i, region in enumerate(result.regions):
                        draw.rectangle(region['box'], outline='red', width=2)
                        draw.text(tuple(region['box'][:2]), str(i), fill='red')
                    marked.save(folder/'text_boxes.png')
            import csv
            with (folder/'fields.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.DictWriter(handle, fieldnames=['field','label','text_raw','value','box','source_region_ids','status','review_required','assignment_method'])
                writer.writeheader()
                writer.writerows(result.fields)
            if result.image is not None:
                marked_fields = result.image.copy()
                draw_fields = ImageDraw.Draw(marked_fields)
                for item in result.fields:
                    if item['box'] is not None:
                        draw_fields.rectangle(item['box'], outline='blue', width=2)
                        draw_fields.text(tuple(item['box'][:2]), item['field'], fill='blue')
                        result.image.crop(item['box']).save(folder/f"field_{item['field']}.png")
                marked_fields.save(folder/'field_boxes.png')
            (folder/'text.txt').write_text(result.text, encoding='utf-8')
            (folder/'text_normalized.txt').write_text(result.text_normalized, encoding='utf-8')
        except Exception as error:
            report = {'status': 'error', 'error': f'{type(error).__name__}: {error}', 'review_required': True}
        (folder/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append({'input': str(source), 'folder': folder.name, 'status': report['status']})
        print(f"{index+1}/{len(inputs)}: {Path(source).name}: {report['status']}", flush=True)
    (run/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return run, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--orientation-override', type=int, choices=[0, 90, 180, 270])
    parser.add_argument('--text-detector', choices=['paddle','craft'], default='paddle')
    parser.add_argument('--paddle-device', choices=['cpu','gpu:0'])
    args = parser.parse_args()
    run, summary = run_batch(args.inputs, args.output_root, args.cache_dir,
                             orientation_override=args.orientation_override,
                             detector_backend=args.text_detector, detector_device=args.paddle_device)
    print('Results:', run, flush=True)
    return 1 if any(r['status'] == 'error' for r in summary) else (
        2 if any(r['status'] != 'ocr_candidate' for r in summary) else 0)


if __name__ == '__main__':
    raise SystemExit(main())

