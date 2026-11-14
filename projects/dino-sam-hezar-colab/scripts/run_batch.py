"""One process reuses the extractor and Hezar models for all uploaded photos."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from PIL import ImageDraw
from persian_id_ocr import PersianCardOCR


def save_selection_audit(source, report, folder):
    from cardid.images import normalize_image
    audit = report.get('metadata', {}).get('extraction', {}).get('metadata', {}).get('candidate_selection')
    if audit is None:
        return
    (folder/'candidate_selection.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    original = normalize_image(source)
    original.thumbnail((1200,1200))
    width, height = report['metadata']['extraction']['metadata']['original_size']
    sx, sy = original.width/width, original.height/height
    draw = ImageDraw.Draw(original)
    for row in audit.get('detections', []):
        x1,y1,x2,y2 = row['box']
        draw.rectangle([x1*sx,y1*sy,x2*sx,y2*sy], outline='orange', width=2)
        draw.text((x1*sx,y1*sy), f"D{row['dino_index']} {row['reason']}", fill='orange')
    for row in audit.get('candidates', []):
        if 'quad' not in row:
            continue
        points = [(x*sx,y*sy) for x,y in row['quad']]
        selected = row['id'] == audit.get('selected_id')
        color = 'lime' if selected else 'red'
        draw.line(points+[points[0]], fill=color, width=3)
        draw.text(points[0], row['id']+' '+row['reason'], fill=color)
    if audit.get('baseline_quad'):
        points = [(x*sx,y*sy) for x,y in audit['baseline_quad']]
        draw.line(points+[points[0]], fill='blue', width=2)
    original.save(folder/'candidate_boxes.png')


def run_batch(inputs, output_root, cache_dir, *, pipeline=None, orientation_override=None,
              detector_backend='paddle', detector_device=None, preprocessing='rectify_best_effort',
              candidate_selection='legacy', bsrgan_model_path=None, bsrgan_source_dir=None):
    run = Path(output_root)/datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    pipeline = pipeline if pipeline is not None else PersianCardOCR(
        cache_dir, detector_backend=detector_backend, detector_device=detector_device,
        preprocessing=preprocessing, candidate_selection=candidate_selection,
        bsrgan_model_path=bsrgan_model_path, bsrgan_source_dir=bsrgan_source_dir)
    summary = []
    for index, source in enumerate(inputs):
        name = re.sub(r'[^\w.-]', '_', Path(source).stem)[:80] or 'image'
        folder = run/f'{index:03d}_{name}'
        folder.mkdir()
        try:
            result = pipeline.process(source, orientation_override=orientation_override)
            report = result.to_dict()
            save_selection_audit(source, report, folder)
            if result.image is not None:
                accepted = result.status in ('ocr_candidate', 'no_text')
                result.image.save(folder/('card.png' if accepted else 'card_needs_review.png'))
                enhanced_preview = getattr(getattr(pipeline, 'reader', None), 'enhanced_preview', None)
                if enhanced_preview is not None:
                    enhanced_preview.save(folder/'bsrgan_card_x2.png')
                if result.regions:
                    marked = result.image.copy()
                    draw = ImageDraw.Draw(marked)
                    for i, region in enumerate(result.regions):
                        added_by_bsrgan = 'bsrgan_x2' in region.get('detection_sources', [])
                        color = 'orange' if added_by_bsrgan else 'red'
                        draw.rectangle(region['box'], outline=color, width=2)
                        draw.text(tuple(region['box'][:2]), str(i), fill=color)
                    marked.save(folder/'text_boxes.png')
            import csv
            with (folder/'fields.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                base_columns = ['field','label','text_raw','value','box','source_region_ids',
                                'status','review_required','assignment_method','numeric_retry']
                extra_columns = [key for item in result.fields for key in item if key not in base_columns]
                writer = csv.DictWriter(handle, fieldnames=base_columns + list(dict.fromkeys(extra_columns)))
                writer.writeheader()
                writer.writerows({**item, 'numeric_retry': json.dumps(item.get('numeric_retry'), ensure_ascii=False)} for item in result.fields)
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
    parser.add_argument('--preprocessing', choices=['orientation_only', 'rectify', 'rectify_best_effort'], default='rectify_best_effort')
    parser.add_argument('--candidate-selection', choices=['legacy', 'v2', 'v3'], default='legacy')
    parser.add_argument('--bsrgan-model', type=Path)
    parser.add_argument('--bsrgan-source-dir', type=Path)
    args = parser.parse_args()
    run, summary = run_batch(args.inputs, args.output_root, args.cache_dir,
                             orientation_override=args.orientation_override,
                             detector_backend=args.text_detector, detector_device=args.paddle_device,
                             preprocessing=args.preprocessing, candidate_selection=args.candidate_selection,
                             bsrgan_model_path=args.bsrgan_model, bsrgan_source_dir=args.bsrgan_source_dir)
    print('Results:', run, flush=True)
    return 1 if any(r['status'] == 'error' for r in summary) else (
        2 if any(r['status'] != 'ocr_candidate' for r in summary) else 0)


if __name__ == '__main__':
    raise SystemExit(main())

