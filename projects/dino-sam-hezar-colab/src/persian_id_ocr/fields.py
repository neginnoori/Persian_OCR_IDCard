"""Six-field assignment for the standard front-side Iranian ID layout.

Coordinates refer to the accepted upright card. This is a geometric template,
not a trained field classifier or a guarantee of transcription accuracy.
"""
import re
from .reader import normalize_text

FIELDS = [('national_id', 'کد ملی'), ('first_name', 'نام'),
          ('last_name', 'نام خانوادگی'), ('date_of_birth', 'تاریخ تولد'),
          ('father_name', 'نام پدر'), ('expiry_date', 'پایان اعتبار')]
CENTERS = [.28, .38, .48, .58, .68, .78]


def assign_fields(regions, size, *, frame=None):
    width, height = size
    left = top = 0
    if frame is not None:
        left, top, right, bottom = frame
        width, height = right-left, bottom-top
        if width <= 0 or height <= 0:
            raise ValueError('Invalid card frame')
    grouped = [[] for _ in FIELDS]
    for index, region in enumerate(regions):
        if not str(region.get('text_raw', '')).strip():
            continue
        x1, y1, x2, y2 = region['box']
        cx, cy = (x1+x2-2*left)/(2*width), (y1+y2-2*top)/(2*height)
        distances = [abs(cy-center) for center in CENTERS]
        row = min(range(6), key=lambda i: distances[i])
        ordered = sorted(distances)
        if distances[row] > .055 or ordered[1]-ordered[0] < .012:
            continue
        # Paddle can merge a numeric value with its label. Admit that region
        # only in the expected row, spanning the value and label columns.
        normal_value = .38 <= cx <= .81 and (x2-left)/width <= .85
        normalized = normalize_text(region['text_raw'])
        mixed_numeric = (
            row in (0, 3, 5)
            and .38 <= (x1-left)/width <= .79
            and .85 < (x2-left)/width <= 1.02
            and re.search(r'[0-9]', normalized) is not None
        )
        if not (normal_value or mixed_numeric):
            continue
        item = dict(region)
        if mixed_numeric:
            # Preserve the full OCR string for audit; extract one numeric run.
            # Never fill missing digits or pick one of several unrelated numbers.
            runs = re.findall(r'[0-9]+(?:[ /]*[0-9]+)*', normalized)
            item['field_value'] = re.sub(r'\s+', '', runs[0]) if len(runs) == 1 else normalized
            item['mixed_label_value'] = True
        grouped[row].append((index, item))
    output = []
    for (field, label), items in zip(FIELDS, grouped):
        numeric = field in ('national_id', 'date_of_birth', 'expiry_date')
        items.sort(key=lambda item: item[1]['box'][0], reverse=not numeric)
        raw = ('' if numeric else ' ').join(str(r['text_raw']) for _, r in items)
        value = normalize_text(('' if numeric else ' ').join(
            str(r.get('field_value', r['text_raw'])) for _, r in items))
        if not numeric and items:
            # Start at the value column's rightmost region. Extend toward the
            # left only across small word gaps on the same baseline.
            ordered_items = sorted(items, key=lambda pair: pair[1]['box'][2], reverse=True)
            kept = [ordered_items[0]]
            for candidate in ordered_items[1:]:
                a = kept[-1][1]['box']
                b = candidate[1]['box']
                ah, bh = a[3]-a[1], b[3]-b[1]
                overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
                gap = max(0, a[0]-b[2])
                if min(ah, bh) <= 0 or overlap/min(ah, bh) < .65 or gap > .65*min(ah, bh):
                    break
                kept.append(candidate)
            items = kept
            raw = ' '.join(str(r['text_raw']) for _, r in items)
            value = normalize_text(raw)
        numeric_unresolved = False
        if numeric:
            # Extract complete numeric runs per region; never join unrelated
            # stamp numbers or erase letters inside a number to invent a value.
            candidates = []
            for index, region in items:
                normalized = normalize_text(region['text_raw'])
                runs = re.findall(r'[0-9]+(?:[ /]*[0-9]+)*', normalized)
                for run in runs:
                    token = re.sub(r'\s+', '', run)
                    if field == 'national_id' and '/' in token:
                        continue
                    candidates.append((token, index, region))
            unique = {token for token, _, _ in candidates}
            numeric_unresolved = bool(items) and len(unique) != 1
            if len(unique) == 1:
                value = next(iter(unique))
                items = [(index, region) for token, index, region in candidates]
            else:
                value = ''
        box = None
        if items:
            boxes = [r['box'] for _, r in items]
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes)]
        status = 'invalid_format' if numeric_unresolved else ('candidate' if value else 'missing')
        if value and numeric:
            pattern = r'\d{10}' if field == 'national_id' else r'1[234]\d{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])'
            if not re.fullmatch(pattern, value):
                status = 'invalid_format'
        output.append({'field': field, 'label': label, 'text_raw': raw, 'value': value or None,
                       'box': box, 'source_region_ids': [i for i, _ in items],
                       'status': status, 'review_required': True,
                       'assignment_method': ('card_frame_layout_approximate_v2' if frame is not None else 'standard_front_layout_v2')})
    return output
