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


def assign_fields(regions, size):
    width, height = size
    grouped = [[] for _ in FIELDS]
    for index, region in enumerate(regions):
        if not str(region.get('text_raw', '')).strip():
            continue
        x1, y1, x2, y2 = region['box']
        cx, cy = (x1+x2)/(2*width), (y1+y2)/(2*height)
        # Exclude portrait, heading and the label column. Never use known answers.
        if not (.38 <= cx <= .81 and x2/width <= .85):
            continue
        distances = [abs(cy-center) for center in CENTERS]
        row = min(range(6), key=lambda i: distances[i])
        ordered = sorted(distances)
        if distances[row] > .055 or ordered[1]-ordered[0] < .012:
            continue
        grouped[row].append((index, region))
    output = []
    for (field, label), items in zip(FIELDS, grouped):
        numeric = field in ('national_id', 'date_of_birth', 'expiry_date')
        items.sort(key=lambda item: item[1]['box'][0], reverse=not numeric)
        raw = ('' if numeric else ' ').join(str(r['text_raw']) for _, r in items)
        value = normalize_text(raw)
        box = None
        if items:
            boxes = [r['box'] for _, r in items]
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes)]
        status = 'candidate' if value else 'missing'
        if value and numeric:
            pattern = r'\d{10}' if field == 'national_id' else r'1[234]\d{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])'
            if not re.fullmatch(pattern, value):
                status = 'invalid_format'
        output.append({'field': field, 'label': label, 'text_raw': raw, 'value': value or None,
                       'box': box, 'source_region_ids': [i for i, _ in items],
                       'status': status, 'review_required': True,
                       'assignment_method': 'standard_front_layout_v1'})
    return output
