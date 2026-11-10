"""Six-field fallback on an oriented page, without inventing a card boundary.

Requires a consistent column of at least three label anchors. Unknown/ambiguous
pages still yield six rows, but never populate names from arbitrary stamp text.
"""
import re
from difflib import SequenceMatcher
import numpy as np
from .fields import FIELDS, assign_fields
from .reader import normalize_text

LABELS = [('شماره ملی', 'کد ملی'), ('نام',), ('نام خانوادگی',),
          ('تاریخ تولد',), ('نام پدر',), ('پایان اعتبار', 'تاریخ اعتبار')]


def label_prefix(text):
    text = normalize_text(text).strip()
    # OCR often misses the space separating a label and a numeric value.
    words = re.findall(r'[\u0600-\u06ff]+|[0-9/]+|[^\s]', text)
    hits = []
    for row, aliases in enumerate(LABELS):
        for label in aliases:
            n = len(label.split())
            if len(words) < n:
                continue
            prefix = ''.join(words[:n])
            target = label.replace(' ', '')
            if not all('\u0600' <= c <= '\u06ff' for c in prefix):
                continue
            similarity = SequenceMatcher(None, prefix, target).ratio()
            threshold = .80 if n == 1 else .72
            if similarity >= threshold:
                # Prefer the longer label (نام پدر before نام).
                hits.append((similarity + .15*(n-1), row, n))
    if not hits:
        return None
    _, row, n = max(hits)
    # Preserve value spacing using regex spans in the normalized original.
    tokens = list(re.finditer(r'[\u0600-\u06ff]+|[0-9/]+|[^\s]', text))
    value = text[tokens[n-1].end():].lstrip(' :؛،-')
    return row, value


def assign_anchored_fields(regions, size):
    fields = assign_fields([], size)
    for f in fields:
        f['assignment_method'] = 'label_anchors_v1'
    anchors = []
    for i,r in enumerate(regions):
        match = label_prefix(r.get('text_raw', ''))
        if match:
            anchors.append((match[0], i, r, match[1]))
    # More than one candidate for a label may mean multiple documents. Abstain.
    counts = {row: sum(a[0] == row for a in anchors) for row in range(6)}
    if any(n > 1 for n in counts.values()):
        return fields, {'reason': 'ambiguous_label_anchors', 'anchor_count': len(anchors)}
    if len(anchors) < 3:
        return fields, {'reason': 'insufficient_label_anchors', 'anchor_count': len(anchors)}
    anchors.sort()
    rows = np.array([a[0] for a in anchors], float)
    centers = np.array([(a[2]['box'][1]+a[2]['box'][3])/2 for a in anchors])
    right_edges = np.array([a[2]['box'][2] for a in anchors])
    heights = np.array([a[2]['box'][3]-a[2]['box'][1] for a in anchors])
    line_height = max(float(np.median(heights)), 1.)
    step, intercept = np.polyfit(rows, centers, 1)
    if (step < .9*line_height or np.max(np.abs(centers-(rows*step+intercept))) > .40*step
            or np.ptp(right_edges) > .12*size[0]):
        return fields, {'reason': 'inconsistent_label_layout', 'anchor_count': len(anchors)}
    anchor_ids = {a[1] for a in anchors}
    used = set()
    for row, anchor_id, anchor, inline in anchors:
        selected = []
        if inline:
            selected = [(anchor_id, anchor, inline)]
        else:
            ax1, ay1, ax2, ay2 = anchor['box']
            cy = (ay1+ay2)/2
            for i,r in enumerate(regions):
                if i in anchor_ids or i in used or not r.get('text_raw','').strip():
                    continue
                x1,y1,x2,y2 = r['box']
                if (x2 <= ax1+.15*line_height and ax1-x2 < .45*size[0]
                        and abs((y1+y2)/2-cy) <= .4*step):
                    selected.append((i,r,r['text_raw']))
            # Start next to the label; do not absorb a distant stamp on the left.
            selected.sort(key=lambda a:a[1]['box'][2], reverse=True)
            kept = selected[:1]
            for item in selected[1:]:
                prev = kept[-1][1]['box']; box = item[1]['box']
                overlap = min(prev[3],box[3])-max(prev[1],box[1])
                height = min(prev[3]-prev[1],box[3]-box[1])
                if overlap < .65*height or prev[0]-box[2] > .65*height:
                    break
                kept.append(item)
            selected = kept
        if not selected:
            continue
        numeric = row in (0,3,5)
        if numeric:
            selected.sort(key=lambda a:a[1]['box'][0])
        text = (' ' if not numeric else '').join(a[2] for a in selected)
        value = normalize_text(text)
        status = 'candidate'
        if numeric:
            runs = re.findall(r'[0-9]+(?:[ /]*[0-9]+)*', value)
            runs = list(dict.fromkeys(re.sub(r'\s+', '', x) for x in runs))
            value = runs[0] if len(runs) == 1 else None
            pattern = r'[0-9]{10}' if row == 0 else r'1[234][0-9]{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])'
            if not value or not re.fullmatch(pattern, value):
                status = 'invalid_format'
        boxes = [a[1]['box'] for a in selected]
        fields[row].update(text_raw=' '.join(a[1]['text_raw'] for a in selected), value=value or None,
                           box=[min(b[0] for b in boxes),min(b[1] for b in boxes),
                                max(b[2] for b in boxes),max(b[3] for b in boxes)],
                           source_region_ids=[a[0] for a in selected], status=status)
        used.update(a[0] for a in selected)
    return fields, {'reason': 'consistent_label_anchors', 'anchor_count': len(anchors),
                    'anchor_region_ids': sorted(anchor_ids)}
