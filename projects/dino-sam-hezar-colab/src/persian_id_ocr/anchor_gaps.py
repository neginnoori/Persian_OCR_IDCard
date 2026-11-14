"""Recover unlabelled name rows inside an already established value column.

Used only by adaptive v3. No absolute card coordinates or identity vocabulary
are used, and interpolation never supplies text that OCR did not actually read.
"""
from copy import deepcopy

import numpy as np

from .anchors import label_prefix
from .reader import normalize_text


def supplement_name_rows(regions, proposals, evidence, compatible):
    output = deepcopy(proposals)
    audit = {'reason': 'insufficient_separate_values', 'rows': []}
    if evidence.get('reason') != 'consistent_label_anchors':
        audit['reason'] = 'no_consistent_anchors'
        return output, audit

    anchor_ids = set(evidence['anchor_region_ids'])
    supports = []
    for row, item in enumerate(proposals):
        ids = set(item['source_region_ids'])
        if (item['status'] == 'candidate' and compatible(item['field'], item['value'])
                and ids and not ids.intersection(anchor_ids)):
            supports.append((row, item))
    # Separate values establish the value-column edge; merged label/value boxes
    # cannot establish that edge. Require numeric evidence as well as names.
    if len(supports) < 3 or sum(row in (0, 3, 5) for row, _ in supports) < 2:
        return output, audit

    rows = np.array([row for row, _ in supports], dtype=float)
    boxes = np.array([item['box'] for _, item in supports], dtype=float)
    heights = boxes[:, 3] - boxes[:, 1]
    centers = (boxes[:, 1] + boxes[:, 3]) / 2
    step, origin = np.polyfit(rows, centers, 1)
    line_height = float(np.median(heights))
    right_edge = float(np.median(boxes[:, 2]))
    if (line_height <= 0 or step < .9 * line_height
            or np.max(np.abs(centers - (origin + step * rows))) > .25 * step
            or np.max(np.abs(boxes[:, 2] - right_edge)) > .5 * line_height):
        audit['reason'] = 'inconsistent_value_column'
        return output, audit

    audit.update(reason='consistent_value_column', row_step=float(step),
                 row_origin=float(origin), value_right_edge=right_edge,
                 support_fields=[item['field'] for _, item in supports])
    used = anchor_ids | {i for item in proposals for i in item['source_region_ids']}
    for row in (1, 2, 4):
        if proposals[row]['status'] != 'missing' or not rows.min() < row < rows.max():
            continue
        target_y = float(origin + step * row)
        candidates = []
        for i, region in enumerate(regions):
            text = region.get('text_raw', '')
            if i in used or label_prefix(text) or not compatible(proposals[row]['field'], text):
                continue
            x1, y1, x2, y2 = region['box']
            if (0 < y2 - y1 <= 1.4 * step and x1 < x2 <= right_edge + .35 * line_height
                    and x1 >= right_edge - 12 * line_height
                    and abs((y1 + y2) / 2 - target_y) <= .3 * step):
                candidates.append((i, region))

        # Group adjacent RTL words. Overlapping alternative detections remain
        # separate so two competing values cause abstention.
        candidates.sort(key=lambda pair: pair[1]['box'][2], reverse=True)
        groups = []
        for candidate in candidates:
            box = candidate[1]['box']
            if groups:
                previous = groups[-1][-1][1]['box']
                height = min(previous[3] - previous[1], box[3] - box[1])
                overlap = min(previous[3], box[3]) - max(previous[1], box[1])
                gap = previous[0] - box[2]
                if overlap >= .65 * height and -.1 * height <= gap <= .65 * height:
                    groups[-1].append(candidate)
                    continue
            groups.append([candidate])
        aligned = [group for group in groups
                   if abs(group[0][1]['box'][2] - right_edge) <= .5 * line_height]
        decision = {'field': proposals[row]['field'], 'expected_center_y': target_y,
                    'candidate_region_groups': [[i for i, _ in group] for group in aligned]}
        if len(aligned) != 1:
            decision['reason'] = 'ambiguous_row_values' if aligned else 'no_aligned_value'
            audit['rows'].append(decision)
            continue

        selected = aligned[0]
        raw = ' '.join(region['text_raw'] for _, region in selected)
        value = normalize_text(raw)
        if not compatible(proposals[row]['field'], value):
            continue
        boxes = [region['box'] for _, region in selected]
        output[row].update(text_raw=raw, value=value,
                           box=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                                max(b[2] for b in boxes), max(b[3] for b in boxes)],
                           source_region_ids=[i for i, _ in selected], status='candidate',
                           assignment_method='interpolated_anchor_row_v1')
        used.update(i for i, _ in selected)
        decision['reason'] = 'aligned_value_between_anchors'
        audit['rows'].append(decision)
    return output, audit
