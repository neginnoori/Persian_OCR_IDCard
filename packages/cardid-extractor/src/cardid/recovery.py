"""Additive recovery for a stamp selected as a card; never discard the baseline."""
import re
import cv2
import numpy as np


def layout_evidence(results, width, height):
    digits = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
    stamp_keywords, rows, date_rows = set(), [], []
    for box, text, confidence in results:
        if float(confidence) < .35:
            continue
        text = str(text).translate(digits).replace('ي', 'ی').replace('ك', 'ک')
        compact = re.sub(r'\s+', '', text)
        # Positive evidence only. Missing/poorly read card labels never reject a crop.
        stamp_keywords.update(word for word in ('کپی', 'برابراصل', 'شرکت', 'شمارهثبت') if word in compact)
        p = np.asarray(box, dtype=float).reshape(-1, 2)
        x1, y1 = p.min(axis=0) / [width, height]
        x2, y2 = p.max(axis=0) / [width, height]
        cy = (y1+y2)/2
        if not (.38 <= x1 < .85 and .55 <= x2 <= .99 and .20 <= cy <= .86):
            continue
        if not (.012 <= y2-y1 <= .12 and (x2-x1)*width >= 1.25*(y2-y1)*height):
            continue
        if len(re.sub(r'\W', '', compact)) < 2:
            continue
        rows.append(cy)
        if re.search(r'1[234][0-9]{2}/[0-9]{1,2}/[0-9]{1,2}', compact):
            date_rows.append(cy)
    def distinct(values):
        centers = []
        for y in sorted(values):
            if not centers or y-centers[-1] >= .055:
                centers.append(y)
        return centers
    centers = distinct(rows)
    return dict(stamp_keywords=sorted(stamp_keywords), value_row_count=len(centers),
                value_row_span=float(centers[-1]-centers[0]) if centers else 0.,
                date_row_count=len(distinct(date_rows)))


def supported_card(evidence):
    return bool(evidence.get('face_on_left') and evidence.get('value_row_count', 0) >= 4
                and evidence.get('value_row_span', 0) >= .30 and evidence.get('date_row_count', 0) >= 1)


def selected_evidence(orientation):
    scores = orientation.get('orientation_scores', [])
    index = int(orientation.get('corner_shift_degrees', 0)) // 90
    return scores[index] if index < len(scores) else {}


def recovery_needed(chosen, orientation, input_size):
    evidence = selected_evidence(orientation)
    area = abs(cv2.contourArea(np.asarray(chosen['quad'], np.float32)))
    return bool(evidence.get('stamp_keywords') and not supported_card(evidence)
                and not evidence.get('face_on_left') and not evidence.get('date_row_count', 0)
                and evidence.get('value_row_count', 0) <= 3
                and area < .60*input_size[0]*input_size[1])


def recover_stamp_candidate(pil, backend, config, chosen, orientation, metrics):
    # Import here because geometry also uses the pure layout evidence helper.
    from .geometry import quad_from_mask, warp_candidates, choose_orientation
    audit = {'attempted': False, 'replaced': False, 'reason': 'baseline_preserved', 'candidates': []}
    if not recovery_needed(chosen, orientation, pil.size):
        return chosen, orientation, metrics, audit
    audit.update(attempted=True, reason='stamp_evidence_in_small_crop')
    audit['baseline_evidence'] = selected_evidence(orientation)
    baseline_area = abs(cv2.contourArea(np.asarray(chosen['quad'], np.float32)))
    proposals = []
    if hasattr(backend, 'detect_alternative_masks'):
        detection = backend.detect_alternative_masks(pil)
        audit['alternative_detection_max_score'] = detection.dino_score
        for mask, score in zip(detection.masks, detection.sam_scores):
            try:
                quad, info = quad_from_mask(mask, min_iou=config.min_quad_iou)
            except ValueError:
                continue
            info.update(sam_predicted_iou=float(score), dino_score=None)
            if abs(cv2.contourArea(quad)) < 1.5*baseline_area:
                continue
            if any(np.max(np.linalg.norm(quad-q, axis=1)) < .03*max(pil.size) for q, _ in proposals):
                continue
            proposals.append((quad, info))
            if len(proposals) == 2:
                break
    # A full scanned card may have no useful border. This is an explicit approximate
    # frame proposal, not a fabricated SAM mask or a claim of verified geometry.
    w, h = pil.size
    if 1.25 <= max(w,h)/min(w,h) <= 1.9:
        quad = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        if abs(cv2.contourArea(quad)) >= 1.5*baseline_area:
            proposals.append((quad, {'quad_method': 'full_image_frame_recovery', 'geometry_ok': False,
                                     'geometry_failure_reasons': ['approximate_full_image_frame'],
                                     'sam_predicted_iou': None, 'dino_score': None}))
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    accepted = []
    for quad, info in proposals:
        candidate, direction = choose_orientation(warp_candidates(bgr, quad, config.width, config.height),
                                                  backend.reader, backend.face_detector,
                                                  min_margin=config.min_orientation_margin)
        evidence = selected_evidence(direction)
        supported = supported_card(evidence)
        audit['candidates'].append({'method': info['quad_method'], 'supported': supported, 'evidence': evidence})
        if supported:
            accepted.append((candidate, direction, info))
    if accepted:
        chosen, orientation, metrics = max(accepted, key=lambda item: (
            item[2].get('geometry_ok', False), selected_evidence(item[1]).get('score', 0)))
        audit.update(replaced=True, reason='larger_frame_with_face_and_aligned_value_rows')
    else:
        audit['reason'] = 'no_stronger_candidate_baseline_preserved'
    return chosen, orientation, metrics, audit
