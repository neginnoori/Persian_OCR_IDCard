"""Opt-in card selection. Legacy inference is untouched; thresholds are heuristics.

Audit coordinates always refer to the EXIF-corrected input. A good SAM mask is
not evidence that its contents are an ID card. No identity value is used as truth.
"""
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from .types import ExtractionResult
from .recovery import recovery_needed


def diverse_boxes(boxes, scores, limit=4):
    """Reserve a slot for the largest distinct proposal, not for duplicate stamps."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    areas = np.maximum(0, boxes[:, 2]-boxes[:, 0])*np.maximum(0, boxes[:, 3]-boxes[:, 1])
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    unique, audit = [], {}
    for i in order:
        row = {'dino_index': i, 'box': boxes[i].tolist(), 'score': float(scores[i])}
        duplicate = None
        for j in unique:
            extent = np.maximum(0, np.minimum(boxes[i, 2:], boxes[j, 2:])-np.maximum(boxes[i, :2], boxes[j, :2]))
            intersection = float(np.prod(extent))
            iou = intersection/max(areas[i]+areas[j]-intersection, 1.)
            if iou >= .85:
                duplicate = j
                break
        if areas[i] <= 0 or not np.isfinite(boxes[i]).all():
            row['reason'] = 'invalid_box'
        elif duplicate is not None:
            row.update(reason='duplicate', duplicate_of=duplicate)
        else:
            unique.append(i)
            row['reason'] = 'proposal_budget'
        audit[i] = row
    selected = unique[:limit]
    if unique and limit >= 2:
        largest = max(unique, key=lambda i: areas[i])
        if largest not in selected:
            selected[-1] = largest
    for i in selected:
        audit[i]['reason'] = 'segmented'
    return selected, [audit[i] for i in order]


def card_support(evidence):
    """Multiple independent signals; neither face nor a readable date is mandatory."""
    rows = min(6, int(evidence.get('value_row_count', 0)))
    span = float(evidence.get('value_row_span', 0))
    labels = set(evidence.get('matched_keywords', [])) & {'نام', 'خانوادگی', 'تولد', 'پدر', 'اعتبار', 'ملی'}
    face = bool(evidence.get('face_on_left'))
    dates = min(2, int(evidence.get('date_row_count', 0)))
    structured = rows >= 4 and span >= .28
    # Two independent families beyond row geometry are preferred. Strong labels
    # plus a face can support a low-contrast card with just three detected rows.
    supported = (structured and (len(labels) >= 2 or (face and dates >= 1))) or (
        rows >= 3 and span >= .23 and face and len(labels) >= 3)
    score = rows + min(span, .5)*4 + 2*len(labels) + 3*face + 2*dates
    if evidence.get('stamp_keywords') and not supported:
        score -= 5
    return {'supported': bool(supported), 'score': float(score),
            'row_count': rows, 'label_count': len(labels), 'face': face, 'date_count': dates}


def select_card_v2(pil, backend, config, baseline, override=None):
    from .geometry import quad_from_mask, warp_candidates, score_orientation
    started = perf_counter()
    audit = {'version': 'card_selection_v2', 'attempted': False, 'replaced': False,
             'reason': 'baseline_preserved', 'detections': [], 'candidates': [],
             'coordinate_space': 'exif_corrected_input'}
    if baseline is not None:
        meta = baseline.metadata
        chosen = {'quad': np.float32(meta['source_corners'])}
        audit['baseline_quad'] = meta['source_corners']
        if override is not None or not recovery_needed(chosen, meta, pil.size):
            baseline.metadata['candidate_selection'] = audit
            return baseline
    audit.update(attempted=True, reason='stamp_or_missing_card')
    detection = backend.detect_diverse_masks(pil)
    audit['detections'] = detection.candidates
    proposals = []
    for i, (mask, sam_score) in enumerate(zip(detection.masks, detection.sam_scores)):
        provenance = detection.mask_sources[i] if i < len(detection.mask_sources) else {}
        entry = {'id': f'mask_{i}', **provenance, 'sam_predicted_iou': float(sam_score)}
        try:
            quad, info = quad_from_mask(mask, min_iou=config.min_quad_iou)
        except ValueError as error:
            audit['candidates'].append({**entry, 'reason': 'no_quad', 'detail': str(error)})
            continue
        entry['quad'] = quad.tolist()
        # Keep at most one nearly identical mask quad; compare cyclically.
        if any(min(np.mean(np.linalg.norm(quad-np.roll(q, k, axis=0), axis=1))
                   for k in range(4)) < .015*max(pil.size) for q, _, _ in proposals):
            audit['candidates'].append({**entry, 'reason': 'duplicate_quad'})
            continue
        info.update(sam_predicted_iou=float(sam_score), **provenance)
        proposals.append((quad, info, entry))
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    accepted = []
    for quad, info, entry in proposals:
        options = warp_candidates(bgr, quad, config.width, config.height)
        scores = [score_orientation(c['card'], backend.reader, backend.face_detector) for c in options]
        supports = [card_support(e) for e in scores]
        entry['orientations'] = [{'degrees': i*90, **s, 'evidence': scores[i]} for i,s in enumerate(supports)]
        # Evaluate all rotations against the SAME card evidence used for selection.
        rank = sorted(range(4), key=lambda i: (supports[i]['supported'], supports[i]['score']), reverse=True)
        best = rank[0]
        entry.update(selected_degrees=best*90, support_score=supports[best]['score'],
                     reason='supported' if supports[best]['supported'] else 'insufficient_card_evidence')
        audit['candidates'].append(entry)
        if supports[best]['supported']:
            margin = supports[best]['score']-supports[rank[1]]['score']
            direction = dict(orientation_ok=margin >= config.min_orientation_margin,
                             orientation_method='card_layout_v2', corner_shift_degrees=best*90,
                             orientation_margin=margin, orientation_scores=scores)
            # Semantic evidence first; area and SAM quality cannot win on their own.
            area = abs(cv2.contourArea(quad))/(pil.width*pil.height)
            accepted.append((supports[best]['score'], .1*area, options[best], info, direction, entry['id']))
    if accepted:
        _, _, chosen, info, direction, selected_id = max(accepted, key=lambda x: x[:2])
        audit.update(replaced=True, reason='supported_card_selected', selected_id=selected_id)
        metadata = {**info, **direction, 'candidate_selection': audit,
                    'original_size': list(pil.size), 'output_size': [config.width, config.height],
                    'coordinate_space': 'exif_corrected_input', 'runtime': getattr(backend, 'info', {}),
                    'source_corners': chosen['quad'].tolist(),
                    'original_to_card_H': chosen['H'].tolist(),
                    'card_to_original_H': np.linalg.inv(chosen['H']).tolist()}
        reasons = []
        if not info.get('geometry_ok'):
            reasons.append('unreliable_geometry')
        if not direction['orientation_ok']:
            reasons.append('ambiguous_orientation')
        output = Image.fromarray(cv2.cvtColor(chosen['card'], cv2.COLOR_BGR2RGB))
        result = ExtractionResult('needs_review' if reasons else 'ok', output, ','.join(reasons) or None, metadata)
    else:
        # Never enlarge a stamp into an alleged card. Preserve all source pixels
        # and let Paddle find anchors; do not apply a card template to the page.
        rgb = np.asarray(pil)
        scores = []
        for k in range(4):
            preview = Image.fromarray(np.ascontiguousarray(np.rot90(rgb, k)))
            preview.thumbnail((1400, 1400))
            scores.append(score_orientation(cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR),
                                            backend.reader, backend.face_detector))
        supports = [card_support(e) for e in scores]
        rank = sorted(range(4), key=lambda i: (supports[i]['supported'], supports[i]['score']), reverse=True)
        k = rank[0] if override is None else override//90
        margin = supports[k]['score']-supports[rank[1]]['score']
        w, h = pil.size
        matrices = [np.eye(3), np.array([[0,1,0],[-1,0,w-1],[0,0,1]]),
                    np.array([[-1,0,w-1],[0,-1,h-1],[0,0,1]]),
                    np.array([[0,-1,h-1],[1,0,0],[0,0,1]])]
        matrix = matrices[k].astype(float)
        output = Image.fromarray(np.ascontiguousarray(np.rot90(rgb, k)))
        audit.update(reason='full_source_anchor_fallback', selected_id='full_source',
                     source_orientations=[{'degrees': i*90, **s, 'evidence': scores[i]} for i,s in enumerate(supports)])
        result = ExtractionResult('needs_review', output, 'unverified_card_frame', {
            'candidate_selection': audit, 'preprocessing_mode': 'orientation_only',
            'field_assignment_mode': 'anchors_v1', 'rotation_ccw_degrees': k*90,
            'perspective_warp_applied': False, 'card_frame': None,
            'orientation_ok': bool(override is not None or (supports[k]['supported'] and margin >= config.min_orientation_margin)),
            'orientation_scores': scores, 'original_size': [w,h], 'output_size': list(output.size),
            'coordinate_space': 'exif_corrected_input', 'runtime': getattr(backend, 'info', {}),
            'original_to_card_H': matrix.tolist(), 'card_to_original_H': np.linalg.inv(matrix).tolist(),
            'warnings': ['full_source_anchor_fallback']})
    result.metadata['elapsed_ms'] = (perf_counter()-started)*1000 + (baseline.metadata.get('elapsed_ms',0) if baseline else 0)
    return result
