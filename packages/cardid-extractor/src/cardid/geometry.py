"""Perspective and four-way orientation helpers; coordinates use EXIF-corrected pixels."""
import re
import cv2
import numpy as np


def order_points(pts):
    """Cyclic, non-duplicating order: image top-left first, then clockwise."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    if not np.isfinite(pts).all() or len(np.unique(pts, axis=0)) != 4:
        raise ValueError('Four distinct finite corners are required')
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)) or abs(cv2.contourArea(pts)) < 4:
        raise ValueError('Invalid card quadrilateral')
    return pts


def _quad_metrics(quad, mask):
    poly = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(poly, np.rint(quad).astype(np.int32), 1)
    intersection = int(np.count_nonzero(poly & mask))
    union = int(np.count_nonzero(poly | mask))
    return intersection / max(union, 1), intersection / max(int(mask.sum()), 1)


def _refine_edges(quad, contour):
    """Fit straight edge interiors, then intersect adjacent lines (rounded corners excluded)."""
    points = contour.reshape(-1, 2).astype(np.float32)
    diagonal = max(np.linalg.norm(quad[0]-quad[2]), np.linalg.norm(quad[1]-quad[3]))
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        edge = b-a
        length = np.linalg.norm(edge)
        projection = ((points-a) @ edge) / (length*length)
        distance = np.abs(edge[0]*(points[:, 1]-a[1])-edge[1]*(points[:, 0]-a[0])) / length
        selected = points[(projection > .12) & (projection < .88) & (distance < .025*diagonal)]
        if len(selected) < 8:
            return None
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
        lines.append((np.array([x, y]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        p, u = lines[(i-1) % 4]
        q, v = lines[i]
        system = np.column_stack((u, -v))
        if abs(np.linalg.det(system)) < .08:
            return None
        t = np.linalg.solve(system, q-p)[0]
        corners.append(p+t*u)
    try:
        return order_points(corners)
    except ValueError:
        return None


def quad_from_mask(mask, min_iou=.88):
    """Recover a projective quadrilateral; never substitute a rotated rectangle."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError('Expected one 2-D card mask')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('Empty SAM mask')
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        raise ValueError('Card mask is too small')
    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 1, cv2.FILLED)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    candidates = []
    seen = set()
    for epsilon in np.linspace(.002, .08, 100):
        approx = cv2.approxPolyDP(hull, float(epsilon*perimeter), True)
        if len(approx) != 4:
            continue
        try:
            q = order_points(approx)
        except ValueError:
            continue
        signature = tuple(q.reshape(-1))
        if signature in seen:
            continue
        seen.add(signature)
        for variant, method in [(q, 'hull_quad'), (_refine_edges(q, contour), 'edge_intersections')]:
            if variant is None:
                continue
            iou, coverage = _quad_metrics(variant, component)
            candidates.append((iou, coverage, variant, method))
    if not candidates:
        raise ValueError('Four reliable card edges could not be recovered')
    iou, coverage, quad, method = max(candidates, key=lambda x: x[0])
    height, width = mask.shape
    clipped = bool(component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any())
    inside = bool((quad[:, 0] >= 0).all() and (quad[:, 0] <= width-1).all()
                  and (quad[:, 1] >= 0).all() and (quad[:, 1] <= height-1).all())
    frame = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    # A tightly cropped, upright card can legitimately occupy the whole image.
    # Require evidence on all four sides, rather than accepting any border contact.
    near_frame = bool((np.abs(quad-frame) <= np.float32([.04*width, .04*height])).all())
    all_sides = bool(component[0].any() and component[-1].any()
                     and component[:, 0].any() and component[:, -1].any())
    tight_crop = bool(inside and all_sides and near_frame
                      and component.sum()/(width*height) >= .90)
    border_ok = not clipped or tight_crop
    return quad, {'quad_method': method, 'mask_quad_iou': float(iou),
                  'mask_coverage': float(coverage), 'touches_image_border': clipped,
                  'border_status': 'tight_crop_candidate' if tight_crop else ('clipped_or_uncertain' if clipped else 'clear'),
                  'geometry_ok': bool(iou >= min_iou and coverage >= .97 and inside and border_ok)}


def warp_candidates(img, quad, width=856, height=540):
    """Four cyclic corner assignments; each H maps original pixels directly to final pixels."""
    quad = order_points(quad)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    # Start with the longer opposing edge pair as horizontal; text determines the final direction.
    if lengths[0]+lengths[2] < lengths[1]+lengths[3]:
        quad = np.roll(quad, -1, axis=0)
    destination = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    candidates = []
    for turns in range(4):
        source = np.roll(quad, -turns, axis=0).copy()
        matrix = cv2.getPerspectiveTransform(source, destination)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
            raise ValueError('Degenerate perspective transformation')
        card = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)
        candidates.append({'card': card, 'H': matrix, 'quad': source,
                           'corner_shift_degrees': turns*90})
    return candidates


def _normalize_persian(text):
    return re.sub(r'[^\w]', '', str(text).replace('ي', 'ی').replace('ك', 'ک').replace('\u200c', ''))


def score_orientation(card, reader, face_detector=None):
    """Heuristic evidence, not a calibrated probability. Do not rotate individual OCR boxes."""
    rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1, paragraph=False, rotation_info=None,
                              decoder='greedy', batch_size=1, workers=0)
    h, w = card.shape[:2]
    labels = ['نام', 'خانوادگی', 'تولد', 'پدر', 'اعتبار', 'ملی', 'ایران', 'جمهوری', 'اسلامی']
    matched = set()
    text_evidence = 0.0
    valid_count = 0
    for box, text, confidence in results:
        confidence = float(confidence)
        if confidence < .35:
            continue
        points = np.asarray(box, np.float32)
        box_width = float(np.linalg.norm(points[1]-points[0]))
        box_height = float(np.linalg.norm(points[3]-points[0]))
        if box_width < 1.25*box_height:
            continue
        clean = _normalize_persian(text)
        if len(clean) < 2:
            continue
        matched.update(label for label in labels if label in clean)
        text_evidence += confidence*min(len(clean), 12)/12
        valid_count += 1
    face_found = False
    if face_detector is not None and not face_detector.empty():
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(max(24, w//18), max(24, h//12)))
        for x, y, fw, fh in faces:
            cx, cy = (x+fw/2)/w, (y+fh/2)/h
            if .02 < cx < .36 and .18 < cy < .82 and .05 < fw/w < .34:
                face_found = True
                break
    score = 3*len(matched) + min(text_evidence, 4.0) + 3*int(face_found)
    return {'score': float(score), 'keyword_count': len(matched),
            'matched_keywords': sorted(matched), 'text_count': valid_count,
            'face_on_left': face_found}


def choose_orientation(candidates, reader, face_detector=None, override=None, min_margin=1.2):
    """Override 0/90/180/270 is a cyclic corner shift, relative to the geometric base warp."""
    if override is not None:
        if override not in (0, 90, 180, 270):
            raise ValueError('Orientation override must be 0, 90, 180 or 270')
        chosen = next(c for c in candidates if c['corner_shift_degrees'] == override)
        return chosen, {'orientation_ok': True, 'orientation_method': 'manual',
                        'corner_shift_degrees': override, 'orientation_scores': []}
    scores = [score_orientation(c['card'], reader, face_detector) for c in candidates]
    rank = sorted(range(4), key=lambda i: scores[i]['score'], reverse=True)
    best, second = rank[:2]
    evidence = scores[best]
    margin = evidence['score']-scores[second]['score']
    enough = evidence['keyword_count'] >= 2 or (evidence['face_on_left'] and evidence['keyword_count'] >= 1)
    ok = bool(enough and margin >= min_margin)
    return candidates[best], {'orientation_ok': ok, 'orientation_method': 'persian_text_and_face',
                             'corner_shift_degrees': candidates[best]['corner_shift_degrees'],
                             'orientation_margin': float(margin), 'orientation_scores': scores}


def rectify_card(img, quad, reader=None, face_detector=None, override=None, width=856, height=540):
    candidates = warp_candidates(img, quad, width, height)
    if reader is None and override is None:
        raise ValueError('An orientation reader or manual override is required')
    chosen, metadata = choose_orientation(candidates, reader, face_detector, override)
    return chosen['card'], chosen['H'], chosen['quad'], metadata

