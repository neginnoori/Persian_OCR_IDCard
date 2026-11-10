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
    return quad, {'quad_method': method, 'mask_quad_iou': float(iou),
                  'mask_coverage': float(coverage), 'touches_image_border': clipped,
                  'geometry_ok': bool(iou >= min_iou and coverage >= .97 and inside and not clipped)}


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


def quad_iou(first, second):
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - intersection
    return intersection / union


def _card_outline(contour, shape):
    height, width = shape
    area = cv2.contourArea(contour)
    if not .15 < area / (height * width) < .97:
        return None
    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return None
    try:
        quad = order_points(approx)
    except ValueError:
        return None
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    opposing = (sides[::2].mean(), sides[1::2].mean())
    ratio = max(opposing) / min(opposing)
    coverage = area / cv2.contourArea(quad)
    if not 1.25 <= ratio <= 2.1 or coverage < .8:
        return None
    return coverage - .3 * abs(ratio - 1.586), quad


def card_edge_quads(bgr):
    """Propose card outlines from edges and large light or dark regions."""
    height, width = bgr.shape[:2]
    scale = min(1, 1024 / max(height, width))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    h, w = gray.shape
    edge_proposals = []
    for low, high in ((25, 75), (10, 35)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            proposal = _card_outline(contour, gray.shape)
            if proposal is None:
                continue
            _, quad = proposal
            if np.all(quad.min(axis=0) >= 2) and np.all(quad.max(axis=0) <= [w-3, h-3]):
                edge_proposals.append(proposal)
    distinct = []
    for _, quad in sorted(edge_proposals, key=lambda proposal: proposal[0], reverse=True):
        if all(quad_iou(quad, other) < .88 for other in distinct):
            distinct.append(quad)
            if len(distinct) == 2:
                break

    # A closed tone region can recover an outline broken at a stamp or the image border.
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    tone_proposals = []
    for region in (bright, cv2.bitwise_not(bright)):
        closed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            proposal = _card_outline(contour, gray.shape)
            if proposal is None:
                continue
            score, quad = proposal
            refined = _refine_edges(quad, contour)
            bounds = np.array([w - 1, h - 1], dtype=np.float32)
            margin = .02 * min(h, w)
            if refined is not None and np.all(refined >= -margin) and np.all(refined <= bounds + margin):
                clipped = np.clip(refined, 0, bounds)
                if quad_iou(quad, clipped) >= .88:
                    quad = clipped
            tone_proposals.append((score, quad))
    for _, quad in sorted(tone_proposals, key=lambda proposal: proposal[0], reverse=True):
        if all(quad_iou(quad, other) < .88 for other in distinct):
            distinct.append(quad)
            break
    bounds = np.array([width - 1, height - 1], dtype=np.float32)
    return [np.clip(quad / scale, 0, bounds) for quad in distinct]


def card_frame_quads(bgr):
    """Suggest full-bleed cards and cards separated from a white side margin."""
    height, width = bgr.shape[:2]
    portrait = height > width
    short_side, long_side = sorted((height, width))
    min_width = int(np.ceil(1.48 * short_side))
    max_width = int(1.70 * short_side)

    def outline(start, stop):
        quad = np.array([[start, 0], [stop - 1, 0], [stop - 1, short_side - 1],
                         [start, short_side - 1]], np.float32)
        return order_points(quad[:, ::-1]) if portrait else quad

    if min_width <= long_side <= max_width:
        return [outline(0, long_side)]
    if long_side < min_width:
        return []

    # Project near-white coverage along the long axis. A few stamps can sit inside the margin.
    near_white = np.min(bgr, axis=2) >= 240
    white = near_white.mean(axis=1 if portrait else 0)
    window = max(3, round(.025 * short_side))
    positions = np.arange(min_width, min(max_width, long_side - window) + 1)
    if not positions.size:
        return []
    proposals = []
    for reverse, profile in enumerate((white, white[::-1])):
        cumulative = np.r_[0., profile.cumsum()]
        left = (cumulative[positions] - cumulative[positions - window]) / window
        right = (cumulative[positions + window] - cumulative[positions]) / window
        margin = (cumulative[-1] - cumulative[positions]) / (long_side - positions)
        strength = np.where(margin >= .8, right - left, 0)
        best = int(np.argmax(strength))
        if strength[best] >= .45:
            boundary = int(positions[best])
            proposals.append(outline(long_side - boundary, long_side) if reverse else outline(0, boundary))
    return proposals


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
    field_rows = set()
    right_text_rows = set()
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
        center = points.mean(axis=0)
        if center[0] > .5*w and .20*h < center[1] < .87*h and len(clean) >= 3:
            right_text_rows.add(round(12 * center[1] / h))
        if center[0] > .45*w and .14*h < center[1] < .87*h:
            if any(label in clean for label in (
                'نام', 'خانوادگی', 'تولد', 'پدر', 'اعتبار', 'شمارهملی', 'کدملی'
            )):
                field_rows.add(round(12 * center[1] / h))
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
            'face_on_left': face_found, 'field_row_count': len(field_rows),
            'right_text_row_count': len(right_text_rows)}


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
    enough = evidence['field_row_count'] >= 2 or (
        evidence['face_on_left'] and evidence['field_row_count'] >= 1)
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
