"""DocAligner ONNX + Hezar. RGB PIL public API; no ground-truth at inference."""
from pathlib import Path
import hashlib
import json
import os
import re
import time
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image, ImageOps

DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')


def normalize_text(text):
    text = str(text).translate(DIGITS).replace('ي', 'ی').replace('ك', 'ک')
    return re.sub(r'\s+', ' ', text).strip()


def read_rgb(source):
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert('RGB')
    with Image.open(source) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def resize_limit(image, limit):
    w, h = image.size
    factor = min(1.0, limit / max(w, h))
    return image.resize((max(1, round(w * factor)), max(1, round(h * factor))),
                        Image.Resampling.LANCZOS)


def order_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0).copy()


def validate_quad(points, size):
    w, h = size
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return 'corners_not_found'
    p = order_quad(p)
    if not cv2.isContourConvex(p.reshape(-1, 1, 2)):
        return 'invalid_quad'
    if (p[:, 0] < 0).any() or (p[:, 0] > w-1).any() or (p[:, 1] < 0).any() or (p[:, 1] > h-1).any():
        return 'corners_outside_image'
    area_fraction = abs(cv2.contourArea(p)) / (w*h)
    sides = np.linalg.norm(p-np.roll(p, 1, axis=0), axis=1)
    if not .015 <= area_fraction <= .995 or sides.min() < 15:
        return 'invalid_quad'
    # Geometry checks cannot certify that the selected object is an ID card.
    return 'geometry_candidate'


def warp_card(image, corners, max_side=1800):
    p = order_quad(corners)
    width = max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))
    height = max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))
    scale = min(1., max_side / max(width, height))
    w, h = max(2, round(width*scale)), max(2, round(height*scale))
    target = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    matrix = cv2.getPerspectiveTransform(p, target)
    return Image.fromarray(cv2.warpPerspective(np.array(image), matrix, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))


class CardOCR:
    """Single-card images. Locator runs on CPU; Hezar can use CUDA."""
    WEIGHTS = {
        'fastvit_t8': ('fastvit_t8_h_e_bifpn_256_fp32.onnx', '1W88oOkHjMza6xXbDxrG8k_C-Tf2aYKJ4'),
        'fastvit_sa24': ('fastvit_sa24_h_e_bifpn_256_fp32.onnx', '14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl'),
        'lcnet100': ('lcnet100_h_e_bifpn_256_fp32.onnx', '1IlbLPkCv-TdaBLOPh_4J97P1_KHYYJ7a'),
    }

    def __init__(self, model_dir, variant='fastvit_t8', device='cpu'):
        import gdown
        import onnxruntime as ort
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('HF_HOME', str(self.model_dir/'huggingface'))
        from hezar.models import Model
        filename, file_id = self.WEIGHTS[variant]
        path = self.model_dir / filename
        if not path.is_file():
            temp = path.with_suffix('.download')
            gdown.download(id=file_id, output=str(temp), quiet=False, use_cookies=False)
            if not temp.is_file() or temp.stat().st_size < 100_000:
                raise RuntimeError('دانلود وزن DocAligner کامل نشد. اتصال/محدودیت Google Drive را بررسی کنید.')
            temp.replace(path)
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.variant = variant
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.detector = Model.load('hezarai/CRAFT', device=device)
        self.reader = Model.load('hezarai/crnn-base-fa-v2', device=device)
        self.detector.eval()
        self.reader.eval()
        self.processor = self.detector.preprocessor.image_processor
        # Detect on <=1280 pixels. ratio=1 avoids Hezar 1.0.0's inverse-scale bug.
        self.processor.config.square_size = 2560
        self.processor.config.mag_ratio = 1.0

    def locate(self, image):
        # Official DocAligner heatmap weights expect BGR, NCHW, [0,1], 256x256.
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        tensor = cv2.resize(bgr, (256, 256)).transpose(2, 0, 1)[None].astype(np.float32)/255.
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        named = dict(zip([x.name for x in self.session.get_outputs()], outputs))
        heatmaps = named.get('heatmap')
        if heatmaps is None:
            raise RuntimeError(f'خروجی heatmap پیدا نشد: {list(named)}')
        if heatmaps.ndim != 4 or heatmaps.shape[1] != 4:
            raise RuntimeError(f'شکل heatmap غیرمنتظره: {heatmaps.shape}')
        # Adaptation of the official threshold/contour-centroid decoder.
        # Decode at capped resolution, then map to the original image.
        preview = resize_limit(image, 1200)
        sw, sh = preview.size
        points, peaks = [], []
        for heat in heatmaps[0]:
            peaks.append(float(heat.max()))
            resized = cv2.resize(heat, (sw, sh), interpolation=cv2.INTER_LINEAR)
            mask = np.uint8(resized >= .3)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, peaks
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                return None, peaks
            points.append([moments['m10']/moments['m00']*image.width/sw,
                           moments['m01']/moments['m00']*image.height/sh])
        return np.float32(points), peaks

    def text_boxes(self, image):
        import torch
        small = resize_limit(image, 1280)
        ratio = self.processor.get_ratio(np.array(small))
        if abs(ratio - 1.) > 1e-6:
            raise RuntimeError(f'CRAFT ratio باید یک باشد، مقدار فعلی: {ratio}')
        with torch.inference_mode():
            result = self.detector.predict(small)[0]
        boxes = []
        for x, y, w, h in result['boxes']:
            sx, sy = image.width/small.width, image.height/small.height
            x1, y1 = max(0, int(x*sx)), max(0, int(y*sy))
            x2, y2 = min(image.width, int(np.ceil((x+w)*sx))), min(image.height, int(np.ceil((y+h)*sy)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return boxes

    @staticmethod
    def merge_fragments(boxes):
        # Join close fragments, but do not combine distant label/value columns.
        boxes = [list(b) for b in boxes]
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                a = boxes[i]
                for j in range(i+1, len(boxes)):
                    b = boxes[j]
                    ah, bh = a[3]-a[1], b[3]-b[1]
                    overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
                    gap = max(0, max(a[0], b[0])-min(a[2], b[2]))
                    if overlap/min(ah, bh) >= .55 and gap <= .65*max(ah, bh):
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        boxes.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return boxes

    def read_regions(self, image):
        import torch
        rows = []
        for box in self.merge_fragments(self.text_boxes(image)):
            x1, y1, x2, y2 = box
            h = y2-y1
            px, py = max(8, round(.20*h)), max(8, round(.30*h))
            crop_box = [max(0, x1-px), max(0, y1-py), min(image.width, x2+px), min(image.height, y2+py)]
            crop = image.crop(crop_box)
            with torch.inference_mode():
                output = self.reader.predict(crop)
            raw = str(output[0]['text']) if output else ''
            rows.append({'box': box, 'crop_box': crop_box, 'text_raw': raw,
                         'text': normalize_text(raw)})
        return rows

    @staticmethod
    def orientation_score(regions):
        # This is an OCR heuristic, not calibrated model confidence.
        texts = [r['text'] for r in regions]
        anchors = ['شماره ملی', 'نام', 'نام خانوادگی', 'تاریخ تولد', 'نام پدر', 'پایان اعتبار']
        compact = [re.sub(r'\s+', '', t) for t in texts if t]
        anchor_score = sum(max([SequenceMatcher(None, a.replace(' ', ''), t).ratio()
                                for t in compact] or [0]) for a in anchors)
        joined = ' '.join(texts)
        date_count = min(2, len(re.findall(r'(?<!\d)1[234]\d{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])(?!\d)', joined)))
        id_count = min(1, len(re.findall(r'(?<!\d)\d{10}(?!\d)', joined)))
        return float(anchor_score + 2*date_count + 2*id_count)

    @staticmethod
    def reading_order(regions):
        lines = []
        for item in sorted(regions, key=lambda r: (r['box'][1]+r['box'][3])/2):
            x1, y1, x2, y2 = item['box']
            cy, height = (y1+y2)/2, y2-y1
            target = next((line for line in lines if abs(line['cy']-cy) < .5*min(line['height'], height)), None)
            if target is None:
                target = {'cy': cy, 'height': height, 'items': []}
                lines.append(target)
            target['items'].append(item)
        return '\n'.join(' '.join(r['text_raw'] for r in sorted(line['items'], key=lambda r: -r['box'][0]))
                         for line in lines)

    def process(self, source, output_dir):
        started = time.perf_counter()
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        result = {'source': str(source) if not isinstance(source, Image.Image) else 'PIL.Image',
                  'status': 'started', 'text': '', 'regions': [], 'variant': self.variant,
                  'model_sha256': self.model_sha256, 'review_required': True}
        try:
            image = read_rgb(source)
            points, peaks = self.locate(image)
            result['corner_heatmap_peaks'] = peaks
            status = validate_quad(points, image.size) if points is not None else 'corners_not_found'
            result['geometry_status'] = status
            if points is not None:
                result['corners'] = order_quad(points).tolist()
                marked = np.array(image).copy()
                cv2.polylines(marked, [np.int32(order_quad(points))], True, (0, 255, 0), max(2, image.width//500))
                resize_limit(Image.fromarray(marked), 1400).save(folder/'boundary.jpg')
            if status != 'geometry_candidate':
                result['status'] = status
            else:
                card = warp_card(image, points)
                card.save(folder/'rectified.png')
                trials = []
                # A rectangular warp has two landscape orientations; no manual rotation.
                angles = [0, 180] if card.width >= card.height else [90, 270]
                for angle in angles:
                    print(f'  OCR orientation: {angle}', flush=True)
                    rotated = card.rotate(angle, expand=True)
                    regions = self.read_regions(rotated)
                    trials.append({'angle': angle, 'score': self.orientation_score(regions),
                                   'regions': regions, 'image': rotated})
                trials.sort(key=lambda t: t['score'], reverse=True)
                best, other = trials
                result['orientation_trials'] = [{'angle': t['angle'], 'score': t['score']} for t in trials]
                result['angle_ccw'] = best['angle']
                result['regions'] = best['regions']
                result['text'] = self.reading_order(best['regions'])
                # Thresholds are development defaults, not measured accuracy guarantees.
                certain = best['score'] >= 4 and best['score']-other['score'] >= 1
                result['status'] = ('ocr_candidate' if certain else 'orientation_uncertain') if best['regions'] else 'no_text'
                best['image'].save(folder/'oriented.png')
                marked = np.array(best['image']).copy()
                for i, item in enumerate(best['regions']):
                    x1, y1, x2, y2 = item['box']
                    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(marked, str(i), (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 0), 1)
                    best['image'].crop(item['crop_box']).save(folder/f'crop_{i:02d}.png')
                Image.fromarray(marked).save(folder/'text_boxes.png')
        except Exception as exc:
            result['status'] = 'error'
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        (folder/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        (folder/'text.txt').write_text(result['text'], encoding='utf-8')
        return result
