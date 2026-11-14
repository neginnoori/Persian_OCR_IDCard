"""Detect text with PaddleOCR; read crops with Hezar CRNN."""
from pathlib import Path
from threading import RLock
import math
import re
import numpy as np
from PIL import Image


def normalize_text(text):
    digits = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
    return re.sub(r'\s+', ' ', str(text).translate(digits).replace('ي', 'ی').replace('ك', 'ک')).strip()


def reading_order(regions):
    """Group nearby vertical centers, then read each line right to left."""
    lines = []
    for item in sorted(regions, key=lambda r: (r['box'][1] + r['box'][3]) / 2):
        x1, y1, x2, y2 = item['box']
        center, height = (y1 + y2) / 2, y2 - y1
        line = next((row for row in lines if abs(row['center'] - center) < .5 * min(row['height'], height)), None)
        if line is None:
            line = {'center': center, 'height': height, 'items': []}
            lines.append(line)
        line['items'].append(item)
    return [item for row in lines for item in sorted(row['items'], key=lambda r: -r['box'][0])]


def merge_fragments(boxes):
    """Join close, aligned fragments before recognition; preserve distant columns."""
    groups = [{'box': list(box), 'source_boxes': [list(box)]} for box in boxes]
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(groups):
            a = first['box']
            for j in range(i+1, len(groups)):
                b = groups[j]['box']
                ah, bh = a[3]-a[1], b[3]-b[1]
                overlap = max(0, min(a[3], b[3])-max(a[1], b[1]))
                gap = max(0, max(a[0], b[0])-min(a[2], b[2]))
                if overlap/min(ah, bh) >= .65 and gap <= .65*min(ah, bh):
                    first['box'] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    first['source_boxes'].extend(groups[j]['source_boxes'])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


class HezarReader:
    """Reuse one instance; CUDA for Colab, or CPU for reading pre-extracted cards.

    Headless OpenCV is installed explicitly rather than hezar[vision], which would
    install a second conflicting cv2 distribution alongside EasyOCR's dependency.
    """
    def __init__(self, cache_dir, device='cuda', batch_size=8, *, detector_backend='paddle', detector_device=None,
                 bsrgan_model_path=None, bsrgan_source_dir=None):
        if device not in ('cuda', 'cuda:0', 'cpu'):
            raise ValueError('device must be cuda, cuda:0, or cpu')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if detector_backend not in ('paddle', 'craft'):
            raise ValueError('detector_backend must be paddle or craft')
        self.detector_backend = detector_backend
        self.detector_device = detector_device
        self.cache_dir, self.device = Path(cache_dir), device
        self.batch_size = batch_size
        if (bsrgan_model_path is None) != (bsrgan_source_dir is None):
            raise ValueError('BSRGAN needs both a weights file and source directory')
        if bsrgan_model_path is not None and detector_backend != 'paddle':
            raise ValueError('The BSRGAN text-detection experiment currently requires PaddleOCR')
        self.bsrgan_model_path = bsrgan_model_path
        self.bsrgan_source_dir = bsrgan_source_dir
        self.enhancer = None
        self.enhancement_load_error = None
        self.enhancement_audit = None
        self.enhanced_preview = None
        self.detector = self.recognizer = None
        self._lock = RLock()

    def load(self):
        with self._lock:
            if self.detector is not None and self.recognizer is not None:
                return self
            import torch
            from hezar.models import Model
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('Hezar CUDA requested but no CUDA GPU is available')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.detector_backend == 'paddle':
                from .paddle_client import PaddleWorkerDetector
                detector = PaddleWorkerDetector(self.cache_dir/'paddle', device=self.detector_device).load()
            else:
                detector = Model.load('hezarai/CRAFT', device=self.device, cache_dir=str(self.cache_dir))
                detector.eval()
                detector.preprocessor.image_processor.config.square_size = 2560
                detector.preprocessor.image_processor.config.mag_ratio = 1.0
            try:
                recognizer = Model.load('hezarai/crnn-base-fa-v2', device=self.device, cache_dir=str(self.cache_dir))
                recognizer.eval()
                enhancer = None
                if self.bsrgan_model_path is not None:
                    from .bsrgan import BSRGANX2
                    try:
                        enhancer = BSRGANX2(self.bsrgan_model_path, self.bsrgan_source_dir, device=self.device)
                    except Exception as error:
                        self.enhancement_load_error = type(error).__name__
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except BaseException:
                if hasattr(detector, 'close'):
                    detector.close()
                raise
            self.detector, self.recognizer, self.enhancer = detector, recognizer, enhancer
        return self

    def read(self, card):
        if not isinstance(card, Image.Image):
            raise TypeError('read() expects an upright PIL card image')
        card = card.convert('RGB')
        with self._lock:
            self.enhanced_preview = None
            self.load()
            import torch
            regions, crops, boxes = [], [], []
            detected = []
            if self.detector_backend == 'paddle':
                detected = self.detector.detect(card)
                original_count = len(detected)
                enhanced_count = 0
                added_count = 0
                enhancement_status = 'disabled'
                if self.enhancement_load_error:
                    enhancement_status = 'load_failed'
                elif self.enhancer is not None:
                    enhancement_status = 'failed'
                    try:
                        enhanced_image = self.enhancer.enhance(card)
                        self.enhanced_preview = enhanced_image
                        enhanced_detections = self.detector.detect(enhanced_image)
                        enhanced_count = len(enhanced_detections)
                        scale_x = card.width / enhanced_image.width
                        scale_y = card.height / enhanced_image.height
                        for item in enhanced_detections:
                            candidate = dict(item)
                            candidate['box'] = [
                                max(0, int(math.floor(item['box'][0] * scale_x))),
                                max(0, int(math.floor(item['box'][1] * scale_y))),
                                min(card.width, int(math.ceil(item['box'][2] * scale_x))),
                                min(card.height, int(math.ceil(item['box'][3] * scale_y))),
                            ]
                            candidate['polygon'] = [
                                [float(point[0] * scale_x), float(point[1] * scale_y)]
                                for point in item['polygon']
                            ]
                            candidate['detection_source'] = 'bsrgan_x2'
                            if not any(_box_iou(candidate['box'], original['box']) >= .50
                                       for original in detected):
                                detected.append(candidate)
                                added_count += 1
                        enhancement_status = 'applied'
                    except Exception as error:
                        self.enhancement_load_error = type(error).__name__
                        self.enhancer = None
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                for item in detected:
                    item.setdefault('detection_source', 'original')
                self.enhancement_audit = {
                    'method': 'bsrgan_x2' if self.bsrgan_model_path is not None else None,
                    'status': enhancement_status,
                    'error_type': self.enhancement_load_error,
                    'original_detection_count': original_count,
                    'enhanced_detection_count': enhanced_count,
                    'added_box_count': added_count,
                    'recognition_crops_from': 'original_card',
                }
                boxes = [item['box'] for item in detected]
            else:
                scale = min(1., 1280 / max(card.size))
                small = card.resize((max(1, round(card.width * scale)), max(1, round(card.height * scale))), Image.Resampling.LANCZOS)
                ratio = self.detector.preprocessor.image_processor.get_ratio(np.asarray(small))
                if abs(ratio - 1.) > 1e-6:
                    raise RuntimeError(f'Unexpected CRAFT resize ratio: {ratio}')
                with torch.inference_mode():
                    detections = self.detector.predict(small)
                if len(detections) != 1:
                    raise RuntimeError('CRAFT did not return one detection result')
                sx, sy = card.width / small.width, card.height / small.height
                for box in detections[0]['boxes']:
                    if len(box) != 4 or not np.isfinite(box).all():
                        raise RuntimeError('CRAFT returned an invalid box')
                    x, y, w, h = map(float, box)
                    if w <= 0 or h <= 0:
                        continue
                    x1, y1 = max(0, math.floor(x * sx)), max(0, math.floor(y * sy))
                    x2, y2 = min(card.width, math.ceil((x + w) * sx)), min(card.height, math.ceil((y + h) * sy))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
            groups = ([{'box': d['box'], 'source_boxes': [d['box']]} for d in detected]
                      if self.detector_backend == 'paddle' else merge_fragments(boxes))
            for group in groups:
                x1, y1, x2, y2 = group['box']
                height = y2 - y1
                px, py = max(8, round(.20 * height)), max(8, round(.30 * height))
                crop_box = [max(0, x1-px), max(0, y1-py), min(card.width, x2+px), min(card.height, y2+py)]
                regions.append({'box': [x1, y1, x2, y2], 'crop_box': crop_box,
                                'source_boxes': group['source_boxes']})
                if self.detector_backend == 'paddle':
                    from .detection import crop_polygon
                    sources = [d for d in detected if d['box'] in group['source_boxes']]
                    regions[-1].update(source_polygons=[d['polygon'] for d in sources],
                                       detection_scores=[d['detection_score'] for d in sources],
                                       detection_sources=[d['detection_source'] for d in sources])
                    if len(sources) == 1:
                        regions[-1]['recognition_crop_method'] = 'perspective_polygon_with_padding'
                        crops.append(crop_polygon(card, sources[0]['polygon']))
                    else:
                        regions[-1]['recognition_crop_method'] = 'merged_axis_aligned_with_margin'
                        crops.append(card.crop(crop_box))
                else:
                    regions[-1]['recognition_crop_method'] = 'axis_aligned_with_margin'
                    crops.append(card.crop(crop_box))
            for start in range(0, len(crops), self.batch_size):
                batch = crops[start:start+self.batch_size]
                with torch.inference_mode():
                    predictions = self.recognizer.predict(batch)
                if len(predictions) != len(batch):
                    raise RuntimeError('Hezar recognition result count differs from crop count')
                for region, prediction in zip(regions[start:start+self.batch_size], predictions):
                    raw = str(prediction['text'])
                    region.update(text_raw=raw, text=normalize_text(raw))
            return reading_order(regions)

    def recognize_crops(self, crops):
        """Read supplied value crops without running detection again."""
        with self._lock:
            self.load()
            import torch
            with torch.inference_mode():
                predictions = self.recognizer.predict(crops)
            if len(predictions) != len(crops):
                raise RuntimeError('Hezar recognition count differs from crop count')
            return [str(p['text']) for p in predictions]

    def close(self):
        with self._lock:
            if self.detector is not None and hasattr(self.detector, 'close'):
                self.detector.close()
            if self.enhancer is not None:
                del self.enhancer
                self.enhancer = None


def _box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0

