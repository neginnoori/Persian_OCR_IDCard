"""Read an already upright RGB card with Hezar CRAFT + Persian CRNN."""
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
    def __init__(self, cache_dir, device='cuda', batch_size=8):
        if device not in ('cuda', 'cuda:0', 'cpu'):
            raise ValueError('device must be cuda, cuda:0, or cpu')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.cache_dir, self.device = Path(cache_dir), device
        self.batch_size = batch_size
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
            detector = Model.load('hezarai/CRAFT', device=self.device, cache_dir=str(self.cache_dir))
            recognizer = Model.load('hezarai/crnn-base-fa-v2', device=self.device, cache_dir=str(self.cache_dir))
            detector.eval()
            recognizer.eval()
            # Hezar 1.0.0 coordinate adjustment is reliable here when ratio == 1.
            detector.preprocessor.image_processor.config.square_size = 2560
            detector.preprocessor.image_processor.config.mag_ratio = 1.0
            self.detector, self.recognizer = detector, recognizer
        return self

    def read(self, card):
        if not isinstance(card, Image.Image):
            raise TypeError('read() expects an upright PIL card image')
        card = card.convert('RGB')
        with self._lock:
            self.load()
            import torch
            scale = min(1., 1280 / max(card.size))
            small = card.resize((max(1, round(card.width * scale)), max(1, round(card.height * scale))), Image.Resampling.LANCZOS)
            ratio = self.detector.preprocessor.image_processor.get_ratio(np.asarray(small))
            if abs(ratio - 1.) > 1e-6:
                raise RuntimeError(f'Unexpected CRAFT resize ratio: {ratio}')
            with torch.inference_mode():
                detections = self.detector.predict(small)
            if len(detections) != 1:
                raise RuntimeError('CRAFT did not return one detection result')
            regions, crops, boxes = [], [], []
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
            for group in merge_fragments(boxes):
                x1, y1, x2, y2 = group['box']
                height = y2 - y1
                px, py = max(8, round(.20 * height)), max(8, round(.30 * height))
                crop_box = [max(0, x1-px), max(0, y1-py), min(card.width, x2+px), min(card.height, y2+py)]
                regions.append({'box': [x1, y1, x2, y2], 'crop_box': crop_box,
                                'source_boxes': group['source_boxes']})
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

