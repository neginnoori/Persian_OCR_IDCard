"""PaddleOCR detection only. Recognition is deliberately not constructed here."""
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image

MODEL_NAME = 'PP-OCRv5_server_det'
MODEL_REVISION = 'ca867c897ecbca8873081573a802ad70d499cb94'


class PaddleTextDetector:
    def __init__(self, cache_dir, device='gpu:0', *, model_dir=None):
        if device not in ('cpu', 'gpu:0'):
            raise ValueError('Paddle detector device must be cpu or gpu:0')
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.model_dir = model_dir
        self.model = None

    def load(self):
        if self.model is not None:
            return self
        import paddle
        from paddleocr import TextDetection
        if self.device == 'gpu:0':
            if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
                raise RuntimeError('Paddle GPU requested, but CUDA Paddle/GPU is unavailable; no CPU fallback')
            paddle.set_device(self.device)
            (paddle.ones([1]) + 1).numpy()  # real device operation, not availability alone
        if self.model_dir is None:
            from huggingface_hub import snapshot_download
            directory = snapshot_download('PaddlePaddle/' + MODEL_NAME,
                                          revision=MODEL_REVISION, cache_dir=str(self.cache_dir/'huggingface'))
        else:
            directory = str(self.model_dir)
        self.model = TextDetection(model_name=MODEL_NAME, model_dir=directory,
                                   device=self.device, use_tensorrt=False,
                                   enable_mkldnn=False, cpu_threads=4,
                                   limit_side_len=1280, limit_type='max')
        return self

    def detect(self, card):
        if not isinstance(card, Image.Image):
            raise TypeError('Detector expects an upright PIL image')
        self.load()
        # Paddle's ndarray input is OpenCV BGR; returned polygons are input pixels.
        bgr = cv2.cvtColor(np.asarray(card.convert('RGB')), cv2.COLOR_RGB2BGR)
        results = list(self.model.predict(bgr, batch_size=1))
        if len(results) != 1:
            raise RuntimeError('Paddle detector must return exactly one result')
        result = results[0]
        if 'dt_polys' not in result and 'res' in result:
            result = result['res']
        if 'dt_polys' not in result or 'dt_scores' not in result:
            raise RuntimeError('Paddle output is missing dt_polys/dt_scores')
        polygons, scores = result['dt_polys'], result['dt_scores']
        if len(polygons) != len(scores):
            raise RuntimeError('Paddle polygon/score counts differ')
        regions = []
        for polygon, score in zip(polygons, scores):
            p = np.asarray(polygon, dtype=np.float32)
            if p.shape != (4, 2) or not np.isfinite(p).all() or not math.isfinite(float(score)):
                raise RuntimeError('Paddle returned invalid polygon/score')
            p[:, 0] = np.clip(p[:, 0], 0, card.width-1)
            p[:, 1] = np.clip(p[:, 1], 0, card.height-1)
            if not cv2.isContourConvex(p) or abs(cv2.contourArea(p)) < 4:
                continue
            x1, y1 = np.floor(p.min(axis=0)).astype(int)
            x2, y2 = np.ceil(p.max(axis=0)).astype(int) + 1
            regions.append({'box': [int(x1),int(y1),min(int(x2),card.width),min(int(y2),card.height)],
                            'polygon': p.tolist(), 'detection_score': float(score)})
        return regions


def crop_polygon(card, polygon):
    """Rectify one text line, then add background padding for Hezar."""
    p = np.float32(polygon)
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:,1]-center[1], p[:,0]-center[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    w = max(2, round(max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))))
    h = max(2, round(max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))))
    target = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    pixels = cv2.warpPerspective(np.asarray(card), cv2.getPerspectiveTransform(p,target), (w,h),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Constant edge-median padding avoids stretching glyphs outside a tight crop.
    color = tuple(int(v) for v in np.median(np.concatenate([pixels[0],pixels[-1]]),axis=0))
    px, py = max(8,round(.20*h)), max(8,round(.30*h))
    return Image.fromarray(cv2.copyMakeBorder(pixels,py,py,px,px,cv2.BORDER_CONSTANT,value=color))
