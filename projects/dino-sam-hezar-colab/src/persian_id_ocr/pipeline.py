from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any
from PIL import Image
from cardid import CardExtractor, ExtractorConfig
from .reader import HezarReader
from .fields import assign_fields


@dataclass
class OCRResult:
    status: str
    image: Image.Image | None = None
    text: str = ''
    text_normalized: str = ''
    regions: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: list[dict] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.fields)

    def to_dict(self):
        return {'status': self.status, 'text': self.text, 'text_normalized': self.text_normalized,
                'regions': self.regions, 'fields': self.fields,
                'metadata': self.metadata, 'review_required': True}


class PersianCardOCR:
    """DINO/SAM extraction, then Hezar OCR exactly once on an accepted upright card.

    EasyOCR is used only by the extractor for orientation. Final text comes from
    Hezar. Runtime failures raise; uncertain extraction never silently reaches OCR.
    No files are written by this API. Calls on one instance are serialized.
    """
    def __init__(self, cache_dir='./models', *, hezar_device='cuda', batch_size=8,
                 extractor=None, reader=None, detector_backend='paddle', detector_device=None):
        cache = Path(cache_dir)
        self.extractor = extractor if extractor is not None else CardExtractor(ExtractorConfig(cache_dir=cache/'extraction'))
        self.reader = reader if reader is not None else HezarReader(cache/'hezar', device=hezar_device, batch_size=batch_size, detector_backend=detector_backend, detector_device=detector_device)
        self._lock = RLock()

    def process(self, image, *, color_order='RGB', orientation_override=None):
        with self._lock:
            started = perf_counter()
            extraction = self.extractor.extract_result(image, color_order=color_order,
                                                       orientation_override=orientation_override)
            extracted_at = perf_counter()
            metadata = {'extraction': extraction.to_dict(), 'coordinate_space': 'extracted_card',
                        'extraction_seconds': extracted_at-started,
                        'text_detector': ('PP-OCRv5_server_det' if getattr(self.reader,'detector_backend','paddle') == 'paddle' else 'hezarai/CRAFT'), 'text_recognizer': 'hezarai/crnn-base-fa-v2'}
            if not extraction.ok:
                metadata.update(ocr_seconds=0., total_seconds=perf_counter()-started)
                return OCRResult(extraction.status, image=extraction.image, metadata=metadata,
                                 fields=assign_fields([], (856,540)))
            if extraction.image is None:
                raise RuntimeError('Extractor accepted a card but returned no image')
            regions = self.reader.read(extraction.image)
            metadata['text_detector_device'] = getattr(getattr(self.reader, 'detector', None), 'device', None)
            metadata.update(ocr_seconds=perf_counter()-extracted_at, total_seconds=perf_counter()-started)
            has_text = any(r['text_raw'].strip() for r in regions)
            return OCRResult('ocr_candidate' if has_text else 'no_text', image=extraction.image,
                             text='\n'.join(r['text_raw'] for r in regions),
                             text_normalized='\n'.join(r['text'] for r in regions), regions=regions,
                             metadata=metadata, fields=assign_fields(regions, extraction.image.size))

    def close(self):
        with self._lock:
            if hasattr(self.reader, 'close'):
                self.reader.close()

