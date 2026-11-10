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
    """Rectify the card, detect text with Paddle and read it with Hezar.

    Default rectify_best_effort restores perspective correction without blocking OCR.
    Opt into rectify mode for the previous strict perspective/geometry gate.
    """
    def __init__(self, cache_dir='./models', *, hezar_device='cuda', batch_size=8,
                 extractor=None, reader=None, detector_backend='paddle', detector_device=None, preprocessing='rectify_best_effort', candidate_selection='legacy'):
        if preprocessing not in ('orientation_only', 'rectify', 'rectify_best_effort'):
            raise ValueError('preprocessing must be orientation_only, rectify or rectify_best_effort')
        if candidate_selection not in ('legacy', 'v2'):
            raise ValueError('candidate_selection must be legacy or v2')
        self.preprocessing = preprocessing
        cache = Path(cache_dir)
        self.extractor = extractor if extractor is not None else CardExtractor(ExtractorConfig(cache_dir=cache/'extraction', candidate_selection=candidate_selection))
        self.reader = reader if reader is not None else HezarReader(cache/'hezar', device=hezar_device, batch_size=batch_size, detector_backend=detector_backend, detector_device=detector_device)
        self._lock = RLock()

    def process(self, image, *, color_order='RGB', orientation_override=None):
        with self._lock:
            started = perf_counter()
            prepare = (self.extractor.orient_result if self.preprocessing == 'orientation_only'
                       else self.extractor.extract_result)
            extraction = prepare(image, color_order=color_order,
                                                       orientation_override=orientation_override)
            effective_mode = 'orientation_only' if self.preprocessing == 'orientation_only' else 'rectify'
            rectification_attempt = None
            warnings = list(extraction.metadata.get('warnings', []))
            if self.preprocessing == 'rectify_best_effort':
                if extraction.reason:
                    warnings.extend(extraction.reason.split(','))
                if extraction.image is None:
                    # Preserve the failed attempt; never fabricate a card quadrilateral.
                    rectification_attempt = extraction.to_dict()
                    extraction = self.extractor.orient_result(
                        image, color_order=color_order, orientation_override=orientation_override)
                    effective_mode = 'orientation_only'
                    warnings.append('rectification_unavailable_orientation_only_fallback')
                    warnings.extend(extraction.metadata.get('warnings', []))
            if extraction.metadata.get('field_assignment_mode') == 'anchors_v1':
                effective_mode = 'orientation_only'
            extracted_at = perf_counter()
            metadata = {'extraction': extraction.to_dict(), 'coordinate_space': ('oriented_input' if effective_mode == 'orientation_only' else 'extracted_card'),
                        'preprocessing_mode': self.preprocessing,
                        'effective_preprocessing': effective_mode,
                        'warnings': list(dict.fromkeys(warnings)),
                        'extraction_seconds': extracted_at-started,
                        'text_detector': ('PP-OCRv5_server_det' if getattr(self.reader,'detector_backend','paddle') == 'paddle' else 'hezarai/CRAFT'), 'text_recognizer': 'hezarai/crnn-base-fa-v2'}
            if rectification_attempt is not None:
                metadata['rectification_attempt'] = rectification_attempt
            if not extraction.ok and (self.preprocessing != 'rectify_best_effort' or extraction.image is None):
                metadata.update(ocr_seconds=0., total_seconds=perf_counter()-started)
                return OCRResult(extraction.status, image=extraction.image, metadata=metadata,
                                 fields=assign_fields([], (856,540)))
            if extraction.image is None:
                raise RuntimeError('Extractor accepted a card but returned no image')
            regions = self.reader.read(extraction.image)
            metadata['text_detector_device'] = getattr(getattr(self.reader, 'detector', None), 'device', None)
            metadata.update(ocr_seconds=perf_counter()-extracted_at, total_seconds=perf_counter()-started)
            has_text = any(r['text_raw'].strip() for r in regions)
            frame = extraction.metadata.get('card_frame') if effective_mode == 'orientation_only' else None
            field_regions = regions
            if effective_mode == 'orientation_only' and frame is None:
                # OCR still runs on the whole image; no card layout was located.
                field_regions = []
            metadata['field_assignment'] = 'approximate_card_frame' if frame else ('unavailable' if effective_mode == 'orientation_only' else 'rectified_template')
            anchor_mode = extraction.metadata.get('field_assignment_mode') == 'anchors_v1'
            if anchor_mode:
                from .anchors import assign_anchored_fields
                fields, anchor_audit = assign_anchored_fields(regions, extraction.image.size)
                metadata['field_assignment'] = 'label_anchors_v1'
                metadata['anchor_assignment'] = anchor_audit
            else:
                fields = assign_fields(field_regions, extraction.image.size, frame=frame)
            if hasattr(self.reader, 'recognize_crops') and field_regions and not anchor_mode:
                from .numeric_retry import retry_numeric_fields
                fields = retry_numeric_fields(extraction.image, fields,
                                              self.reader.recognize_crops, frame=frame)
            metadata.update(ocr_seconds=perf_counter()-extracted_at,
                            total_seconds=perf_counter()-started)
            return OCRResult('ocr_candidate' if has_text else 'no_text', image=extraction.image,
                             text='\n'.join(r['text_raw'] for r in regions),
                             text_normalized='\n'.join(r['text'] for r in regions), regions=regions,
                             metadata=metadata, fields=fields)

    def close(self):
        with self._lock:
            if hasattr(self.reader, 'close'):
                self.reader.close()

