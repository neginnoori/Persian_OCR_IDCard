from dataclasses import dataclass, field
from copy import deepcopy
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
                 extractor=None, reader=None, detector_backend='paddle', detector_device=None,
                 preprocessing='rectify_best_effort', candidate_selection='legacy',
                 bsrgan_model_path=None, bsrgan_source_dir=None):
        if preprocessing not in ('orientation_only', 'rectify', 'rectify_best_effort'):
            raise ValueError('preprocessing must be orientation_only, rectify or rectify_best_effort')
        if candidate_selection not in ('legacy', 'v2', 'v3'):
            raise ValueError('candidate_selection must be legacy, v2 or v3')
        self.preprocessing = preprocessing
        self.candidate_selection = candidate_selection
        cache = Path(cache_dir)
        self.extractor = extractor if extractor is not None else CardExtractor(ExtractorConfig(cache_dir=cache/'extraction', candidate_selection=candidate_selection))
        self.reader = reader if reader is not None else HezarReader(
            cache/'hezar', device=hezar_device, batch_size=batch_size,
            detector_backend=detector_backend, detector_device=detector_device,
            bsrgan_model_path=bsrgan_model_path, bsrgan_source_dir=bsrgan_source_dir)
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
            if getattr(self.reader, 'enhancement_audit', None) is not None:
                metadata['text_detection_enhancement'] = self.reader.enhancement_audit
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
            adaptive_regions = regions
            enhancement = getattr(self.reader, 'enhancement_audit', None)
            if (enhancement and enhancement.get('enhanced_recognition_status') == 'applied'
                    and any('text_bsrgan_raw' in region for region in regions)):
                from .adaptive import compatible
                bsrgan_regions = deepcopy(regions)
                for region in bsrgan_regions:
                    if 'text_bsrgan_raw' in region:
                        region['text_raw'] = region['text_bsrgan_raw']
                        region['text'] = region['text_bsrgan']
                if anchor_mode:
                    from .anchors import assign_anchored_fields
                    bsrgan_fields, bsrgan_assignment_audit = assign_anchored_fields(
                        bsrgan_regions, extraction.image.size)
                else:
                    bsrgan_fields = assign_fields(field_regions and bsrgan_regions or [],
                                                  extraction.image.size, frame=frame)
                    bsrgan_assignment_audit = None
                baseline_fields = deepcopy(fields)
                selected_fields, decisions = [], []
                enhanced_source_region_ids = set()
                for original, enhanced in zip(baseline_fields, bsrgan_fields):
                    enhanced_valid = (enhanced.get('status') == 'candidate' and
                                      compatible(enhanced['field'], enhanced.get('value')))
                    if enhanced_valid:
                        chosen = deepcopy(enhanced)
                        chosen['assignment_method'] += '+bsrgan_x2_ocr'
                        source = 'bsrgan_x2'
                        enhanced_source_region_ids.update(enhanced.get('source_region_ids', []))
                    else:
                        chosen = deepcopy(original)
                        source = 'original_card'
                    chosen['ocr_source'] = source
                    selected_fields.append(chosen)
                    decisions.append({'field': original['field'], 'selected_source': source,
                                      'original_value': original.get('value'),
                                      'bsrgan_value': enhanced.get('value'),
                                      'bsrgan_type_valid': bool(enhanced_valid),
                                      'selected_value': chosen.get('value')})
                adaptive_regions = deepcopy(regions)
                for index in enhanced_source_region_ids:
                    if 0 <= index < len(adaptive_regions) and 'text_bsrgan_raw' in adaptive_regions[index]:
                        adaptive_regions[index]['text_raw'] = adaptive_regions[index]['text_bsrgan_raw']
                        adaptive_regions[index]['text'] = adaptive_regions[index]['text_bsrgan']
                        adaptive_regions[index]['ocr_source'] = 'bsrgan_x2'
                fields = selected_fields
                metadata['bsrgan_ocr_comparison'] = {
                    'selection_policy': 'use_bsrgan_when_field_type_is_valid_else_preserve_original',
                    'original_fields': baseline_fields,
                    'bsrgan_fields': bsrgan_fields,
                    'selected_fields': deepcopy(selected_fields),
                    'decisions': decisions,
                    'field_assignment_audit': bsrgan_assignment_audit,
                }
            if self.candidate_selection == 'v3':
                from .adaptive import repair_fields
                fields, repair_audit = repair_fields(extraction.image, adaptive_regions, fields,
                    getattr(self.reader, 'recognize_crops', None))
                metadata['adaptive_fields'] = repair_audit
                if any(d['replaced'] for d in repair_audit['decisions']):
                    metadata['field_assignment'] = 'adaptive_label_anchors_v1'
            if 'bsrgan_ocr_comparison' in metadata:
                enhanced_ids = enhanced_source_region_ids
                for item in fields:
                    item_ids = set(item.get('source_region_ids', []))
                    item['ocr_source'] = ('bsrgan_x2' if item_ids.intersection(enhanced_ids)
                                          else item.get('ocr_source', 'original_card'))
                metadata['bsrgan_ocr_comparison']['final_fields'] = deepcopy(fields)
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

