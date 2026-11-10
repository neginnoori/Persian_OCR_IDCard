"""Selection-to-six-field regressions with synthetic observations, no model weights.

The real CardExtractor geometry, recovery policy and assign_fields run here.
Injected OCR observations do not measure recognition accuracy.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

from cardid import CardExtractor
from cardid.backend import Detection
from cardid.geometry import quad_from_mask, warp_candidates
from persian_id_ocr import PersianCardOCR
from persian_id_ocr.fields import FIELDS, CENTERS
from persian_id_ocr.reader import normalize_text


VALUES = ['۱۲۳۴۵۶۷۸۹۰', 'نمونه', 'آزمایشی', '۱۳۷۰/۰۲/۱۲', 'نمونه', '۱۴۰۸/۰۳/۲۱']


def ocr_observation(text, y, confidence=.95):
    return ([[470,y-10], [665,y-10], [665,y+10], [470,y+10]], text, confidence)


class FieldsReader:
    def __init__(self):
        self.calls = 0

    def read(self, image):
        self.calls += 1
        assert image.size == (856,540)
        return [dict(box=[470, round(c*540)-10, 665, round(c*540)+10],
                     text_raw=text, text=normalize_text(text))
                for c,text in zip(CENTERS,VALUES)]


def scene():
    rgb = np.full((600,1000,3), 235, np.uint8)
    rgb[200:380,300:600] = [40,80,190]
    mask = np.zeros((600,1000), np.uint8)
    mask[200:380,300:600] = 1
    return Image.fromarray(rgb), mask


def install_backend(monkeypatch, mask, kind):
    import cardid.extractor as implementation

    class OrientationReader:
        def __init__(self): self.calls = 0
        def readtext(self, *args, **kwargs):
            self.calls += 1
            if kind == 'empty': return []
            if kind == 'weak': return [ocr_observation('آزمایش', 160, .4)]
            if self.calls <= 4 or kind == 'unsupported_stamp':
                return [ocr_observation('کپی برابر اصل است', 160)]
            return [ocr_observation(text, round(c*540)) for c,text in zip(CENTERS,VALUES)]

    class Faces:
        def __init__(self, reader): self.reader = reader
        def empty(self): return False
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[80,150,120,160]]) if self.reader.calls > 4 and kind == 'supported_rescue' else []

    class Backend:
        info = {'device':'injected-test-backend'}
        def __init__(self, config):
            self.reader = OrientationReader()
            self.face_detector = Faces(self.reader)
        def detect_masks(self, image):
            return Detection([mask], [.99], .9)
        def detect_alternative_masks(self, image):
            return Detection([], [], None)

    monkeypatch.setattr(implementation, 'TorchBackend', Backend)


def assert_six_fields(result):
    assert result.status == 'ocr_candidate'
    assert result.metadata['effective_preprocessing'] == 'rectify'
    assert [field['field'] for field in result.fields] == [key for key,_ in FIELDS]
    assert [field['value'] for field in result.fields] == [normalize_text(v) for v in VALUES]
    assert all(field['status'] == 'candidate' and field['box'] for field in result.fields)


@pytest.mark.parametrize('kind', ['empty','weak'])
def test_weak_orientation_evidence_keeps_all_six_fields(monkeypatch, kind):
    image,mask = scene()
    install_backend(monkeypatch,mask,kind)
    reader = FieldsReader()
    result = PersianCardOCR(extractor=CardExtractor(), reader=reader,
                            preprocessing='rectify_best_effort').process(image)
    assert_six_fields(result)
    assert reader.calls == 1
    assert 'ambiguous_orientation' in result.metadata['warnings']
    assert not result.metadata['extraction']['metadata']['stamp_recovery']['attempted']


def test_supported_rescue_keeps_six_fields_and_has_no_stale_stamp_iou(monkeypatch):
    image,mask = scene()
    install_backend(monkeypatch,mask,'supported_rescue')
    result = PersianCardOCR(extractor=CardExtractor(), reader=FieldsReader(),
                            preprocessing='rectify_best_effort').process(image)
    assert_six_fields(result)
    metadata = result.metadata['extraction']['metadata']
    assert metadata['stamp_recovery']['replaced']
    assert metadata['quad_method'] == 'full_image_frame_recovery'
    assert not metadata['geometry_ok']
    for key in ('mask_quad_iou','mask_coverage','border_status','geometry_candidate_diagnostics'):
        assert key not in metadata
    assert metadata['sam_predicted_iou'] is None
    assert 'unreliable_geometry' in result.metadata['warnings']


def test_unsupported_rescue_preserves_baseline_pixels_transform_and_fields(monkeypatch):
    image,mask = scene()
    install_backend(monkeypatch,mask,'unsupported_stamp')
    result = PersianCardOCR(extractor=CardExtractor(), reader=FieldsReader(),
                            preprocessing='rectify_best_effort').process(image)
    assert_six_fields(result)
    metadata = result.metadata['extraction']['metadata']
    assert metadata['stamp_recovery']['attempted']
    assert not metadata['stamp_recovery']['replaced']
    quad,_ = quad_from_mask(mask)
    expected = warp_candidates(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR),quad)[0]
    np.testing.assert_array_equal(np.asarray(result.image),
                                  cv2.cvtColor(expected['card'],cv2.COLOR_BGR2RGB))
    np.testing.assert_allclose(metadata['original_to_card_H'],expected['H'])
    assert metadata['mask_quad_iou'] > .99
