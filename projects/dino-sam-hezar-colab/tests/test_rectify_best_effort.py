import runpy
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from PIL import Image
from cardid import CardExtractor, ExtractionResult
from cardid.backend import Detection
from persian_id_ocr import PersianCardOCR


@pytest.mark.parametrize('turns', range(4))
def test_default_rectifies_diagonal_card_before_ocr(monkeypatch, turns):
    import cardid.extractor as implementation
    scene_path = Path(__file__).parents[3]/'packages/cardid-extractor/tests/test_geometry.py'
    card, photo, mask, _ = runpy.run_path(str(scene_path))['scene']()
    reference = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(np.rot90(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), turns))
    mask = np.ascontiguousarray(np.rot90(mask, turns))
    class OrientationReader:
        def readtext(self, image, **kwargs):
            if np.mean(np.abs(image.astype(float)-reference.astype(float))) < 3:
                return [([[300,100],[650,100],[650,130],[300,130]], 'نام خانوادگی', .94),
                        ([[300,200],[650,200],[650,230],[300,230]], 'تاریخ تولد', .94)]
            return []
    class Backend:
        info = {'device':'injected-test-backend'}
        reader = OrientationReader()
        face_detector = None
        def __init__(self, config): pass
        def detect_masks(self, image): return Detection([mask], [.95], .9)
    monkeypatch.setattr(implementation, 'TorchBackend', Backend)
    seen = []
    def read(image):
        seen.append(image)
        assert image.size == (856,540)
        assert np.mean(np.abs(np.asarray(image).astype(float)-reference.astype(float))) < 3
        return [{'box':[500,130,680,160], 'text_raw':'۱۲۳۴۵۶۷۸۹۰', 'text':'1234567890'}]
    result = PersianCardOCR(extractor=CardExtractor(), reader=SimpleNamespace(read=read)).process(rgb)
    assert len(seen) == 1 and result.status == 'ocr_candidate'
    assert result.metadata['effective_preprocessing'] == 'rectify'
    assert result.metadata['field_assignment'] == 'rectified_template'
    assert result.fields[0]['value'] == '1234567890'


@pytest.mark.parametrize('reason', ['unreliable_geometry', 'ambiguous_orientation', 'unreliable_geometry,ambiguous_orientation'])
def test_review_image_still_reaches_ocr_with_original_warning(reason):
    image = Image.new('RGB',(856,540))
    extraction = ExtractionResult('needs_review', image, reason=reason, metadata={'geometry_ok':False})
    extractor = SimpleNamespace(extract_result=lambda *a, **k: extraction)
    seen = []
    result = PersianCardOCR(extractor=extractor, reader=SimpleNamespace(read=lambda im: seen.append(im) or [])).process(image)
    assert seen == [image] and result.status == 'no_text'
    assert result.metadata['extraction']['status'] == 'needs_review'
    assert result.metadata['extraction']['metadata']['geometry_ok'] is False
    assert result.metadata['warnings'] == reason.split(',')


def test_missing_quad_falls_back_with_explicit_coordinates():
    image = Image.new('RGB',(1000,700))
    extractor = SimpleNamespace(
        extract_result=lambda *a, **k: ExtractionResult('needs_review', reason='no_reliable_quadrilateral'),
        orient_result=lambda *a, **k: ExtractionResult('ok', image, metadata={'warnings':['ambiguous_orientation_best_candidate_used']}))
    result = PersianCardOCR(extractor=extractor, reader=SimpleNamespace(read=lambda im: [])).process(image)
    assert result.metadata['coordinate_space'] == 'oriented_input'
    assert result.metadata['effective_preprocessing'] == 'orientation_only'
    assert result.metadata['rectification_attempt']['reason'] == 'no_reliable_quadrilateral'
    assert 'rectification_unavailable_orientation_only_fallback' in result.metadata['warnings']
    assert result.metadata['field_assignment'] == 'unavailable'


def test_model_exception_is_not_silently_fallback():
    def broken(*a, **k): raise RuntimeError('GPU inference failed')
    with pytest.raises(RuntimeError, match='GPU inference failed'):
        PersianCardOCR(extractor=SimpleNamespace(extract_result=broken), reader=object()).process('test.jpg')
