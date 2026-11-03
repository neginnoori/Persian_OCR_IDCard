import numpy as np
import pytest
from PIL import Image
from cardid import CardExtractor
from cardid.backend import Detection
from persian_id_ocr import PersianCardOCR


@pytest.mark.parametrize('turns', range(4))
def test_orientation_only_preserves_pixels_and_skips_geometry(monkeypatch, turns):
    import cardid.extractor as implementation
    rgb = np.arange(40*60*3, dtype=np.uint8).reshape(40,60,3)
    mask = np.ones((40,60), np.uint8)  # touches every border
    class Backend:
        info = {}
        reader = face_detector = None
        def __init__(self, config): pass
        def detect_masks(self, image): return Detection([mask], [.9], .8)
    def forbidden(*args, **kwargs):
        raise AssertionError('Geometry must not run in orientation_only mode')
    monkeypatch.setattr(implementation, 'TorchBackend', Backend)
    monkeypatch.setattr(implementation, 'quad_from_mask', forbidden)
    monkeypatch.setattr(implementation, 'warp_candidates', forbidden)
    class Reader:
        calls = 0
        def read(self, image):
            self.calls += 1
            np.testing.assert_array_equal(np.asarray(image), np.rot90(rgb, turns))
            return [{'box':[2,2,10,10], 'text_raw':'نمونه', 'text':'نمونه'}]
    reader = Reader()
    result = PersianCardOCR(extractor=CardExtractor(), reader=reader).process(
        Image.fromarray(rgb), orientation_override=90*turns)
    assert result.status == 'ocr_candidate'
    assert reader.calls == 1
    meta = result.metadata['extraction']['metadata']
    assert meta['geometry_check'] == 'not_applied'
    assert not meta['perspective_warp_applied']
    # Verify that H maps an identifiable source pixel to the rotated location.
    point = np.array([5,7,1.])
    x,y,_ = np.asarray(meta['original_to_card_H']) @ point
    np.testing.assert_array_equal(np.asarray(result.image)[int(y),int(x)], rgb[7,5])


def test_uncertain_orientation_and_missing_mask_still_run_text_reader(monkeypatch):
    import cardid.extractor as implementation
    class Backend:
        info = {}
        reader = face_detector = None
        def __init__(self, config): pass
        def detect_masks(self, image): return Detection([], [], None)
    monkeypatch.setattr(implementation, 'TorchBackend', Backend)
    monkeypatch.setattr(implementation, 'choose_orientation', lambda candidates, *a, **kw:
                        (candidates[1], {'orientation_ok':False, 'orientation_margin':0.1}))
    class Reader:
        calls = 0
        def read(self, image):
            self.calls += 1
            assert image.size == (40,60)
            return [{'box':[2,2,10,10], 'text_raw':'نمونه', 'text':'نمونه'}]
    reader = Reader()
    result = PersianCardOCR(extractor=CardExtractor(), reader=reader).process(Image.new('RGB',(60,40)))
    assert reader.calls == 1 and result.status == 'ocr_candidate'
    assert result.text == 'نمونه'
    meta = result.metadata['extraction']['metadata']
    assert 'ambiguous_orientation_best_candidate_used' in meta['warnings']
    assert 'no_card_mask_using_full_image' in meta['warnings']
    assert result.metadata['field_assignment'] == 'unavailable'
    assert all(f['status'] == 'missing' for f in result.fields)


def test_field_frame_uses_offset_without_changing_output_coordinates():
    from persian_id_ocr.fields import assign_fields
    regions=[{'box':[650,300,850,350], 'text_raw':'۱۲۳۴۵۶۷۸۹۰', 'text':'1234567890'}]
    fields=assign_fields(regions, (1400,900), frame=[100,100,1100,900])
    assert fields[0]['value'] == '1234567890'
    assert fields[0]['box'] == regions[0]['box']
    assert fields[0]['assignment_method'] == 'card_frame_layout_approximate_v1'
