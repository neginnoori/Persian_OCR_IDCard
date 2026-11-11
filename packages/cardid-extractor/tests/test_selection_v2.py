from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from cardid.backend import Detection
from cardid.types import ExtractionResult, ExtractorConfig
from cardid.selection import diverse_boxes, card_support, select_card_v2


def evidence(**kwargs):
    return dict(value_row_count=6, value_row_span=.5, date_row_count=0,
                face_on_left=False, matched_keywords=['نام','تولد','پدر'],
                stamp_keywords=[], **kwargs)


def baseline(stamp=True):
    e = dict(value_row_count=1, date_row_count=0, face_on_left=False,
             stamp_keywords=['شرکت'] if stamp else [], score=20)
    return ExtractionResult('needs_review', Image.new('RGB',(856,540)), 'ambiguous_orientation',
        dict(source_corners=[[300,200],[500,200],[500,320],[300,320]],
             corner_shift_degrees=0, orientation_scores=[e]*4, elapsed_ms=1))


def test_diversity_keeps_nested_card_despite_duplicate_stamp_scores():
    boxes = [[300,200,500,320],[301,201,501,321],[299,200,499,320],
             [20,20,950,580],[10,5,100,50],[20,90,100,140]]
    indices, audit = diverse_boxes(boxes,[.99,.98,.97,.6,.7,.65],limit=2)
    assert indices == [0,3]
    assert audit[1]['reason'] == 'duplicate'


def test_face_and_date_not_both_mandatory():
    assert card_support(evidence())['supported']
    assert not card_support(dict(value_row_count=6,value_row_span=.5,
                                 stamp_keywords=['شرکت'],score=100))['supported']


def test_working_baseline_does_not_run_new_models():
    original = baseline(stamp=False)
    actual = select_card_v2(Image.new('RGB',(1000,600)), object(), ExtractorConfig(), original)
    assert actual is original
    assert not actual.metadata['candidate_selection']['attempted']


def test_manual_override_preserves_baseline():
    original = baseline()
    assert select_card_v2(Image.new('RGB',(1000,600)), object(), ExtractorConfig(), original,90) is original


def test_all_rotations_ranked_by_card_evidence(monkeypatch):
    import cardid.geometry as geometry
    mask = np.zeros((600,1000),np.uint8);mask[20:580,20:980]=1
    backend=SimpleNamespace(detect_diverse_masks=lambda _: Detection([mask],[.99],.8),reader=None,face_detector=None)
    scores = iter([dict(score=100,stamp_keywords=['شرکت']),dict(score=0),evidence(score=2),dict(score=0)])
    monkeypatch.setattr(geometry,'score_orientation',lambda *a:next(scores))
    actual=select_card_v2(Image.new('RGB',(1000,600)),backend,ExtractorConfig(),baseline())
    assert actual.metadata['corner_shift_degrees']==180
    assert actual.metadata['candidate_selection']['replaced']
    assert actual.metadata['orientation_ok']
    assert len(actual.metadata['candidate_selection']['candidates'][0]['orientations'])==4


def test_unresolved_stamp_returns_full_source_not_template(monkeypatch):
    import cardid.geometry as geometry
    backend=SimpleNamespace(detect_diverse_masks=lambda _: Detection([],[],None),reader=None,face_detector=None)
    scores=iter([{},evidence(score=2),{},{}])
    monkeypatch.setattr(geometry,'score_orientation',lambda *a:next(scores))
    pixels=np.arange(800*600*3,dtype=np.uint8).reshape(600,800,3)
    actual=select_card_v2(Image.fromarray(pixels),backend,ExtractorConfig(),baseline())
    assert actual.metadata['field_assignment_mode']=='anchors_v1'
    assert actual.metadata['rotation_ccw_degrees']==90
    assert np.array_equal(np.asarray(actual.image),np.rot90(pixels))
    H=np.array(actual.metadata['original_to_card_H'])
    assert np.allclose(H @ np.array([799,0,1]),[0,0,1])
    assert actual.metadata['card_frame'] is None


def test_model_errors_remain_visible():
    def fail(_):raise RuntimeError('model failed')
    with pytest.raises(RuntimeError,match='model failed'):
        select_card_v2(Image.new('RGB',(1000,600)),SimpleNamespace(detect_diverse_masks=fail),ExtractorConfig(),baseline())


def test_config_defaults_to_legacy():
    assert ExtractorConfig().candidate_selection=='legacy'
    with pytest.raises(ValueError): ExtractorConfig(candidate_selection='typo')


def test_full_extractor_v2_preserves_working_card_pixels_and_transform(monkeypatch):
    import cardid.extractor as implementation
    from cardid import CardExtractor
    mask=np.zeros((600,1000),np.uint8); mask[30:570,70:930]=1
    class Reader:
        def readtext(self,*a,**k):
            return [([[400,100],[700,100],[700,125],[400,125]],'نام خانوادگی',.95),
                    ([[400,200],[700,200],[700,225],[400,225]],'تاریخ تولد',.95)]
    class Backend:
        reader=Reader();face_detector=None; info={}
        def __init__(self,c):pass
        def detect_masks(self,p):return Detection([mask],[.99],.8)
        def detect_diverse_masks(self,p):raise AssertionError('working baseline changed')
    monkeypatch.setattr(implementation,'TorchBackend',Backend)
    pixels=np.arange(600*1000*3,dtype=np.uint8).reshape(600,1000,3)
    legacy=CardExtractor(ExtractorConfig()).extract_result(pixels)
    v2=CardExtractor(ExtractorConfig(candidate_selection='v2')).extract_result(pixels)
    assert np.array_equal(np.asarray(legacy.image),np.asarray(v2.image))
    for key in ['original_to_card_H','source_corners','orientation_scores']:
        assert legacy.metadata[key]==v2.metadata[key]
    assert not v2.metadata['candidate_selection']['attempted']


def test_v3_recovers_unreadable_small_stamp_without_keywords(monkeypatch):
    import cardid.geometry as geometry
    backend=SimpleNamespace(detect_diverse_masks=lambda _: Detection([],[],None),reader=None,face_detector=None)
    monkeypatch.setattr(geometry,'score_orientation',lambda *a:{})
    original=baseline(stamp=False)
    result=select_card_v2(Image.new('RGB',(1000,600)),backend,ExtractorConfig(candidate_selection='v3'),original)
    assert result.metadata['candidate_selection']['attempted']
    assert result.metadata['field_assignment_mode']=='anchors_v1'


def test_v3_preserves_supported_baseline(monkeypatch):
    original=baseline(stamp=False)
    original.metadata['orientation_scores']=[evidence()]*4
    result=select_card_v2(Image.new('RGB',(1000,600)),object(),ExtractorConfig(candidate_selection='v3'),original)
    assert result is original
    assert not result.metadata['candidate_selection']['attempted']
