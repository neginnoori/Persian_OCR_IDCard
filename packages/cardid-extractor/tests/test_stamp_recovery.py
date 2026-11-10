"""Conservative recovery must not replace a usable crop on weak OCR evidence.

Synthetic OCR observations exercise selection policy, not model accuracy.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cardid.types import ExtractorConfig
from cardid.backend import Detection
import cardid.recovery as recovery


def observation(text, y, x=.55, width=.23, confidence=.95):
    return ([[x*1000, y*600], [(x+width)*1000, y*600],
             [(x+width)*1000, (y+.04)*600], [x*1000, (y+.04)*600]],
            text, confidence)


def card_observations():
    return [observation(text, y) for text, y in zip(
        ['۱۲۳۴۵۶۷۸۹۰', 'نمونه', 'آزمایشی', '۱۳۷۰/۰۲/۱۲', 'نمونه', '۱۴۰۸/۰۳/۲۱'],
        [.26, .36, .46, .56, .66, .76])]


def evidence(**updates):
    result = dict(stamp_keywords=['کپی', 'اصل'], face_on_left=True,
                  value_row_count=6, value_row_span=.5, date_row_count=2)
    result.update(updates)
    return result


def orientation(selected, turn=0):
    scores = [evidence(face_on_left=False, value_row_count=0,
                       value_row_span=0., date_row_count=0) for _ in range(4)]
    scores[turn//90] = selected
    return dict(corner_shift_degrees=turn, orientation_scores=scores,
                orientation_ok=False, orientation_margin=.1)


def candidate(quad=None):
    return {'quad': np.float32(quad if quad is not None else
                              [[400,200], [600,200], [600,320], [400,320]]),
            'card': np.zeros((540,856,3), np.uint8), 'H': np.eye(3),
            'corner_shift_degrees': 0}


def test_separate_right_column_rows_are_positive_layout_evidence():
    result = recovery.layout_evidence(card_observations(), 1000, 600)
    assert result['value_row_count'] == 6
    assert result['date_row_count'] == 2
    assert result['value_row_span'] >= .45


def test_duplicate_boxes_on_one_row_do_not_make_a_card_layout():
    results = [observation('۱۳۷۰/۰۲/۱۲', .4, x=.52+i*.01) for i in range(6)]
    result = recovery.layout_evidence(results, 1000, 600)
    assert result['value_row_count'] == 1
    assert result['date_row_count'] == 1
    assert not recovery.supported_card(dict(result, face_on_left=True))


def test_left_side_stamp_text_does_not_supply_value_rows():
    results = [observation(text, y, x=.05, width=.20)
               for text,y in zip(['شرکت', 'کپی برابر اصل است', 'نمونه', '۱۲۳۴۵'],
                                 [.26,.36,.46,.56])]
    result = recovery.layout_evidence(results, 1000, 600)
    assert result['stamp_keywords']
    assert result['value_row_count'] == 0


def test_low_confidence_rows_do_not_support_replacement():
    rows = [(box,text,.1) for box,text,_ in card_observations()]
    result = recovery.layout_evidence(rows, 1000, 600)
    assert not recovery.supported_card(dict(result, face_on_left=True))


@pytest.mark.parametrize('updates', [
    {'face_on_left':False}, {'value_row_count':3},
    {'value_row_span':.29}, {'date_row_count':0},
])
def test_supported_card_requires_combined_evidence(updates):
    assert recovery.supported_card(evidence())
    assert not recovery.supported_card(evidence(**updates))


def test_selected_evidence_uses_the_chosen_rotation():
    chosen = evidence(value_row_count=4)
    assert recovery.selected_evidence(orientation(chosen, 270)) == chosen


def test_a_card_with_a_stamp_is_not_rejected_when_layout_is_supported():
    assert not recovery.recovery_needed(candidate(), orientation(evidence()), (1000,600))


def test_missing_keywords_alone_never_trigger_recovery():
    weak = evidence(stamp_keywords=[], face_on_left=False, value_row_count=0,
                    value_row_span=0., date_row_count=0)
    assert not recovery.recovery_needed(candidate(), orientation(weak), (1000,600))


def test_positive_small_stamp_can_trigger_recovery():
    weak = evidence(face_on_left=False, value_row_count=1, value_row_span=0., date_row_count=0)
    assert recovery.recovery_needed(candidate(), orientation(weak), (1000,600))


@pytest.mark.parametrize('existing_card_signal', [
    {'face_on_left':True}, {'date_row_count':1}, {'value_row_count':4},
])
def test_partial_card_evidence_protects_baseline_even_with_stamp_text(existing_card_signal):
    weak = evidence(face_on_left=False, value_row_count=1, value_row_span=0.,
                    date_row_count=0, keyword_count=0, matched_keywords=[])
    weak.update(existing_card_signal)
    assert not recovery.supported_card(weak)  # A full rescue proof is intentionally stricter.
    assert not recovery.recovery_needed(candidate(), orientation(weak), (1000,600))


def test_large_crop_is_preserved_despite_stamp_text():
    large = candidate([[0,0],[999,0],[999,599],[0,599]])
    weak = evidence(face_on_left=False, value_row_count=1, value_row_span=0., date_row_count=0)
    assert not recovery.recovery_needed(large, orientation(weak), (1000,600))


def test_failed_rescue_preserves_original_image_transform_and_diagnostics():
    class EmptyReader:
        def readtext(self, *args, **kwargs):
            return []

    backend = SimpleNamespace(reader=EmptyReader(), face_detector=None,
                              detect_alternative_masks=lambda *args, **kwargs: Detection([], [], None))
    original = candidate()
    direction = orientation(evidence(face_on_left=False, value_row_count=1,
                                     value_row_span=0., date_row_count=0))
    metrics = {'geometry_ok':True, 'mask_quad_iou':.99, 'sam_predicted_iou':.99}
    chosen, actual_direction, actual_metrics, audit = recovery.recover_stamp_candidate(
        Image.new('RGB', (1000,600), 'white'), backend, ExtractorConfig(),
        original, direction, metrics)
    assert chosen is original
    assert actual_direction == direction
    assert actual_metrics == metrics
    assert audit is not None


def test_supported_baseline_never_runs_alternative_models():
    def unexpected(*args, **kwargs):
        raise AssertionError('Working baseline must be retained without a rescue call')
    backend = SimpleNamespace(detect_alternative_masks=unexpected)
    original = candidate()
    direction = orientation(evidence())
    metrics = {'geometry_ok':True}
    chosen, _, _, audit = recovery.recover_stamp_candidate(
        Image.new('RGB', (1000,600)), backend, ExtractorConfig(), original, direction, metrics)
    assert chosen is original
    assert not audit['attempted']


@pytest.mark.parametrize('image_size', [(1000,600), (600,1000)])
def test_larger_supported_full_card_can_replace_small_stamp(monkeypatch, image_size):
    import cardid.geometry as geometry
    original = candidate()
    direction = orientation(evidence(face_on_left=False, value_row_count=1,
                                     value_row_span=0., date_row_count=0))
    def choose(candidates, *args, **kwargs):
        return candidates[0], orientation(evidence())
    monkeypatch.setattr(geometry, 'choose_orientation', choose)
    backend = SimpleNamespace(reader=None, face_detector=None)
    chosen, actual_direction, metrics, audit = recovery.recover_stamp_candidate(
        Image.new('RGB', image_size), backend, ExtractorConfig(), original, direction,
        {'geometry_ok':True})
    assert chosen is not original
    assert recovery.supported_card(recovery.selected_evidence(actual_direction))
    assert metrics['quad_method'] == 'full_image_frame_recovery'
    assert not metrics['geometry_ok']  # An inferred frame is never called verified geometry.
    assert audit['replaced']
    assert np.isfinite(chosen['H']).all()


def test_high_text_score_without_card_structure_does_not_replace_baseline(monkeypatch):
    import cardid.geometry as geometry
    original = candidate()
    weak = evidence(face_on_left=False, value_row_count=1, value_row_span=0., date_row_count=0)
    direction = orientation(weak)
    stamp_like = evidence(score=100., face_on_left=False)
    monkeypatch.setattr(geometry, 'choose_orientation',
                        lambda candidates, *args, **kwargs: (candidates[0], orientation(stamp_like)))
    chosen, _, _, audit = recovery.recover_stamp_candidate(
        Image.new('RGB', (1000,600)), SimpleNamespace(reader=None, face_detector=None),
        ExtractorConfig(), original, direction, {'geometry_ok':True})
    assert chosen is original
    assert not audit['replaced']


def test_smaller_alternative_mask_cannot_displace_baseline():
    mask = np.zeros((800,800), np.uint8)
    mask[200:260, 200:300] = 1
    backend = SimpleNamespace(reader=None, face_detector=None,
        detect_alternative_masks=lambda *args, **kwargs: Detection([mask], [1.], .99))
    original = candidate()
    direction = orientation(evidence(face_on_left=False, value_row_count=1,
                                     value_row_span=0., date_row_count=0))
    chosen, _, _, audit = recovery.recover_stamp_candidate(
        Image.new('RGB', (800,800)), backend, ExtractorConfig(), original, direction,
        {'geometry_ok':True})
    assert chosen is original
    assert not audit['replaced']
