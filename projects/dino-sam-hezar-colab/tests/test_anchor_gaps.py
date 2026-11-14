"""Synthetic shifted rows with unreadable name labels; no real identity data."""
from copy import deepcopy

from PIL import Image

from persian_id_ocr.adaptive import repair_fields
from persian_id_ocr.fields import assign_fields


SIZE = (856, 540)
VALUES = ['0012345678', 'سارا', 'آزمایشی', '1370/01/01', 'نمونه', '1408/02/01']


def shifted_regions():
    labels = ['ماره ملی', 'امو', 'غ خاوادئی', 'ارخ نولد', 'ام بدر', 'بن ،عبدر']
    regions = [{'box': [340, 190, 450, 220], 'text_raw': 'مهر تایید'}]
    for row, (label, value) in enumerate(zip(labels, VALUES)):
        cy = 252 + row * 39
        regions.append({'box': [680, cy - 12, 770, cy + 12], 'text_raw': label})
        regions.append({'box': [500 if row in (0, 3, 5) else 560, cy - 15, 660, cy + 15],
                        'text_raw': value})
    return regions


def repair(regions, baseline=None):
    if baseline is None:
        baseline = assign_fields(regions, SIZE)
    return repair_fields(Image.new('RGB', SIZE), regions, baseline)


def test_shifted_names_with_unreadable_labels_recover_from_value_rows():
    regions = shifted_regions()
    baseline = assign_fields(regions, SIZE)
    original = deepcopy(baseline)
    assert baseline[1]['value'] == 'مهر تایید'
    assert baseline[2]['value'] == VALUES[0]
    assert baseline[4]['value'] == VALUES[3]
    output, audit = repair(regions, baseline)
    assert [item['value'] for item in output] == VALUES
    assert audit['baseline_fields'] == baseline == original
    assert output[1]['source_region_ids'] == [4]
    assert output[2]['source_region_ids'] == [6]
    assert output[1]['assignment_method'] == 'adaptive_interpolated_row_v1'
    assert audit['row_recovery']['support_fields'] == ['national_id', 'date_of_birth', 'father_name']


def test_recovered_working_fields_are_preserved_on_another_pass():
    regions = shifted_regions()
    baseline, _ = repair(regions)
    output, audit = repair(regions, baseline)
    assert output == baseline
    assert not any(item['replaced'] for item in audit['decisions'])


def test_missing_name_does_not_use_unreadable_label_or_distant_stamp():
    regions = shifted_regions()
    regions[4]['text_raw'] = ''
    regions.append({'box': [300, 276, 450, 306], 'text_raw': 'تایید شده'})
    baseline = assign_fields(regions, SIZE)
    baseline[1].update(value=None, status='missing', box=None, source_region_ids=[])
    output, audit = repair(regions, baseline)
    assert output[1]['value'] is None
    assert audit['row_recovery']['rows'][0]['reason'] == 'no_aligned_value'


def test_two_competing_name_regions_abstain():
    regions = shifted_regions()
    regions.append({'box': [555, 276, 658, 306], 'text_raw': 'سیمین'})
    baseline = assign_fields(regions, SIZE)
    baseline[1].update(value=None, status='missing', box=None, source_region_ids=[])
    output, audit = repair(regions, baseline)
    assert output[1]['value'] is None
    assert audit['row_recovery']['rows'][0]['reason'] == 'ambiguous_row_values'


def test_unreadable_labels_need_three_separate_support_values():
    regions = shifted_regions()
    regions[9]['text_raw'] = 'نشان'
    output, audit = repair(regions)
    assert audit['row_recovery']['reason'] == 'no_consistent_anchors'
    assert output[1]['value'] != VALUES[1]


def test_inconsistent_value_column_does_not_interpolate():
    regions = shifted_regions()
    regions[10]['box'] = [470, 393, 540, 423]
    output, audit = repair(regions)
    assert audit['row_recovery']['reason'] == 'inconsistent_value_column'
    assert output[1]['value'] != VALUES[1]


def test_name_outside_supported_rows_is_not_extrapolated():
    regions = shifted_regions()
    regions[1]['text_raw'] = 'نشان'
    regions[11]['text_raw'] = 'پایان اعتبار'
    output, audit = repair(regions)
    assert audit['row_recovery']['reason'] == 'consistent_value_column'
    assert audit['row_recovery']['rows'] == []
    assert output[1]['value'] != VALUES[1]


def test_invalid_name_type_is_reported_without_losing_original_field():
    baseline = assign_fields([], SIZE)
    baseline[2].update(value='0012345678', text_raw='۰۰۱۲۳۴۵۶۷۸', status='candidate')
    output, audit = repair([], baseline)
    assert output[2]['value'] is None
    assert output[2]['status'] == 'invalid_format'
    assert output[2]['text_raw'] == baseline[2]['text_raw']
    assert audit['baseline_fields'][2]['value'] == '0012345678'
    assert audit['decisions'][2]['reason'] == 'invalid_name_value_rejected'


def test_interpolation_cannot_overwrite_plausible_name_on_same_row():
    regions = shifted_regions()
    baseline = assign_fields(regions, SIZE)
    baseline[1].update(value='آزاده', text_raw='آزاده', box=[550, 276, 655, 306], source_region_ids=[99])
    output, audit = repair(regions, baseline)
    assert output[1] == baseline[1]
    assert audit['decisions'][1]['reason'] == 'conflicting_candidate_preserved_for_review'


def test_adjacent_name_words_are_joined_but_distant_stamp_is_excluded():
    regions = shifted_regions()
    regions[4].update(box=[600, 276, 660, 306], text_raw='نمونه')
    regions.extend([
        {'box': [540, 276, 595, 306], 'text_raw': 'دوم'},
        {'box': [300, 276, 450, 306], 'text_raw': 'تایید شده'},
    ])
    output, _ = repair(regions)
    assert output[1]['value'] == 'نمونه دوم'
    assert output[1]['source_region_ids'] == [4, 13]


def test_numeric_noise_cannot_supply_an_unlabelled_name():
    regions = shifted_regions()
    regions[4]['text_raw'] = '۴۲۶۸۷'
    output, audit = repair(regions)
    assert output[1]['value'] != '42687'
    assert audit['row_recovery']['rows'][0]['reason'] == 'no_aligned_value'


def test_v2_pipeline_keeps_the_previous_template_output():
    from types import SimpleNamespace
    from cardid.types import ExtractionResult
    from persian_id_ocr.pipeline import PersianCardOCR

    image = Image.new('RGB', SIZE)
    regions = shifted_regions()
    for region in regions:
        region['text'] = region['text_raw']
    extractor = SimpleNamespace(extract_result=lambda *a, **k: ExtractionResult('ok', image, None, {}))
    engine = PersianCardOCR(extractor=extractor, reader=SimpleNamespace(read=lambda _: regions),
                           candidate_selection='v2')
    result = engine.process(image)
    assert result.fields == assign_fields(regions, SIZE)
    assert 'adaptive_fields' not in result.metadata
