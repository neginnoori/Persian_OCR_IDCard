from persian_id_ocr.fields import assign_fields, FIELDS


def test_six_fields_and_label_exclusion():
    regions = []
    texts = ['۱۲۳۴۵۶۷۸۹۰', 'سارا', 'احمدی', '۱۳۸۰/۰۱/۲۱', 'عباس', '۱۴۰۹/۰۴/۰۱']
    for i, text in enumerate(texts):
        y = (0.28+.1*i)*540
        regions.append({'box':[450,y-10,680,y+10], 'text_raw':text})
        regions.append({'box':[720,y-10,850,y+10], 'text_raw':'عنوان'})
    rows = assign_fields(regions,(856,540))
    assert [r['field'] for r in rows] == [f for f,_ in FIELDS]
    assert rows[0]['value'] == '1234567890'
    assert all(r['status']=='candidate' and len(r['source_region_ids'])==1 for r in rows)


def test_missing_field_does_not_shift_other_fields():
    rows=assign_fields([{'box':[450, .48*540-10,680,.48*540+10], 'text_raw':'احمدی'}],(856,540))
    assert rows[1]['status']=='missing'
    assert rows[2]['value']=='احمدی'
    assert len(rows)==6


def test_merged_numeric_labels_are_not_missing_and_raw_is_preserved():
    regions = [
        {'box': [483, 134, 841, 180], 'text_raw': 'شماره ملی۵۶۰۳۷۷۸۲۶'},
        {'box': [482, 414, 806, 463], 'text_raw': 'بابان اعار ۱۴۰۳/۱۲/۳۳'},
    ]
    rows = assign_fields(regions, (856, 540))
    assert rows[0]['value'] == '560377826'
    assert rows[0]['status'] == 'invalid_format'
    assert rows[0]['text_raw'] == regions[0]['text_raw']
    assert rows[0]['source_region_ids'] == [0]
    assert rows[5]['value'] == '1403/12/33'
    assert rows[5]['status'] == 'invalid_format'


def test_merged_numeric_value_in_offset_card_frame_and_external_stamp():
    regions = [
        {'box': [687, 716, 1129, 774], 'text_raw': 'شماره ملی ۱۲۳۴۵۶۷۸۹۰'},
        {'box': [630, 1450, 1100, 1500], 'text_raw': 'شماره ثبت ۹۸۷۶۵۴۳۲۱۰'},
    ]
    rows = assign_fields(regions, (1240, 1754), frame=[95, 560, 1149, 1196])
    assert rows[0]['value'] == '1234567890'
    assert rows[0]['source_region_ids'] == [0]
    assert all(row['source_region_ids'] != [1] for row in rows)


def test_label_only_and_multiple_numbers_do_not_become_valid_id():
    rows = assign_fields([
        {'box': [720, 134, 841, 180], 'text_raw': 'شماره ملی'},
    ], (856, 540))
    assert rows[0]['status'] == 'missing'
    rows = assign_fields([
        {'box': [483, 134, 841, 180], 'text_raw': '۱۲۳۴۵۶ شماره ۷۸۹۰'},
    ], (856, 540))
    assert rows[0]['status'] == 'invalid_format'


def test_numeric_values_exclude_stamp_text_and_keep_source_box():
    rows = assign_fields([
        {'box':[273,136,419,191], 'text_raw':'چثگگگگ'},
        {'box':[473,136,672,178], 'text_raw':'۳۹۳۲۶۵۲۹۳۲'},
        {'box':[273,294,459,370], 'text_raw':'کپی برابر اصل است'},
        {'box':[475,308,661,345], 'text_raw':'۱۳۵۰/۰۴/۰۱'},
    ], (856,540))
    assert rows[0]['value'] == '3932652932'
    assert rows[0]['source_region_ids'] == [1]
    assert rows[3]['value'] == '1350/04/01'
    assert rows[3]['source_region_ids'] == [3]


def test_unrelated_numbers_are_not_concatenated_and_leading_zero_survives():
    rows = assign_fields([{'box':[450,140,680,170], 'text_raw':'کد ملی ۰۰۱۲۳۴۵۶۷۸'}],(856,540))
    assert rows[0]['value'] == '0012345678'
    rows = assign_fields([{'box':[450,140,680,170], 'text_raw':'۱۲۳۴ مهر ۵۶۷۸۹۰'}],(856,540))
    assert rows[0]['value'] is None
    assert rows[0]['status'] == 'invalid_format'
