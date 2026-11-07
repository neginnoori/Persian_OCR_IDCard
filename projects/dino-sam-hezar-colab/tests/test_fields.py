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
