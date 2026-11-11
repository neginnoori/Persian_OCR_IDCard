from PIL import Image
from persian_id_ocr.adaptive import repair_fields, compatible
from persian_id_ocr.fields import assign_fields
from persian_id_ocr.anchors import assign_anchored_fields

def sample():
    texts=['شماره ملی ۰۰۱۲۳۴۵۶۷۸','نام نمونه','نام خانوادگی آزمایشی','تاریخ تولد ۱۳۷۰/۰۱/۰۱','نام پدر نمونه','پایان اعتبار ۱۴۰۸/۰۲/۰۱']
    return [dict(box=[500,240+45*i,900,265+45*i],text_raw=t,text=t) for i,t in enumerate(texts)]

def test_shifted_rows_repaired_and_baseline_retained():
    r=sample(); b=assign_fields(r,(1000,700))
    out,a=repair_fields(Image.new('RGB',(1000,700)),r,b)
    assert all(f['value'] for f in out)
    assert out[1]['value']=='نمونه'
    assert a['baseline_fields']==b
    assert any(d['replaced'] for d in a['decisions'])

def test_working_fields_unchanged():
    r=sample();b,_=assign_anchored_fields(r,(1000,700))
    out,a=repair_fields(Image.new('RGB',(1000,700)),r,b)
    assert out==b and not any(d['replaced'] for d in a['decisions'])

def test_stamp_does_not_replace_fields_without_anchors():
    r=sample();b,_=assign_anchored_fields(r,(1000,700))
    out,a=repair_fields(Image.new('RGB',(1000,700)),[dict(box=[0,0,100,40],text_raw='شرکت نمونه')],b)
    assert out==b

def test_numeric_and_date_are_not_names():
    assert not compatible('father_name','1370/01/01')
    assert not compatible('last_name','0012345678')

def test_valid_conflicting_same_region_not_overwritten():
    r=sample();b,_=assign_anchored_fields(r,(1000,700));b[0]['value']='0098765432'
    out,a=repair_fields(Image.new('RGB',(1000,700)),r,b)
    assert out[0]['value']=='0098765432'
    assert a['decisions'][0]['reason']=='conflicting_candidate_preserved_for_review'

def test_short_name_reread_is_audited_not_guessed():
    r=sample();r[4]['text_raw']='نام پدر لی';b,_=assign_anchored_fields(r,(1000,700))
    out,a=repair_fields(Image.new('RGB',(1000,700)),r,b,lambda crops:['قلی','قلی'])
    assert out[4]['value']=='لی'
    assert a['rereads'][0]['values']==['قلی','قلی']
    assert not a['rereads'][0]['accepted']

def test_missing_field_without_anchor_not_invented():
    r=sample()[1:];b=assign_fields([],(1000,700))
    out,a=repair_fields(Image.new('RGB',(1000,700)),r,b)
    assert out[0]['value'] is None
    assert out[4]['value']=='نمونه'

def test_pipeline_v3_keeps_raw_and_six_fields_with_audit():
    from types import SimpleNamespace
    from cardid.types import ExtractionResult
    from persian_id_ocr.pipeline import PersianCardOCR
    image=Image.new('RGB',(1000,700));r=sample()
    extractor=SimpleNamespace(extract_result=lambda *a,**k:ExtractionResult('ok',image,None,{}))
    engine=PersianCardOCR(extractor=extractor,reader=SimpleNamespace(read=lambda _:r),candidate_selection='v3')
    result=engine.process(image)
    assert result.text=='\n'.join(x['text_raw'] for x in r)
    assert len(result.fields)==6
    assert result.metadata['adaptive_fields']['baseline_fields']==assign_fields(r,image.size)
    assert result.fields[1]['value']=='نمونه'
