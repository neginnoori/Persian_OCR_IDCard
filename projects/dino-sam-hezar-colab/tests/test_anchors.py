from types import SimpleNamespace
from PIL import Image
from cardid.types import ExtractionResult
from persian_id_ocr.anchors import assign_anchored_fields
from persian_id_ocr.pipeline import PersianCardOCR


def regions():
    return [dict(box=[500,150+60*i,900,180+60*i],text_raw=text,text=text) for i,text in enumerate([
        'شماره ملی ۰۰۱۲۳۴۵۶۷۸','نام نمونه','نام خانوادگی آزمایشی',
        'تاریخ تولد ۱۳۷۷/۰۹/۰۹','نام پدر نمونه','پایان اعتبار ۱۴۰۸/۱۰/۰۲'])]


def test_inline_labels_produce_six_values_without_labels():
    fields,audit=assign_anchored_fields(regions(),(1000,700))
    assert [f['value'] for f in fields]==['0012345678','نمونه','آزمایشی','1377/09/09','نمونه','1408/10/02']
    assert all(f['assignment_method']=='label_anchors_v1' for f in fields)
    assert audit['anchor_count']==6


def test_stamp_only_never_supplies_name():
    r=[dict(box=[200,150,800,210],text_raw='شرکت نمونه نام شرکت'),dict(box=[200,250,800,310],text_raw='کپی برابر اصل است')]
    fields,_=assign_anchored_fields(r,(1000,700))
    assert len(fields)==6 and all(f['value'] is None for f in fields)


def test_multiple_documents_abstain():
    fields,audit=assign_anchored_fields(regions()+regions(),(1000,1400))
    assert all(f['value'] is None for f in fields)
    assert audit['reason']=='ambiguous_label_anchors'


def test_out_of_order_labels_abstain():
    r=regions();r[0]['box'],r[4]['box']=r[4]['box'],r[0]['box']
    fields,_=assign_anchored_fields(r,(1000,700))
    assert all(f['value'] is None for f in fields)


def test_anchor_fallback_reaches_paddle_and_returns_fields():
    image=Image.new('RGB',(1000,700))
    extraction=ExtractionResult('needs_review', image, 'unverified_card_frame',
        {'field_assignment_mode':'anchors_v1','preprocessing_mode':'orientation_only','card_frame':None})
    engine=PersianCardOCR(extractor=SimpleNamespace(extract_result=lambda *a,**k:extraction),
                          reader=SimpleNamespace(read=lambda _:regions()),candidate_selection='v2')
    result=engine.process(image)
    assert result.metadata['effective_preprocessing']=='orientation_only'
    assert result.metadata['field_assignment']=='label_anchors_v1'
    assert len(result.fields)==6 and all(f['value'] for f in result.fields)


def test_separate_labels_and_values_exclude_distant_stamp():
    r=[]
    from persian_id_ocr.anchors import LABELS
    for i,label in enumerate(LABELS):
        y=150+60*i
        r.append(dict(box=[800,y,950,y+30],text_raw=label[0]))
        value=['۱۲۳۴۵۶۷۸۹۰','نمونه','آزمایشی','۱۳۷۰/۰۱/۰۱','نمونه','۱۴۰۸/۰۲/۰۱'][i]
        r.append(dict(box=[590,y,780,y+30],text_raw=value))
    r.append(dict(box=[200,210,400,240],text_raw='شرکت نمونه'))
    fields,audit=assign_anchored_fields(r,(1000,700))
    assert all(f['value'] for f in fields)
    assert fields[1]['value']=='نمونه'
    assert fields[1]['source_region_ids']==[3]


def test_missing_one_label_leaves_only_that_field_missing():
    r=regions();del r[1]
    fields,audit=assign_anchored_fields(r,(1000,700))
    assert fields[1]['value'] is None
    assert sum(f['value'] is not None for f in fields)==5


def test_batch_saves_selection_audit_in_original_coordinates(tmp_path):
    import json, runpy
    from pathlib import Path
    script=Path(__file__).parents[1]/'scripts/run_batch.py'
    module=runpy.run_path(str(script))
    source=tmp_path/'synthetic.png';Image.new('RGB',(1000,700)).save(source)
    result=SimpleNamespace(status='no_text', image=None, fields=[], regions=[], text='',text_normalized='')
    audit={'version':'card_selection_v2','attempted':True,'selected_id':'mask_1',
           'detections':[{'dino_index':1,'box':[100,100,900,600],'reason':'segmented'}],
           'candidates':[{'id':'mask_1','quad':[[100,100],[900,100],[900,600],[100,600]],'reason':'supported'}]}
    result.to_dict=lambda:{'status':'no_text','metadata':{'extraction':{'metadata':{'original_size':[1000,700],'candidate_selection':audit}}}}
    engine=SimpleNamespace(process=lambda *a,**k:result)
    run,summary=module['run_batch']([source],tmp_path/'out',tmp_path/'models',pipeline=engine)
    folder=run/summary[0]['folder']
    assert summary[0]['status']=='no_text'
    assert (folder/'candidate_boxes.png').is_file()
    assert json.loads((folder/'candidate_selection.json').read_text())==audit
