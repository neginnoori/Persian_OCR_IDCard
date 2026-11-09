from PIL import Image
from persian_id_ocr.fields import assign_fields
from persian_id_ocr.numeric_retry import retry_numeric_fields


def sample():
    return assign_fields([
        {'box': [482,133,841,180], 'text_raw':'شماره ملی۵۶۰۳۷۷۸۲۶'},
        {'box': [594,199,682,236], 'text_raw':'حمیده'},
        {'box': [430,249,678,295], 'text_raw':'متقیان'},
        {'box': [606,363,682,406], 'text_raw':'شاور'},
        {'box': [481,413,806,463], 'text_raw':'عنوان ۱۴۰۳/۱۲/۳۳'},
    ], (856,540))


def test_retry_preserves_raw_and_accepts_agreeing_numeric_reads():
    original = sample()
    answers = iter([['۱۲۳۴۵۶۷۸۹۰']*2, ['۱۴۰۳/۱۲/۲۳']*2])
    def recognize(crops):
        assert len(crops) == 2
        assert crops[0].height < crops[1].height
        return next(answers)
    rows = retry_numeric_fields(Image.new('RGB',(856,540)), original, recognize)
    assert rows[0]['value'] == '1234567890'
    assert rows[5]['value'] == '1403/12/23'
    assert rows[0]['text_raw'] == original[0]['text_raw']
    assert rows[0]['box'][2] < 700
    assert original[0]['value'] == '560377826'
    assert rows[0]['review_required']


def test_disagreeing_or_invalid_retry_never_overwrites_original():
    answers = iter([['۱۲۳۴۵۶۷۸۹۰','۱۲۳۴۵۶۷۸۹۱'], ['۱۴۰۳/۱۲/۳۳']*2])
    rows = retry_numeric_fields(Image.new('RGB',(856,540)), sample(), lambda crops: next(answers))
    assert rows[0]['value'] == '560377826'
    assert rows[5]['value'] == '1403/12/33'
    assert not rows[0]['numeric_retry']['accepted']
    assert not rows[5]['numeric_retry']['accepted']


def test_no_retry_without_value_column_evidence():
    fields = sample()[:1]
    def unexpected(crops):
        raise AssertionError('must not guess column')
    assert retry_numeric_fields(Image.new('RGB',(856,540)), fields, unexpected) == fields
