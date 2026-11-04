import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cardid import ExtractionResult
from persian_id_ocr import HezarReader, PersianCardOCR
from persian_id_ocr.reader import normalize_text


class Extractor:
    def __init__(self, status='ok'):
        self.status = status
        self.calls = []

    def orient_result(self, source, **kwargs):
        return self.extract_result(source, **kwargs)

    def extract_result(self, source, **kwargs):
        self.calls.append(kwargs)
        return ExtractionResult(self.status, Image.new('RGB', (856, 540)), reason=None)


class Reader:
    def __init__(self):
        self.calls = 0

    def read(self, card):
        self.calls += 1
        assert card.size == (856, 540)
        return [{'box': [20, 20, 80, 40], 'crop_box': [12, 12, 88, 48],
                 'text_raw': '۱۲۳ ي ك', 'text': '123 ی ک'}]


def test_accepted_card_reaches_hezar_once_and_preserves_raw_text():
    extractor, reader = Extractor(), Reader()
    pipeline = PersianCardOCR(extractor=extractor, reader=reader)
    result = pipeline.process('photo.jpg', color_order='BGR', orientation_override=180)
    assert result.status == 'ocr_candidate'
    assert result.text == '۱۲۳ ي ك' and result.text_normalized == '123 ی ک'
    assert extractor.calls == [{'color_order': 'BGR', 'orientation_override': 180}]
    assert reader.calls == 1
    assert json.loads(json.dumps(result.to_dict()))['review_required']
    pipeline.process('another.jpg')
    assert reader.calls == 2


@pytest.mark.parametrize('status', ['needs_review', 'not_found'])
def test_uncertain_extraction_never_runs_ocr(status):
    reader = Reader()
    result = PersianCardOCR(extractor=Extractor(status), reader=reader, preprocessing='rectify').process('photo.jpg')
    assert result.status == status and not result.text and not result.regions
    assert reader.calls == 0 and result.metadata['ocr_seconds'] == 0


def test_runtime_failure_is_visible():
    class Broken(Reader):
        def read(self, card):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        PersianCardOCR(extractor=Extractor(), reader=Broken()).process('photo.jpg')


def test_real_extractor_geometry_connects_to_hezar_reader(monkeypatch):
    import cardid.extractor as implementation
    from cardid import CardExtractor
    from cardid.backend import Detection
    mask = np.zeros((700, 1000), np.uint8)
    mask[80:620, 70:930] = 1
    class Backend:
        info = {'device': 'injected-test-backend'}
        reader = face_detector = None
        def __init__(self, config):
            pass
        def detect_masks(self, pil):
            return Detection([mask], [.95], .9)
    monkeypatch.setattr(implementation, 'TorchBackend', Backend)
    reader = Reader()
    result = PersianCardOCR(extractor=CardExtractor(), reader=reader, preprocessing='rectify').process(
        Image.new('RGB', (1000, 700)), orientation_override=0)
    assert result.status == 'ocr_candidate' and reader.calls == 1
    assert result.metadata['extraction']['metadata']['geometry_ok']
    assert result.image.size == (856, 540)


def test_no_text_has_explicit_status():
    reader = SimpleNamespace(read=lambda card: [])
    assert PersianCardOCR(extractor=Extractor(), reader=reader).process('photo.jpg').status == 'no_text'


def test_tight_crop_geometry_reaches_hezar(monkeypatch):
    import cardid.extractor as implementation
    from cardid import CardExtractor
    from cardid.backend import Detection
    class Backend:
        info = {'device': 'injected-test-backend'}
        reader = face_detector = None
        def __init__(self, config):
            pass
        def detect_masks(self, pil):
            return Detection([np.ones((540, 856), np.uint8)], [.95], .9)
    monkeypatch.setattr(implementation, 'TorchBackend', Backend)
    reader = Reader()
    result = PersianCardOCR(extractor=CardExtractor(), reader=reader, preprocessing='rectify').process(
        Image.new('RGB', (856, 540)), orientation_override=0)
    assert result.status == 'ocr_candidate'
    assert reader.calls == 1
    assert result.metadata['extraction']['metadata']['border_status'] == 'tight_crop_candidate'


def test_empty_recognized_strings_are_not_ocr_candidates():
    reader = SimpleNamespace(read=lambda card: [{'text_raw': ' ', 'text': ''}])
    assert PersianCardOCR(extractor=Extractor(), reader=reader).process('photo.jpg').status == 'no_text'


def fake_hezar(tmp_path, boxes, predictions=None):
    config = SimpleNamespace()
    processor = SimpleNamespace(config=config, get_ratio=lambda pixels: 1.)
    detector = SimpleNamespace(preprocessor=SimpleNamespace(image_processor=processor),
                               predict=lambda image: [{'boxes': boxes}])
    batches = []
    def predict(images):
        batches.append(images)
        return predictions if predictions is not None else [{'text': '۱۲۳ ي ك'} for _ in images]
    reader = HezarReader(tmp_path, device='cpu', batch_size=2, detector_backend='craft')
    reader.detector, reader.recognizer = detector, SimpleNamespace(predict=predict)
    return reader, batches


def test_crop_batching_and_rtl_order(tmp_path):
    reader, batches = fake_hezar(tmp_path, [[0, 10, 40, 20], [100, 10, 40, 20], [50, 60, 40, 20]])
    regions = reader.read(Image.new('RGB', (200, 100)))
    assert [len(batch) for batch in batches] == [2, 1]
    assert [r['box'][0] for r in regions] == [100, 0, 50]
    assert all(r['text'] == '123 ی ک' and r['text_raw'] == '۱۲۳ ي ك' for r in regions)
    assert regions[1]['crop_box'][0] == 0


def test_detection_coordinates_map_back_to_original_card(tmp_path):
    reader, _ = fake_hezar(tmp_path, [[10, 10, 40, 20]])
    result = reader.read(Image.new('RGB', (2560, 1080)))
    assert result[0]['box'] == [20, 20, 100, 60]


def test_empty_boxes_skip_recognizer(tmp_path):
    reader, batches = fake_hezar(tmp_path, [])
    assert reader.read(Image.new('RGB', (856, 540))) == [] and not batches


def test_split_national_id_is_recognized_as_one_crop(tmp_path):
    reader, batches = fake_hezar(tmp_path, [[487,131,76,31], [563,131,121,31], [715,133,129,30]])
    regions = reader.read(Image.new('RGB', (856, 540)))
    merged = next(r for r in regions if len(r['source_boxes']) == 2)
    assert merged['box'] == [487,131,684,162]
    assert len(regions) == 2  # label is still separate
    assert sum(len(batch) for batch in batches) == 2


def test_adjacent_different_rows_do_not_merge(tmp_path):
    reader, _ = fake_hezar(tmp_path, [[10,10,40,20], [12,45,40,20]])
    assert len(reader.read(Image.new('RGB',(200,100)))) == 2


def test_mismatched_recognizer_output_is_error(tmp_path):
    reader, _ = fake_hezar(tmp_path, [[10, 10, 40, 20]], predictions=[])
    with pytest.raises(RuntimeError, match='count'):
        reader.read(Image.new('RGB', (856, 540)))


def test_normalization_preserves_raw_source():
    assert normalize_text('۱۲۳\n ي ك') == '123 ی ک'


def test_batch_results_are_isolated_and_errors_are_reported(tmp_path):
    path = Path(__file__).parents[1]/'scripts/run_batch.py'
    spec = importlib.util.spec_from_file_location('run_batch', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    pipeline = PersianCardOCR(extractor=Extractor(), reader=Reader(), preprocessing='rectify')
    run1, summary1 = worker.run_batch(['same.jpg'], tmp_path, tmp_path, pipeline=pipeline)
    assert summary1[0]['status'] == 'ocr_candidate'
    assert (run1/summary1[0]['folder']/'card.png').is_file()
    pipeline.extractor.status = 'needs_review'
    run2, summary2 = worker.run_batch(['same.jpg'], tmp_path, tmp_path, pipeline=pipeline)
    assert run2 != run1
    assert not (run2/summary2[0]['folder']/'card.png').exists()
    assert (run2/summary2[0]['folder']/'card_needs_review.png').is_file()
    class Broken:
        def process(self, *args, **kwargs):
            raise RuntimeError('model unavailable')
    run3, summary3 = worker.run_batch(['bad.jpg'], tmp_path, tmp_path, pipeline=Broken())
    report = json.loads((run3/summary3[0]['folder']/'result.json').read_text())
    assert report['status'] == 'error' and 'model unavailable' in report['error']

