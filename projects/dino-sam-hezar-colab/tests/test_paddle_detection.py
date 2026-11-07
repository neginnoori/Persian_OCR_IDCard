import numpy as np
import pytest
from PIL import Image
from persian_id_ocr.detection import PaddleTextDetector, crop_polygon
from persian_id_ocr.reader import HezarReader


def detector_with_result(tmp_path, result):
    detector = PaddleTextDetector(tmp_path, device='cpu')
    class Model:
        def predict(self, image, **kwargs):
            assert image[0,0].tolist() == [0,0,255]  # RGB red -> BGR
            return [result]
    detector.model = Model()
    return detector


def test_original_coordinates_polygon_scores_and_bgr(tmp_path):
    detector = detector_with_result(tmp_path, {'dt_polys': [[[10,10],[90,12],[88,32],[8,30]]], 'dt_scores':[.9]})
    result = detector.detect(Image.new('RGB',(200,100),'red'))
    assert result[0]['box'] == [8,10,91,33]
    assert result[0]['polygon'][0] == [10.,10.]
    assert result[0]['detection_score'] == .9


def test_empty_and_malformed_results(tmp_path):
    image = Image.new('RGB',(200,100),'red')
    assert detector_with_result(tmp_path, {'dt_polys':[], 'dt_scores':[]}).detect(image) == []
    with pytest.raises(RuntimeError, match='counts'):
        detector_with_result(tmp_path, {'dt_polys':[], 'dt_scores':[.9]}).detect(image)
    with pytest.raises(RuntimeError, match='invalid'):
        detector_with_result(tmp_path, {'dt_polys':[[[float('nan'),0]]*4], 'dt_scores':[.9]}).detect(image)


def test_crop_is_rgb_and_padded_without_changing_source():
    card = Image.new('RGB',(200,100),(100,150,200))
    crop = crop_polygon(card, [[10,20],[100,20],[100,40],[10,40]])
    assert crop.mode == 'RGB' and crop.width > 90 and crop.height > 20
    assert np.asarray(crop)[crop.height//2,crop.width//2].tolist() == [100,150,200]
    assert card.size == (200,100)


def test_paddle_regions_reach_only_hezar_recognizer(tmp_path):
    reader = HezarReader(tmp_path, device='cpu')
    class Detector:
        def detect(self, card):
            return [{'box':[10,20,100,40], 'polygon':[[10,20],[100,20],[100,40],[10,40]], 'detection_score':.8}]
    class Recognizer:
        def predict(self, crops):
            assert len(crops) == 1 and crops[0].height > 20
            return [{'text':'۱۲۳۴۵۶۷۸۹۰'}]
    reader.detector, reader.recognizer = Detector(), Recognizer()
    result = reader.read(Image.new('RGB',(200,100)))
    assert result[0]['text'] == '1234567890'
    assert result[0]['detection_scores'] == [.8]


def test_no_detections_never_call_recognition(tmp_path):
    reader = HezarReader(tmp_path, device='cpu')
    class Detector:
        def detect(self, card): return []
    class Recognizer:
        def predict(self, crops): raise AssertionError('must not run')
    reader.detector, reader.recognizer = Detector(), Recognizer()
    assert reader.read(Image.new('RGB',(200,100))) == []


def test_worker_missing_environment_is_actionable(tmp_path):
    from persian_id_ocr.paddle_client import PaddleWorkerDetector
    detector = PaddleWorkerDetector(tmp_path, python=str(tmp_path/'missing'))
    with pytest.raises(RuntimeError, match='setup_colab'):
        detector.load()


def test_default_reader_loads_hezar_crnn_but_not_craft(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from persian_id_ocr.paddle_client import PaddleWorkerDetector
    loaded = []
    class Model:
        @staticmethod
        def load(name, **kwargs):
            loaded.append(name)
            return SimpleNamespace(eval=lambda: None)
    monkeypatch.setitem(sys.modules, 'hezar.models', SimpleNamespace(Model=Model))
    monkeypatch.setattr(PaddleWorkerDetector, 'load', lambda self: self)
    reader = HezarReader(tmp_path, device='cpu')
    reader.load()
    reader.load()
    assert loaded == ['hezarai/crnn-base-fa-v2']


def test_isolated_worker_reuses_process_and_closes(tmp_path, monkeypatch):
    import sys
    import persian_id_ocr.paddle_client as client
    monkeypatch.setattr(client, '__file__', str(tmp_path/'paddle_client.py'))
    (tmp_path/'paddle_worker.py').write_text(
        "import sys,json\nprint('diagnostic log',flush=True)\n"
        "print('PADDLE_JSON:'+json.dumps({'ready':True}),flush=True)\n"
        "for line in sys.stdin:\n print('PADDLE_JSON:'+json.dumps({'regions':[]}),flush=True)\n")
    detector = client.PaddleWorkerDetector(tmp_path, python=sys.executable, timeout=5)
    try:
        assert detector.detect(Image.new('RGB',(20,20))) == []
        process = detector.process
        assert detector.detect(Image.new('RGB',(20,20))) == []
        assert detector.process is process
    finally:
        detector.close()
    assert process.poll() is not None


def test_isolated_worker_failure_is_visible(tmp_path, monkeypatch):
    import sys
    import persian_id_ocr.paddle_client as client
    monkeypatch.setattr(client, '__file__', str(tmp_path/'paddle_client.py'))
    (tmp_path/'paddle_worker.py').write_text("print('PADDLE_JSON:{\"error\":\"model unavailable\"}',flush=True)\n")
    detector = client.PaddleWorkerDetector(tmp_path, python=sys.executable)
    with pytest.raises(RuntimeError, match='model unavailable'):
        detector.load()
    assert detector.process is None
