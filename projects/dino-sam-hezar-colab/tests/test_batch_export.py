import csv
import importlib.util
import json
from pathlib import Path
from PIL import Image
from persian_id_ocr.pipeline import OCRResult
from persian_id_ocr.fields import assign_fields


def test_numeric_retry_export_preserves_success_and_metadata(tmp_path):
    spec = importlib.util.spec_from_file_location('batch_export', Path(__file__).parents[1]/'scripts/run_batch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = assign_fields([], (856,540))
    fields[0]['numeric_retry'] = {'texts': ['۱۲۳'], 'accepted': False}
    class Pipeline:
        def process(self, *args, **kwargs):
            return OCRResult('ocr_candidate', image=Image.new('RGB',(856,540)), fields=fields,
                             metadata={'effective_preprocessing':'rectify'})
    run, summary = module.run_batch(['sample.jpg'], tmp_path, tmp_path, pipeline=Pipeline())
    folder = run/summary[0]['folder']
    assert summary[0]['status'] == 'ocr_candidate'
    report = json.loads((folder/'result.json').read_text())
    assert report['metadata']['effective_preprocessing'] == 'rectify'
    with (folder/'fields.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]['numeric_retry']) == fields[0]['numeric_retry']
    assert json.loads(rows[1]['numeric_retry']) is None
