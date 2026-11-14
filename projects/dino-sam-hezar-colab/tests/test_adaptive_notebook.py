import json
from pathlib import Path
from types import SimpleNamespace


def test_comparison_reports_field_status_not_document_status(tmp_path, monkeypatch):
    import tempfile

    notebook = Path(__file__).parents[1] / 'DINO_SAM_Hezar_Adaptive_V3.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    source = ''.join(next(cell for cell in cells if cell.get('id') == 'compare-selection')['source'])
    baseline_root = tmp_path / 'baseline'
    baseline = baseline_root / 'run_example'
    selected = tmp_path / 'selected'
    summary = [{'input': '/input/synthetic.jpg', 'folder': 'synthetic'}]
    for folder, value, status in ((baseline, '0012345678', 'candidate'),
                                  (selected, None, 'invalid_format')):
        (folder / 'synthetic').mkdir(parents=True)
        report = {'status': 'ocr_candidate', 'fields': [
            {'field': 'last_name', 'value': value, 'status': status}]}
        (folder / 'synthetic' / 'result.json').write_text(json.dumps(report))
        (folder / 'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kwargs: str(baseline_root))
    namespace = {'Path': Path, 'tempfile': tempfile, 'json': json,
                 'command': ['python', '--output-root', str(selected), '--candidate-selection', 'v3'],
                 'subprocess': SimpleNamespace(run=lambda _: SimpleNamespace(returncode=0)),
                 'SUMMARY': summary, 'RESULTS': selected, 'CANDIDATE_SELECTION': 'v3',
                 'display': lambda _: None}
    exec(compile(source, str(notebook), 'exec'), namespace)
    row = next(row for row in namespace['comparison'] if row['field'] == 'last_name')
    assert row['v2_field_status'] == 'candidate'
    assert row['selected_field_status'] == 'invalid_format'
    assert row['v2_value'] == '0012345678'
    assert row['selected_value'] is None
    assert json.loads((selected / 'comparison.json').read_text()) == namespace['comparison']
