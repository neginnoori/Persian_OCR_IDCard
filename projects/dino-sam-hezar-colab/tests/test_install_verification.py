import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_installed_sources_must_match(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/verify_install.py'
    spec = importlib.util.spec_from_file_location('verify_install', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path/'project'
    sources = {'cardid': project/'vendor/cardid-extractor/src/cardid',
               'persian_id_ocr': project/'src/persian_id_ocr'}
    installed = {}
    for name, source in sources.items():
        source.mkdir(parents=True)
        (source/'__init__.py').write_text('# current\n')
        destination = tmp_path/'installed'/name
        destination.mkdir(parents=True)
        (destination/'__init__.py').write_text('# current\n')
        installed[name] = destination
    monkeypatch.setattr(module.importlib.util, 'find_spec',
                        lambda name: SimpleNamespace(origin=str(installed[name]/'__init__.py')))
    assert len(module.verify(project)) == 2
    (installed['cardid']/'__init__.py').write_text('# stale\n')
    with pytest.raises(RuntimeError, match='Stale installed code'):
        module.verify(project)
