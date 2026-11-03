"""Verify that installed Python modules match the selected project source."""
import hashlib
import importlib.util
import json
from pathlib import Path


def verify(project):
    project = Path(project).resolve()
    extractor = project / 'vendor/cardid-extractor'
    if not extractor.is_dir():
        extractor = project.parents[1] / 'packages/cardid-extractor'
    hashes = {}
    for name, source in [('cardid', extractor/'src/cardid'),
                         ('persian_id_ocr', project/'src/persian_id_ocr')]:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f'{name} is not installed in this Python environment')
        installed = Path(spec.origin).parent
        for path in sorted(source.rglob('*.py')):
            relative = path.relative_to(source)
            target = installed / relative
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Stale installed code: {target}. Rerun the installation cell.')
            hashes[f'{name}/{relative}'] = expected
    print(json.dumps({'source_verified': True, 'module_hashes': hashes}, indent=2))
    return hashes


if __name__ == '__main__':
    import sys
    verify(sys.argv[1])
