"""Build a Windows API source ZIP with both OCR packages, never user data/models."""
import argparse
from pathlib import Path
import zipfile


def build(output):
    repository = Path(__file__).resolve().parents[3]
    roots = ['projects/windows-fastapi', 'projects/dino-sam-hezar-colab', 'packages/cardid-extractor']
    selected = [repository/'README.md']
    for relative in roots:
        base = repository/relative
        for path in base.rglob('*'):
            rel = path.relative_to(base)
            if not path.is_file() or any(part.startswith('.') for part in rel.parts[:-1]):
                continue
            if len(rel.parts) == 1:
                include = (path.name.startswith(('requirements', 'README')) or path.name in
                           {'pyproject.toml', 'pytest.ini', 'LICENSE', '.env.example', '.gitignore', 'VALIDATION.md'})
            else:
                include = rel.parts[0] in {'app','src','scripts','tests'} and path.suffix in {'.py','.ps1','.md'}
            if include:
                selected.append(path)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path, Path('Persian_OCR_IDCard')/path.relative_to(repository))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/persian-id-windows-api.zip')
    print(build(parser.parse_args().output))
