"""Build a portable source ZIP with the local extractor, without weights or data."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist/dino-sam-hezar-colab.zip'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    repository = project.parents[1]
    vendor = project/'vendor/cardid-extractor'
    if not vendor.is_dir():
        vendor = repository/'packages/cardid-extractor'
    license_path = project/'LICENSE'
    if not license_path.is_file():
        license_path = repository/'LICENSE'
    if not (vendor/'pyproject.toml').is_file():
        parser.error('Local cardid-extractor source is missing')
    ignored = {'__pycache__', '.pytest_cache', '.venv', 'build', 'dist', 'vendor', '.git'}
    def files(root):
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if any(p in ignored or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.is_file() and not path.is_symlink() and path.suffix not in ('.pyc', '.zip'):
                yield path, relative
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files(project):
            archive.write(path, str(Path('dino-sam-hezar-colab')/relative))
        for path, relative in files(vendor):
            archive.write(path, str(Path('dino-sam-hezar-colab/vendor/cardid-extractor')/relative))
        if not (project/'LICENSE').is_file():
            archive.write(license_path, 'dino-sam-hezar-colab/LICENSE')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert 'dino-sam-hezar-colab/DINO_SAM_Hezar_Colab.ipynb' in names
    print(args.output)


if __name__ == '__main__':
    main()
