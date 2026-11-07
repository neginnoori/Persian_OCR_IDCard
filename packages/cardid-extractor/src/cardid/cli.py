"""Optional CLI: writing an image is always an explicit caller action."""
import argparse
import json
import sys
from pathlib import Path
from . import CardExtractor, ExtractorConfig
from .errors import CardIDError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract an upright ID card using CUDA GPU 0")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="PNG or JPEG destination; uncertain crops are not written")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON diagnostics")
    args = parser.parse_args(argv)
    try:
        extractor = CardExtractor(ExtractorConfig(cache_dir=args.cache_dir, offline=args.offline))
        result = extractor.extract_result(args.input)
        report = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
        if not result.ok:
            print(report, file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(args.output)
        print(str(args.output))
        return 0
    except (CardIDError, OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
