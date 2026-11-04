"""Command-line bridge also usable from any Colab kernel via subprocess."""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import json
import os
import sys

from .config import OCRConfig, QualityConfig
from .pipeline import PersianOCR


@contextmanager
def model_output_to_stderr():
    """Keep Python and native-library progress off the JSON stdout channel."""
    saved_fd = None
    stdout_fd = None
    try:
        try:
            stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            pass  # StringIO/notebook capture: Python redirection is still applicable.
        else:
            sys.stdout.flush()
            saved_fd = os.dup(stdout_fd)
            os.dup2(stderr_fd, stdout_fd)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        if saved_fd is not None:
            sys.stderr.flush()
            os.dup2(saved_fd, stdout_fd)
            os.close(saved_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paddle detection + Hezar Persian card OCR")
    parser.add_argument("image")
    parser.add_argument("--raw-photo", action="store_true", help="Extract card with cardid-extractor first")
    parser.add_argument("--quality", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--json", dest="json_path", help="Write a data-only JSON")
    parser.add_argument("--report", help="Write HTML, annotated images, data and diagnostics")
    parser.add_argument("--config", help="Optional configuration JSON; see examples/config.json")
    parser.add_argument("--detailed", action="store_true", help="Detailed JSON on stdout")
    args = parser.parse_args(argv)
    try:
        config = OCRConfig(recognizer_device=args.device, quality=QualityConfig(mode=args.quality))
        if args.config:
            from pathlib import Path
            from .config import FieldSpec
            raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
            changes = {k: v for k, v in raw.items() if k not in ("quality", "fields")}
            if "quality" in raw:
                changes["quality"] = QualityConfig(**raw["quality"])
            if "fields" in raw:
                changes["fields"] = tuple(FieldSpec(**f) for f in raw["fields"])
            config = replace(config, **changes)
        # Native libraries may print progress; stdout remains exactly one JSON document.
        with model_output_to_stderr():
            engine = PersianOCR(config)
            result = engine.read_photo(args.image) if args.raw_photo else engine.read(args.image)
            if args.json_path:
                result.save_json(args.json_path)
            if args.report:
                result.save_report(args.report)
        print(json.dumps(result.to_dict(detailed=args.detailed), ensure_ascii=False, allow_nan=False))
        return 0 if result.status != "error" else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
