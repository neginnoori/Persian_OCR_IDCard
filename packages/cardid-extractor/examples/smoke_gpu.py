"""Real-weight smoke test. One extractor processes all inputs, including optional grayscale.

python examples/smoke_gpu.py input.jpg --output-dir results --also-gray
Exit 0: all inputs accepted, 2: one or more need review/not found, 1: runtime failure.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
from cardid import CardExtractor, ExtractorConfig, CardIDError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("cardid-results"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--also-gray", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    extractor = CardExtractor(ExtractorConfig(cache_dir=args.cache_dir)).load()
    print(json.dumps(extractor.runtime_info,ensure_ascii=False,indent=2))
    success = True
    for index, path in enumerate(args.images):
        with Image.open(path) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        variants = [("color",photo)]
        if args.also_gray:
            variants.append(("gray",photo.convert("L")))
        for label, photo in variants:
            result = extractor.extract_result(photo)
            stem = f"{index:03d}_{path.stem}_{label}"
            (args.output_dir/f"{stem}.json").write_text(
                json.dumps(result.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
            if result.image is not None:
                suffix = "" if result.ok else "_needs_review"
                result.image.save(args.output_dir/f"{stem}{suffix}.png")
            success = success and result.ok
            print(path.name,label,result.status,result.reason or "",round(result.metadata['elapsed_ms']),"ms")
    return 0 if success else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CardIDError as error:
        raise SystemExit(f"{type(error).__name__}: {error}")
