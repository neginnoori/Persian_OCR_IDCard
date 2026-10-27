"""Run locally; model initialization occurs only for an OCR request."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "kibordevents/iranian-national-id-card-dataset"


def main():
    parser = argparse.ArgumentParser(description="DocAligner + Hezar Persian card OCR")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Local image path")
    source.add_argument("--dataset-id", help="Image ID from the Kaggle dataset")
    source.add_argument("--download-dataset", action="store_true", help="Download/cache Kaggle data only")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument("--models", type=Path, default=ROOT/"models")
    parser.add_argument("--variant", choices=["fastvit_t8", "fastvit_sa24", "lcnet100"], default="fastvit_t8")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.download_dataset or args.dataset_id:
        import kagglehub
        dataset_root = Path(kagglehub.dataset_download(SLUG))
        print("Dataset:", dataset_root)
        if args.download_dataset:
            return
        matches = [p for p in dataset_root.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and p.stem == str(int(args.dataset_id))]
        if len(matches) != 1:
            parser.error(f"Expected one image for ID {args.dataset_id}, found {len(matches)}")
        args.image = matches[0]
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    from .pipeline import CardOCR
    engine = CardOCR(args.models, variant=args.variant, device=args.device)
    result = engine.process(args.image, args.output)
    print(json.dumps({"status": result["status"], "text": result["text"],
                      "output": str(args.output.resolve()), "error": result.get("error")},
                     ensure_ascii=False, indent=2))
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
