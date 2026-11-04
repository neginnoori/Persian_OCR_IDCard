# Validation — 0.1.1

Environment used for local validation: Windows, Python 3.12.9; core image operations use NumPy, Pillow and OpenCV. Test model backends are deterministic substitutes; tests never download weights.

24 tests cover:

- status gating for not_found / no_card_found / no_photos_found; default processing of review images, retained warnings, explicit strict opt-out, and review results without pixels;
- direct PIL, bytes, file-like, path, NumPy RGB/BGR input;
- raw-photo extractor integration without importing cardid for direct-card use;
- original-only processing for good images, gated correction for dark images;
- keeping portrait darkness out of field quality decisions;
- no automatic output writes; original pixels remain unchanged;
- same crop geometry across original/corrected variants and corrected-detection fallback;
- ambiguous/shared box rejection and visible invalid configuration/selection errors;
- conflict handling, explicit variant selection without re-inference;
- Unicode JSON, zero-prefixed numeric strings, portable HTML, true detector overlays;
- lazy imports and adapter contracts (Paddle BGR input, Hezar PIL input, one model load);
- CLI data/error JSON contract.

The Colab example is validated as notebook JSON and its code cells compile. Wheel installation is smoke-tested separately. The model dependency profile is based on the notebook versions and official APIs; a full installation/inference of both real models in every target environment has **not** been tested here. Linux/Colab execution, GPU behavior and dataset accuracy still require integration testing on the user's runtime. No accuracy percentage is claimed.

Run from the package root after installing core dependencies:

```text
python -m unittest discover -s tests -v
```

For an actual-model smoke test after installing `[models]`, use a known aligned card with ground truth:

```text
persian-ocr aligned_card.jpg --quality auto --json actual/data.json --report actual/report
```

Inspect field crops and exact field values against ground truth. A passed format check or matching original/corrected readings is not an OCR accuracy measurement.
