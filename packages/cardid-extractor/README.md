# CardID Extractor

Reusable Python package extracted from the confirmed CardID v04 notebook pipeline.
Input: a photo. Output: the extracted **856 x 540 RGB card image**, with perspective and orientation corrected.
No final field OCR, field boxes, YOLO, plotting, CSV generation, or automatic image writes.

**[راهنمای فارسی](README.fa.md)**

## Quick use

```python
from cardid import CardExtractor

extractor = CardExtractor()  # Create once per process; weights load on first use.
card = extractor.extract("photo.jpg")
card.save("card.png")       # Optional; the package itself does not save images.
# result = your_persian_ocr(card)
```

`card` is a `PIL.Image.Image` in RGB mode. For an OCR function expecting an RGB NumPy array,
use `numpy.asarray(card)`. OpenCV arrays require `extractor.extract(bgr, color_order="BGR")`.
Paths, encoded image bytes, PIL images, and uint8 NumPy arrays are accepted. Grayscale input is
replicated to three identical channels. Arrays must be HxW or HxWx1/3/4.

## Install in an isolated environment

Python **3.10-3.13**, NVIDIA CUDA GPU 0. A CPU fallback is deliberately disabled.
Install the matching CUDA build of PyTorch/torchvision first, then this package:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-colab.txt .
python -m pip check
```

The CUDA 12.6 wheels need a compatible NVIDIA driver. Other matching CUDA-enabled
torch/torchvision pairs within pyproject.toml's ranges may be used, but are not the pinned profile.
The wheel can also be installed directly with `python -m pip install cardid_extractor-0.1.0-py3-none-any.whl`.
PyPI publication is not required and has not been performed.

### Google Colab

Open `examples/colab_gpu.ipynb`, select a GPU runtime, and follow the cells.
It installs this package in `/content/cardid-venv`, then calls the actual package from
that environment. The notebook's own Python kernel remains separate. No Gradio,
Diffusers or global huggingface-hub downgrade is needed.

`scripts/setup_colab.py` checks `pip check` and CUDA. Dependencies/weights need network
on initial installation. `requirements-colab.txt` pins direct dependencies; it is not a full
transitive lockfile. The release's dependency manifest records the resolved metadata profile.

## Rich result / failures

```python
from cardid import CardExtractor, ExtractorConfig

extractor = CardExtractor(ExtractorConfig(cache_dir="./models", offline=False))
extractor.load()  # Optional startup warmup; model instances are reused.
print(extractor.runtime_info)  # Includes the device for every heavy model.
result = extractor.extract_result(uploaded_image_bytes)
if result.ok:
    card = result.image
else:
    print(result.status, result.reason)
    # result.image can contain a candidate for human review, or None.
```

- `extract()` returns the accepted card image. `extract_card(image)` offers a convenience function
  reusing one default extractor per process.
- `extract_result()` returns `ExtractionResult`: `status` is `ok`, `needs_review`, or `not_found`.
- `extract()` raises `CardNotFoundError` or `ReviewRequiredError` for unsuccessful extraction.
  The latter carries `.result` for optional diagnostics/review.
- Invalid images, absent/unusable CUDA, model downloads and inference errors have explicit
  `CardIDError` subclasses. Runtime failures are never represented as successful extraction.
- No input image or final OCR result is uploaded. Models/weights download to the configured cache;
  the model providers receive normal download requests. First-call latency includes model loading.
- `offline=True` disables weight downloads after a successful cache population. For a fully
  offline deployment, dependencies must also be installed in advance.

## Pipeline / scope

Grounding DINO Tiny → SAM ViT Base → four real card edges / homography → four cyclic
corner assignments → Persian EasyOCR evidence + upright face location → final card image.
EasyOCR is used **only to choose orientation**, not to return field text. OpenCV geometry and
face detection execute on CPU; DINO, SAM and EasyOCR execute on CUDA and their devices are checked.

The layout is the **front of an Iranian national ID card**, matching the notebook. The scorer uses
fixed Persian labels and expects a portrait on the left. It is not a generic document/back-side extractor.
If multiple cards exist, the highest-scoring DINO detection is processed, just as in the notebook.
There is no guarantee of equal accuracy on grayscale photos; run the included real GPU smoke test
on your inputs (including `--also-gray`). No extra grayscale conversion is applied to color inputs.

Geometry/orientation acceptance uses heuristic thresholds, not calibrated correctness probabilities.
An `ok` status does not replace dataset-level evaluation. `orientation_override=0/90/180/270` is
available on `extract`/`extract_result`; it has the notebook's cyclic-corner meaning relative to the
base warp, not the raw camera angle, and never overrides bad geometry.

Homographies and corners in metadata refer to EXIF-corrected input pixels. Returned image
orientation is already applied. `result.to_dict()` is JSON-safe and omits image pixels.

## Service integration

Create one extractor per worker/process and optionally call `.load()` at startup. Call `.extract()`
only when your application's flow needs it. Its lock serializes calls on that instance; multiple
processes each load their own models and consume their own GPU memory. Do not fork a CUDA-loaded
process. This package does not start a web server. See `examples/use_in_service.py`.

## CLI and validation

```bash
cardid-extract photo.jpg card.png --report report.json
python examples/smoke_gpu.py photo.jpg --also-gray --output-dir results
python -m pip install -e ".[dev]"
python -m pytest
```

CLI exit codes: 0 accepted/saved, 2 no card or review required (no crop saved), 1 runtime/input failure.
The smoke example explicitly saves review candidates with a `_needs_review` suffix for inspection.
Tests exercise geometry, image input handling, GPU failure paths, model reuse and pipeline
orchestration using injected model evidence. They do **not** measure recognition accuracy.
See `VALIDATION.md` for exactly which checks were run for this release.

See `THIRD_PARTY.md` for external model and dependency references. Model weights are downloaded
separately and are not included in the wheel or source archive.
