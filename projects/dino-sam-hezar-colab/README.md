# DINO/SAM + Hezar Persian OCR — Colab project

**[راهنمای فارسی](README.fa.md)**

Input photo → Grounding DINO + SAM → perspective-corrected, upright 856×540 RGB card → Hezar CRAFT text detection → Hezar Persian CRNN recognition.

EasyOCR is retained only for the existing extractor's orientation decision. Final Persian text comes from Hezar (`hezarai/CRAFT`, `hezarai/crnn-base-fa-v2`). No DocAligner, Kaggle download, or manual six-field template is used in this project.

## Open in Google Colab

1. Unzip `dino-sam-hezar-colab.zip` on your computer.
2. At https://colab.research.google.com choose **File → Upload notebook** and select `DINO_SAM_Hezar_Colab.ipynb` from the extracted folder.
3. Choose **Runtime → Change runtime type → T4 GPU** (or another NVIDIA GPU).
4. Run the notebook cells in order. The first upload asks for the **original project ZIP**. The later upload asks for your card photos.
5. Wait for installation and the initial model downloads. The notebook displays extracted cards, OCR boxes and text, and downloads a results ZIP. JSON reports distinguish `ocr_candidate`, `no_text`, `needs_review`, `not_found` and `error`.

The notebook runs the code using `/content/persian-id-venv/bin/python` in a subprocess. Colab's kernel remains separate. All photos in one run share model instances; rerunning the processing cell starts a fresh process and reloads models. Uploaded photos are processed in the Colab runtime; model providers receive download requests, not your photos. Colab runtime storage is temporary. Do not share a notebook with retained personal-card outputs.

## Reuse extraction in your own project

Install the delivered local extractor and this package in the same isolated environment (the setup script does this). Neither package needs to be published to PyPI.

```python
from cardid import CardExtractor

extractor = CardExtractor()  # create once per worker/process
card = extractor.extract("photo.jpg")  # upright PIL RGB image
# Pass `card` into any OCR implementation. No automatic image writes.
```

To use Hezar as well:

```python
from persian_id_ocr import PersianCardOCR

pipeline = PersianCardOCR(cache_dir="./models")
result = pipeline.process("photo.jpg")
if result.status == "ocr_candidate":
    print(result.text)              # raw recognized text, in reading order
    print(result.text_normalized)   # digit and Arabic/Persian letter normalization
    card = result.image
    report = result.to_dict()       # JSON-safe, no image pixels
else:
    print(result.status, result.metadata)
```

Reading order groups nearby lines and sorts each line right to left; plain text uses one recognized region per line. Region `box` and padded `crop_box` coordinates refer to the extracted card. The extraction report separately includes homographies and source corners relative to the EXIF-corrected input. No field values are guessed or corrected against a database. OCR candidates always require review.

For an already extracted, upright card:

```python
from persian_id_ocr import HezarReader
reader = HezarReader("./models/hezar", device="cuda")
regions = reader.read(card)
```

`HezarReader` also supports CPU; DINO/SAM extraction requires CUDA. `orientation_override=0/90/180/270` on `pipeline.process()` has the extractor's cyclic-corner meaning, not a raw-photo rotation. It resolves orientation only; it does not accept unreliable geometry. Uncertain extractions skip Hezar and return a review candidate when available.

## Environment / validation

Python 3.11–3.13, Linux, compatible NVIDIA driver for PyTorch CUDA 12.6. Direct versions are in `requirements-colab.txt`; transitive dependencies are resolved at install time. Headless OpenCV is installed once; `hezar` is installed without its vision extra to avoid a second cv2 distribution. Do not install this profile into the repository's original `.venv`.

From a repository checkout: `python projects/dino-sam-hezar-colab/scripts/setup_colab.py`. From the project ZIP: `python scripts/setup_colab.py`. Install `[dev]` and run `python -m pytest` in each package for unit tests. Use `scripts/build_zip.py` from the repository to regenerate the portable ZIP.

Real CUDA/model execution and OCR accuracy require a Colab GPU run. CPU tests with injected evidence validate control flow, input handling, crop batching and output isolation; they do not establish learned-model accuracy. See `VALIDATION.md` for this delivery's executed checks.

## Model sources

- https://huggingface.co/IDEA-Research/grounding-dino-tiny
- https://huggingface.co/facebook/sam-vit-base
- https://github.com/JaidedAI/EasyOCR
- https://huggingface.co/hezarai/CRAFT
- https://huggingface.co/hezarai/crnn-base-fa-v2

Weights are downloaded separately, use upstream default revisions, and are not bundled. Model providers' terms apply separately. The repository MIT license is included in the ZIP.
