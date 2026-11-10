# DINO/SAM → PaddleOCR detection → Hezar CRNN → six fields

Version 0.2 adds PP-OCRv5_server_det as the default detector. Hezar loads only the Persian CRNN recognizer. EasyOCR still supplies the existing extraction orientation evidence. Default preprocessing is `rectify_best_effort`: DINO/SAM locate the card, the existing perspective warp straightens it, and EasyOCR/face evidence selects its reading direction. Geometry/orientation uncertainty remains in metadata but does not block Paddle/Hezar when a candidate image exists. If no quadrilateral image is available, an explicit `orientation_only` fallback rotates the full input; it does not remove arbitrary skew. Genuine inference errors propagate. Use `rectify` for the legacy strict gate, or `orientation_only` explicitly to rotate the full photo without rectification.

See [Persian instructions](README.fa.md), [validation](VALIDATION.md), and the included Colab notebook.

Run the Colab notebook with a T4 GPU. The notebook downloads a pinned source commit directly from GitHub. The setup creates separate Torch/Hezar and Paddle environments to avoid conflicting OpenCV providers and mixing CUDA libraries in one process. Paddle detection explicitly defaults to CPU; extraction and Hezar remain on GPU. Optional Paddle gpu:0 uses the official CUDA 12.6 index and has not been tested here. No silent device fallback is performed.

The persistent Paddle worker receives PNG pixels through local pipes and returns JSON polygons/scores. It loads detection only, validates its result shape, preserves oriented-image coordinates, and has startup/request timeouts and explicit shutdown. Call pipeline.close() after use. No external image-upload service is used. Model downloads need network on first run.

TEXT_DETECTOR='craft' retains the earlier detector for comparisons. Default Paddle uses perspective text-line crops with background padding; its boxes are not merged by the CRAFT fragment rule. Polygons and scores are kept in regions metadata.

The API offers result.to_dataframe(), always assigning six named fields for processed/reviewed inputs. In orientation-only mode the field template is normalized to the rotated mask bounding box, so field assignment remains approximate under skew/perspective. Without a card mask it returns empty fields while preserving all raw OCR text. The standard-front-layout field template is heuristic: missing values stay empty, and label/value combined detections may remain unassigned. Do not treat candidate or format validation as correctness. No six-field accuracy benchmark is claimed.

## API
### Stamped scan recovery

The original highest-DINO/geometry candidate remains the baseline. Low or absent
orientation keywords never remove its image. A recovery pass is attempted only for
a small crop with positive company/copy-stamp text and without face/date/value-row
support. It checks at most two additional DINO detections and an aspect-bounded
full-image frame. A replacement must be at least 1.5 times larger and contain a
left-side face plus four aligned right-side rows spanning the card, including a
date. If no stronger proposal exists, the original pixels and transform remain.
An inferred full-image frame is explicitly approximate, not verified geometry.
`metadata.extraction.metadata.stamp_recovery` records the decision. This recovery
is heuristic; it has synthetic regression coverage, not a real-image GPU accuracy
claim. Model errors still propagate. The six-field schema and existing numeric
and name assignment rules are unchanged; unreadable values remain missing.

```python
from persian_id_ocr import PersianCardOCR
engine = PersianCardOCR('./models')
try:
    result = engine.process('card.jpg')
    print(result.text)
    print(result.to_dataframe())
finally:
    engine.close()
```

A custom installation may set PERSIAN_PADDLE_PYTHON to the separate detection environment's Python. Use requirements-colab.txt only in the Torch environment and requirements-paddle.txt only in the Paddle environment. Runtime CPU Paddle 3.2.2 or optional CUDA Paddle 3.2.0 is installed separately by setup_colab.py. The requirements pin direct dependencies, not the entire transitive dependency graph.

## Portable source ZIP
```bash
python scripts/build_zip.py --output dist/dino-sam-hezar-colab.zip
```
The archive includes the local cardid-extractor package, setup scripts and notebook, not model weights or user images.


## Colab: synchronized source and installed code
Open `DINO_SAM_Hezar_Colab.ipynb` from this branch and run cells in order with a T4 runtime. The first cell downloads a pinned source commit directly, without the GitHub API; no project ZIP or manual geometry patch is needed. Installation force-reinstalls both local packages. Before processing, `verify_install.py` compares every source module with the installed file and stops on stale code. The result ZIP records the source commit in `source_version.json`. To update an existing session, rerun source download and installation before processing. GPU inference accuracy must still be checked on real images; source verification is not an OCR accuracy check.
