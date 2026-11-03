# DINO/SAM → PaddleOCR detection → Hezar CRNN → six fields

Version 0.2 adds PP-OCRv5_server_det as the default detector. Hezar loads only the Persian CRNN recognizer. EasyOCR still supplies the existing extraction orientation evidence. Geometry and orientation review gates are unchanged.

See [Persian instructions](README.fa.md), [validation](VALIDATION.md), and the included Colab notebook.

Run the Colab notebook with a T4 GPU. Upload either the complete repository ZIP or the source ZIP produced by scripts/build_zip.py. The setup creates separate Torch/Hezar and Paddle environments to avoid conflicting OpenCV providers and mixing CUDA libraries in one process. Paddle detection explicitly defaults to CPU; extraction and Hezar remain on GPU. Optional Paddle gpu:0 uses the official CUDA 12.6 index and has not been tested here. No silent device fallback is performed.

The persistent Paddle worker receives PNG pixels through local pipes and returns JSON polygons/scores. It loads detection only, validates its result shape, preserves original card coordinates, and has startup/request timeouts and explicit shutdown. Call pipeline.close() after use. No external image-upload service is used. Model downloads need network on first run.

TEXT_DETECTOR='craft' retains the earlier detector for comparisons. Default Paddle uses perspective text-line crops with background padding; its boxes are not merged by the CRAFT fragment rule. Polygons and scores are kept in regions metadata.

The API offers result.to_dataframe(), always assigning six named fields for processed/reviewed inputs. The standard-front-layout field template is heuristic: missing values stay empty, and label/value combined detections may remain unassigned. Do not treat candidate or format validation as correctness. No six-field accuracy benchmark is claimed.

## API
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
Open `DINO_SAM_Hezar_Colab.ipynb` from this branch and run cells in order with a T4 runtime. The first cell resolves the branch to a commit and downloads that exact source; no project ZIP or manual geometry patch is needed. Installation force-reinstalls both local packages. Before processing, `verify_install.py` compares every source module with the installed file and stops on stale code. The result ZIP records the source commit in `source_version.json`. To update an existing session, rerun source download and installation before processing. GPU inference accuracy must still be checked on real images; source verification is not an OCR accuracy check.
