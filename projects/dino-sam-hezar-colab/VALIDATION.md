# Paddle Detection integration validation — 0.2

Executed on macOS ARM64 / Python 3.11 using the existing main test environment and a new separate Paddle CPU environment.

## Passed
- 39 unchanged extraction tests and 27 combined/detection tests. These include protocol-level real subprocess tests, malformed Paddle outputs, BGR input conversion, coordinate preservation, perspective crops, empty results, CRNN-only Hezar loading, worker reuse/error handling, six-field assignment, and existing orchestration regressions. Injected model evidence is used in unit tests.
- Isolated pinned Paddle environment installed successfully and pip check passed: Paddle CPU 3.2.2, PaddleOCR 3.3.2, PaddleX 3.3.13, NumPy 2.2.6, Pillow 11.3.0, OpenCV contrib 4.10.0.84, huggingface-hub 0.36.2.
- Real official PP-OCRv5_server_det weights and real Hezar CRNN ran on two already-upright user images on CPU. The persistent worker was reused across images; a white image returned no text detections. Aggregate results are in validation-paddle.json; no user image/text is included in that report.
- First sample: 16 text regions, six nonempty template fields, one ten-digit numeric region. Second sample: 10 regions, three nonempty fields; some lines combine labels and values. Nonempty does NOT mean correct. No field/character accuracy claim is made.
- All Python files and notebook cells parsed and notebook schema validated.

## Not executed
- DINO/SAM extraction and orientation with actual CUDA models in this environment. The first-stage code is unchanged.
- A full Colab install from a fresh GPU runtime.
- Optional Paddle GPU 3.2.0 CUDA 12.6 installation/inference. Its official index hostname could not be resolved from the validation host. The documented default uses explicitly selected Paddle CPU 3.2.2 from PyPI, not a silent fallback.
- An accuracy/performance comparison against CRAFT on a held-out dataset. CPU smoke timings include startup on the first sample and must not be compared as inference benchmarks.

## Operational details
PaddleX requires its OpenCV contrib provider, whereas EasyOCR requires headless OpenCV. Separate virtual environments and a persistent JSON-lines worker prevent installing both providers in one environment. Paddle GPU and Torch CUDA libraries are in different processes; VRAM is still shared. Main-process Hezar only loads CRNN in the Paddle path. CRAFT remains an explicit comparison option. Worker shutdown is available through pipeline.close().

The six-field template is a development heuristic for the standard Iranian front-side layout. Missing fields remain empty; no known answer is used to repair text. Tight cropping and ambiguous extraction policies remain separate from OCR confidence.

## Geometry candidate-selection regression (2026-11-03)
A supplied Colab report showed a refined corner 2.75 pixels beyond the last image row despite good IoU/coverage and clear mask borders. Selection now ranks candidates passing all existing geometry gates before numeric IoU. No corners are clamped and no quality thresholds are relaxed. If none pass, review remains required. Tests cover both valid in-frame alternatives and inadequate alternatives. 41 extraction + 27 combined tests passed.

Real Paddle/Hezar CPU OCR also ran on the supplied already-rectified card 948 image: 13 text regions and five nonempty template fields. No ten-digit-only numeric region was produced. This is not a successful six-field accuracy result. The exact original SAM mask was not supplied and the original CUDA extraction was not rerun locally; the end-to-end image 948 result must be retried in Colab.

## Bounded numerical edge-fit adjustment (2026-11-03)
A follow-up report confirmed no candidate passed the original bounds gate. For edge-intersection fits only, when the SAM component does not touch the image boundary, allow projection to valid pixel bounds if the maximum corner movement is no more than min(4, max(1, 0.001 * min(image dimensions))) pixels. Recompute mask IoU and coverage after projection and apply all existing quality gates. Raw and bounded proposals remain distinct, with explicit adjustment and per-candidate diagnostics. Large movement and genuinely border-touching masks cannot use this path.

42 extraction and 27 combined tests passed; original CUDA inference/SAM mask for card 948 remains unavailable locally. This is a tested numerical policy change, not an end-to-end GPU accuracy claim.


## Orientation-only processing
42 extractor tests and 34 combined-pipeline tests pass locally (76 total). New integration tests exercise all four exact pixel rotations and coordinate transforms with a border-touching mask, explicitly forbid quadrilateral fitting and perspective warping, and confirm that an ambiguous direction / missing mask still reaches the text reader. A frame-offset test checks six-field coordinate mapping. Model backends are injected in these tests: the complete DINO/SAM + EasyOCR + Paddle + Hezar path has not been validated on a T4 here. Arbitrary deskew, perspective correction and six-field accuracy are not guaranteed by rotation-only processing.


## Rectification regression (2026-11-04)
The default is now rectify_best_effort. Four injected-backend tests project a synthetic card into a diagonal perspective scene, rotate the scene through all four quarter-turns, and assert that the OCR reader receives an upright 856x540 card (mean pixel error <3), with field assignment in card coordinates. Further tests cover nonblocking geometry/orientation warnings, explicit missing-quad fallback, and propagated inference errors. These test real geometry/orchestration with injected model evidence, not live DINO/SAM or recognition accuracy on 958.jpg. T4 end-to-end validation remains required.
