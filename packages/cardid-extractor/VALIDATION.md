# Release validation: 0.1.0

Executed on Windows with Python 3.12.14, NumPy 2.2.6, Pillow 11.3.0 and
opencv-python-headless 4.12.0.88.

## Passed

- **34 pytest cases** covering image normalization (path/bytes/PIL/uint8 arrays),
  grayscale/alpha/BGR, EXIF rotation, invalid input rejection, real projective geometry at
  four rotations, inverse coordinate mapping, manual orientation override, review/no-card
  behavior, repeated-call model reuse, CLI output gating and CUDA failure/success control flow.
- Pipeline tests execute the production orchestrator with **injected model/OCR evidence**.
  They do not download weights and do not evaluate learned-model accuracy.
- Dependency resolver dry run for the pinned profile on Windows/Python 3.12 completed.
- A second cross-target dry run found compatible Linux x86_64/Python 3.12 wheels for the
  pinned profile. This checks wheel availability/dependency metadata from a Windows host;
  platform environment markers still use the host and it is **not** a Linux/CUDA installation test.
- All shipped Python files and notebook code cells parse successfully.

## Release artifact checks

The release includes a standard wheel, source distribution and complete source ZIP.
The build script checks archive integrity and installs the wheel into an isolated target
directory without heavyweight dependencies, then imports the installed package using the
tested image dependencies. See the delivered release-checks.json for results.

## Not executed here

- Full real-weight DINO/SAM/EasyOCR inference on a CUDA GPU.
- The setup notebook inside a live Colab runtime, including CUDA wheel installation.
- Accuracy/performance evaluation on the 2000-image dataset or a real grayscale cohort.

The package preserves the v04 notebook model IDs, geometric method, orientation scoring and
default thresholds that the user reported worked well in Colab. Package-level GPU verification
and real-image checks are available via `examples/smoke_gpu.py` and the included Colab notebook.
Model weights are not included; they download on first load and can be cached for offline use.
