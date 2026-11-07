# Persian ID Card OCR — Windows FastAPI branch

This branch adds a Windows/NVIDIA API around the corrected card pipeline:
DINO/SAM → card rectification and orientation → Paddle text detection → Hezar CRNN → six fields.

**[راهنمای فارسی نصب و Swagger](projects/windows-fastapi/README.fa.md)**

From a complete checkout/ZIP, open PowerShell:

```powershell
cd projects\windows-fastapi
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

Target: Windows x64, Python 3.11 x64, NVIDIA GPU with a suitable CUDA driver.
After startup, open **http://127.0.0.1:8000/docs**. Copy the generated `OCR_API_KEY`
from the private `.env` into Swagger's Authorize dialog, then upload a JPEG/PNG to
`POST /v1/ocr` using the `file` field.

Dependencies: [main requirements](projects/windows-fastapi/requirements.txt),
[API](projects/windows-fastapi/requirements-api.txt),
[isolated Paddle](projects/windows-fastapi/requirements-paddle.txt).
The installer creates both environments and installs the repository's local OCR packages.

Models load once, work is serialized with a bounded queue, and responses preserve raw
text, six fields, missing values and warnings. `ocr_candidate` is not identity verification.
The service binds to loopback by default. Read the deployment notes before company-network use.

[Validation and remaining limits](projects/windows-fastapi/VALIDATION.md).
API tests use injected model doubles; full Windows GPU inference still requires target-machine validation.
The existing Colab notebook remains in `projects/dino-sam-hezar-colab`.
