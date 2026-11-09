# Validation

- Python 3.11 API contract suite: 25 passed on macOS, using injected OCR engines.
- Combined OCR regression suite: 44 passed, including startup-failure process cleanup.
- Pinned API dependencies resolve to Windows x64/Python 3.11 distributions.
- Swagger HTML and OpenAPI multipart/API-key schemas checked through TestClient.
- Upload byte/pixel limits, bad images, EXIF, authentication, shutdown, bounded work,
  slow-upload timeout and queued/running inference timeouts checked.
- Windows PowerShell execution and real DINO/SAM/Paddle/Hezar GPU inference have not
  been run here. CI runs lightweight API tests on Windows; it does not test CUDA models.
- Requirements pin direct dependencies; they are not a complete transitive lockfile.
- Starlette emits an httpx TestClient deprecation warning; current tests pass.

Before company deployment: install on the target Windows GPU, wait for startup,
check /health/ready, submit representative card images through /docs, and measure
latency/VRAM and correctness of all six fields. Swagger's JS/CSS use the default
FastAPI CDN assets; browser internet access is needed unless those assets are self-hosted.

Requests are synchronous. A 504 response does not interrupt an already-running GPU call;
the slot stays occupied until it ends. The queue is in-process, not durable. Run one server
worker and do not use reload. API errors do not expose exception details or uploaded filenames.

Batch regressions cover mixed/all failures, preserved order with duplicate filenames, shared serialized inference, authentication and Swagger list schema, count/aggregate/per-file limits, chunked upload limits, deadline partial results, cancellation and file cleanup.
