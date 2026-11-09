"""Synchronous batch orchestration; reuse the single shared GPU queue."""
import asyncio
import logging
from time import monotonic
from .schemas import BatchItem, BatchResponse, ErrorDetail
from .service import ServiceError

logger = logging.getLogger('card_ocr_api')


async def process_batch(files, service, settings, request_id):
    started = monotonic()
    deadline = started + settings.batch_timeout_seconds
    items = []
    try:
        if not files:
            raise ServiceError(422, 'empty_batch', 'Upload at least one file using the field files.')
        if len(files) > settings.max_batch_files:
            raise ServiceError(413, 'too_many_files', 'Batch exceeds the configured file-count limit.')
        # Starlette has already counted bytes while parsing the bounded request.
        # Validate the entire batch before starting OCR on any item.
        sizes = [file.size for file in files]
        if any(size is None for size in sizes):
            raise ServiceError(400, 'unknown_file_size', 'Unable to determine upload sizes.')
        if sum(sizes) > settings.batch_upload_bytes:
            raise ServiceError(413, 'batch_too_large', 'Combined files exceed the batch byte limit.')
        for index, file in enumerate(files):
            item_id = f'{request_id}:{index}'
            try:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ServiceError(504, 'batch_timeout', 'Batch deadline reached; this item was not started.')
                if file.size > settings.upload_bytes:
                    raise ServiceError(413, 'image_too_large', 'Image exceeds the per-file byte limit.')
                data = await file.read(settings.upload_bytes + 1)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ServiceError(504, 'batch_timeout', 'Batch deadline reached; this item was not started.')
                result = await service.process(data, item_id, timeout=min(settings.request_timeout_seconds, remaining))
                items.append(BatchItem(index=index, item_id=item_id, status='completed', http_status=200, result=result))
            except ServiceError as error:
                items.append(BatchItem(index=index, item_id=item_id, status='error', http_status=error.status,
                                       error=ErrorDetail(code=error.code, message=error.message)))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error('Batch item failed item_id=%s type=%s', item_id, type(error).__name__)
                items.append(BatchItem(index=index, item_id=item_id, status='error', http_status=500,
                                       error=ErrorDetail(code='inference_failed', message='OCR processing failed. Use item_id for support.')))
            finally:
                await file.close()
        completed = sum(item.status == 'completed' for item in items)
        failed = len(items) - completed
        status = 'completed' if not failed else ('failed' if not completed else 'partial_failure')
        return BatchResponse(request_id=request_id, status=status, total=len(items), completed=completed,
                             failed=failed, items=items, elapsed_ms=round((monotonic()-started)*1000,2))
    finally:
        # Also close files not reached after count/aggregate validation or cancellation.
        for file in files:
            await file.close()
