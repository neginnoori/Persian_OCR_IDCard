"""One model owner thread; finite backlog, including timed-out running work."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Lock
from time import perf_counter
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from .schemas import CardFields, FieldValue, OCRResponse


class ServiceError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def create_engine(settings):
    # Import heavy dependencies only in the real worker, never during schema discovery.
    from persian_id_ocr import PersianCardOCR
    engine = PersianCardOCR(cache_dir=settings.model_dir, hezar_device='cuda',
                            detector_backend='paddle', detector_device='cpu',
                            preprocessing='rectify_best_effort')
    try:
        engine.extractor.load()
        engine.reader.load()
    except BaseException:
        engine.close()
        raise
    return engine


def decode_image(data, settings):
    if not data:
        raise ServiceError(400, 'empty_image', 'Upload a nonempty JPEG or PNG image.')
    if len(data) > settings.upload_bytes:
        raise ServiceError(413, 'image_too_large', 'Image exceeds the configured byte limit.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in ('JPEG', 'PNG'):
                    raise ServiceError(415, 'unsupported_image', 'Only JPEG and PNG are supported.')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ServiceError(415, 'animated_image', 'Upload a single-frame image.')
                if source.width * source.height > settings.max_pixels:
                    raise ServiceError(413, 'too_many_pixels', 'Image exceeds the configured pixel limit.')
                source.load()
                return ImageOps.exif_transpose(source).convert('RGB')
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServiceError(413, 'too_many_pixels', 'Image dimensions exceed the safe limit.') from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ServiceError(400, 'invalid_image', 'The uploaded image cannot be decoded.') from None


class OCRService:
    def __init__(self, settings, factory=create_engine):
        self.settings, self.factory = settings, factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='card-ocr')
        self.engine = None
        self.ready = False
        self.outstanding = 0
        self.lock = Lock()

    async def start(self):
        try:
            self.engine = await asyncio.wrap_future(self.executor.submit(self.factory, self.settings))
            self.ready = True
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise

    async def close(self):
        self.ready = False
        # The close operation queues behind any running request, on the same thread.
        try:
            if self.engine is not None:
                await asyncio.wrap_future(self.executor.submit(self.engine.close))
        finally:
            await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)

    def _finished(self, future):
        with self.lock:
            self.outstanding -= 1

    def _process(self, data, request_id):
        started = perf_counter()
        with decode_image(data, self.settings) as image:
            result = self.engine.process(image)
        values = {item['field']: FieldValue.model_validate(item) for item in result.fields}
        metadata = result.metadata
        return OCRResponse(
            request_id=request_id, status=result.status, fields=CardFields(**values),
            text_raw=result.text, text_normalized=result.text_normalized,
            warnings=metadata.get('warnings', []), review_required=True,
            preprocessing=metadata.get('effective_preprocessing', metadata.get('preprocessing_mode', 'unknown')),
            coordinate_space=metadata.get('coordinate_space'),
            processed_image_size=result.image.size if result.image is not None else None,
            processing_ms=round((perf_counter()-started)*1000, 2))

    async def process(self, data, request_id):
        with self.lock:
            if not self.ready:
                raise ServiceError(503, 'not_ready', 'OCR models are not ready.')
            if self.outstanding >= self.settings.max_pending + 1:
                raise ServiceError(503, 'queue_full', 'OCR is busy; retry later.')
            self.outstanding += 1
        try:
            future = self.executor.submit(self._process, data, request_id)
        except BaseException:
            with self.lock:
                self.outstanding -= 1
            raise
        future.add_done_callback(self._finished)
        # Cancellation can stop queued work, not an active CUDA call. Its capacity
        # stays reserved until _finished; never start overlapping model inference.
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.settings.request_timeout_seconds)
        except (TimeoutError, asyncio.CancelledError) as error:
            future.cancel()
            wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            if isinstance(error, asyncio.CancelledError):
                raise
            raise ServiceError(504, 'processing_timeout', 'OCR response timed out; running work may still finish.') from None
