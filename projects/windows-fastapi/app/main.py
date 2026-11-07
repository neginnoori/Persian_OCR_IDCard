import logging
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, File, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from .config import Settings
from .middleware import RequestGuard, error_response
from .schemas import OCRResponse, ErrorResponse
from .service import OCRService, ServiceError, create_engine

logger = logging.getLogger('card_ocr_api')
api_key = APIKeyHeader(name='X-API-Key', auto_error=False)


def create_app(settings=None, engine_factory=create_engine):
    settings = settings or Settings()
    service = OCRService(settings, engine_factory)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title='Persian ID Card OCR', version='1.0.0', lifespan=lifespan,
                  description='Upload a JPEG/PNG card image. Rectification → Paddle detection → Hezar recognition → six fields. OCR candidates require review; this API does not verify identity.')
    app.state.ocr = service
    app.add_middleware(RequestGuard, settings=settings)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return error_response(error.status, request.state.request_id, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Do not echo user-supplied content or filenames in validation details.
        return error_response(422, request.state.request_id, 'invalid_request', 'Send one image in the multipart field named file.')

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return error_response(error.status_code, request.state.request_id, 'http_error', 'The request could not be accepted.')

    @app.get('/health/live', tags=['Health'])
    async def live():
        return {'status': 'alive'}

    @app.get('/health/ready', tags=['Health'], responses={503:{'model':ErrorResponse}})
    async def ready():
        if not service.ready:
            raise ServiceError(503, 'not_ready', 'OCR models are not ready.')
        return {'status': 'ready'}

    @app.post('/v1/ocr', tags=['OCR'], response_model=OCRResponse,
              responses={code:{'model':ErrorResponse} for code in (400,401,408,413,415,422,500,503,504)})
    async def ocr(request: Request, file: Annotated[UploadFile, File(description='One JPEG or PNG image, up to the configured size limit.')],
                  key: Annotated[str | None, Security(api_key)] = None):
        # RequestGuard validates the key before multipart parsing. Security() also
        # exposes the Authorize button and API-key scheme in Swagger/OpenAPI.
        try:
            data = await file.read(settings.upload_bytes + 1)
        finally:
            await file.close()
        try:
            return await service.process(data, request.state.request_id)
        except ServiceError:
            raise
        except Exception as error:
            # No exception messages, image contents, names, IDs, or paths in logs.
            logger.error('OCR failed request_id=%s type=%s', request.state.request_id, type(error).__name__)
            raise ServiceError(500, 'inference_failed', 'OCR processing failed. Use request_id for support.') from None

    return app


# Uvicorn imports this module after --env-file has loaded configuration.
app = create_app()
