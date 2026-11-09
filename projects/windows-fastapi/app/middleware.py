"""Bound raw multipart bytes before parsing; authenticate before accepting uploads."""
import asyncio
import hmac
from uuid import uuid4
from starlette.responses import JSONResponse


def error_response(status, request_id, code, message):
    return JSONResponse({'request_id': request_id, 'error': {'code': code, 'message': message}},
                        status_code=status, headers={'Cache-Control': 'no-store',
                        'X-Request-ID': request_id, **({'Retry-After':'5'} if status == 503 else {})})


class RequestGuard:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.active_uploads = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        scope.setdefault('state', {})['request_id'] = request_id
        async def safe_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = [(k, v) for k, v in message.get('headers', [])
                                      if k.lower() not in (b'cache-control', b'x-request-id')] + [
                    (b'cache-control', b'no-store'), (b'x-request-id', request_id.encode())]
            await send(message)
        path = scope['path'].rstrip('/')
        if path not in ('/v1/ocr', '/v1/ocr/batch'):
            return await self.app(scope, receive, safe_send)
        async def reject(status, code, message):
            await error_response(status, request_id, code, message)(scope, receive, send)
        headers = dict(scope['headers'])
        supplied = headers.get(b'x-api-key', b'')
        expected = self.settings.api_key.get_secret_value().encode('ascii')
        if not hmac.compare_digest(supplied, expected):
            return await reject(401, 'unauthorized', 'A valid X-API-Key is required.')
        if self.active_uploads >= self.settings.max_pending + 1:
            return await reject(503, 'queue_full', 'OCR is busy; retry later.')
        limit = (self.settings.batch_upload_bytes if path == '/v1/ocr/batch'
                 else self.settings.upload_bytes) + 1024 * 1024  # bounded multipart overhead
        try:
            declared = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await reject(400, 'invalid_length', 'Invalid Content-Length.')
        if declared < 0:
            return await reject(400, 'invalid_length', 'Invalid Content-Length.')
        if declared > limit:
            return await reject(413, 'request_too_large', 'Upload exceeds the request size limit.')
        self.active_uploads += 1
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(self.settings.upload_timeout_seconds):
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return
                        chunk = message.get('body', b'')
                        if len(body) + len(chunk) > limit:
                            return await reject(413, 'request_too_large', 'Upload exceeds the request size limit.')
                        body.extend(chunk)
                        if not message.get('more_body', False):
                            break
            except TimeoutError:
                return await reject(408, 'upload_timeout', 'Upload did not finish in time.')
            delivered = False
            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await receive()
            await self.app(scope, bounded_receive, safe_send)
        finally:
            self.active_uploads -= 1
