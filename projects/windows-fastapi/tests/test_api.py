import asyncio
from io import BytesIO
import os
from threading import Event
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from PIL import Image
os.environ.setdefault('OCR_API_KEY', 'test-key-' + 'a'*40)
from app.main import create_app
from app.config import Settings
from app.service import OCRService, ServiceError

KEY = 'test-key-'+'a'*40
HEADERS = {'X-API-Key':KEY}


def picture(fmt='PNG', size=(100,60)):
    out=BytesIO()
    Image.new('RGB',size).save(out,format=fmt)
    return out.getvalue()


class Engine:
    def __init__(self): self.calls=0; self.closed=False
    def process(self,image):
        self.calls+=1
        assert image.mode=='RGB'
        return SimpleNamespace(status='ocr_candidate',image=image,text='۰۰۱۲۳۴۵۶۷۸',text_normalized='0012345678',
            fields=[{'field':'national_id','value':'0012345678','text_raw':'۰۰۱۲۳۴۵۶۷۸', 'status':'candidate', 'box':[1,2,30,20]}],
            metadata={'effective_preprocessing':'rectify','coordinate_space':'extracted_card','warnings':['ambiguous_orientation']})
    def close(self): self.closed=True


def settings(**kwargs): return Settings(api_key=KEY,_env_file=None,**kwargs)


@pytest.fixture
def api():
    engine=Engine(); loads=[]
    def factory(config): loads.append(1); return engine
    with TestClient(create_app(settings(),factory)) as client:
        yield client,engine,loads
    assert engine.closed


def test_swagger_health_and_single_model_lifecycle(api):
    client,engine,loads=api
    assert client.get('/docs').status_code==200
    schema=client.get('/openapi.json').json()
    assert schema['paths']['/v1/ocr']['post']['security']==[{'APIKeyHeader':[]}]
    assert 'multipart/form-data' in schema['paths']['/v1/ocr']['post']['requestBody']['content']
    assert client.get('/health/live').json()['status']=='alive'
    assert client.get('/health/ready').json()['status']=='ready'
    for _ in range(2):
        response=client.post('/v1/ocr',headers=HEADERS,files={'file':('a.png',picture(),'image/png')})
        assert response.status_code==200
        result=response.json()
        assert result['fields']['national_id']['value']=='0012345678'
        assert len(result['fields'])==6 and result['fields']['first_name']['value'] is None
        assert result['warnings']==['ambiguous_orientation'] and result['review_required']
        assert response.headers['x-request-id']==result['request_id']
        assert 'no-store' in response.headers['cache-control']
    assert engine.calls==2 and loads==[1]


@pytest.mark.parametrize('headers',[{}, {'X-API-Key':'wrong'}])
def test_auth_before_upload(api,headers):
    client,engine,_=api
    response=client.post('/v1/ocr',headers=headers,content=b'not a multipart upload')
    assert response.status_code==401 and engine.calls==0


@pytest.mark.parametrize('data,expected',[(b'',400),(b'garbage',400),(picture('GIF'),415)])
def test_invalid_images(api,data,expected):
    client,engine,_=api
    response=client.post('/v1/ocr',headers=HEADERS,files={'file':('fake.jpg',data,'image/jpeg')})
    assert response.status_code==expected and engine.calls==0
    assert set(response.json())=={'request_id','error'}


def test_missing_file(api):
    response=api[0].post('/v1/ocr',headers=HEADERS)
    assert response.status_code==422


def test_pixel_and_file_limits():
    with TestClient(create_app(settings(max_pixels=1024,max_upload_mb=1),lambda _:Engine())) as client:
        assert client.post('/v1/ocr',headers=HEADERS,files={'file':('x.png',picture())}).status_code==413
        assert client.post('/v1/ocr',headers=HEADERS,files={'file':('x.png',b'x'*(1024*1024+1))}).status_code==413
        assert client.post('/v1/ocr',headers=HEADERS,content=b'x'*(3*1024*1024)).status_code==413


def test_errors_do_not_expose_image_or_exception_details():
    class Broken(Engine):
        def process(self,image): raise RuntimeError('SECRET IMAGE PATH AND ID')
    with TestClient(create_app(settings(),lambda _:Broken())) as client:
        response=client.post('/v1/ocr',headers=HEADERS,files={'file':('private-id.png',picture())})
        assert response.status_code==500
        assert 'SECRET' not in response.text and 'private-id' not in response.text


def test_no_gpu_factory_failure_does_not_report_ready():
    def broken(config): raise RuntimeError('missing GPU')
    with pytest.raises(RuntimeError,match='missing GPU'):
        with TestClient(create_app(settings(),broken)): pass


def test_timeout_keeps_running_capacity_reserved_and_health_responsive():
    entered=Event(); release=Event()
    class Slow(Engine):
        def process(self,image):
            entered.set(); release.wait(3)
            return super().process(image)
    engine=Slow()
    with TestClient(create_app(settings(max_pending=0,request_timeout_seconds=.05),lambda _:engine)) as client:
        try:
            response=client.post('/v1/ocr',headers=HEADERS,files={'file':('x.png',picture())})
            assert entered.is_set() and response.status_code==504
            assert client.get('/health/live').status_code==200
            second=client.post('/v1/ocr',headers=HEADERS,files={'file':('x.png',picture())})
            assert second.status_code==503
        finally: release.set()
    assert engine.closed and engine.calls==1


def test_queued_timeout_cancels_before_inference():
    async def scenario():
        release=Event()
        class Slow(Engine):
            def process(self,image):
                release.wait(3);return super().process(image)
        engine=Slow(); service=OCRService(settings(max_pending=1,request_timeout_seconds=.05),lambda _:engine)
        await service.start()
        try:
            results=await asyncio.gather(service.process(picture(),'1'),service.process(picture(),'2'),return_exceptions=True)
            assert all(isinstance(r,ServiceError) and r.status==504 for r in results)
            assert service.outstanding==1
        finally:
            release.set();await service.close()
        assert engine.calls==1
    asyncio.run(scenario())


def test_exif_is_applied_before_engine():
    out=BytesIO(); image=Image.new('RGB',(80,40)); exif=image.getexif();exif[274]=6
    image.save(out,format='JPEG',exif=exif)
    class Checked(Engine):
        def process(self,image):
            assert image.size==(40,80)
            return super().process(image)
    with TestClient(create_app(settings(),lambda _:Checked())) as client:
        assert client.post('/v1/ocr',headers=HEADERS,files={'file':('x.jpg',out.getvalue())}).status_code==200


def test_streaming_body_limit_without_content_length():
    async def scenario():
        app=create_app(settings(max_upload_mb=1),lambda _:Engine())
        calls=0; sent=[]
        async def receive():
            nonlocal calls
            calls+=1
            return {'type':'http.request','body':b'x'*1024*1024,'more_body':True}
        async def send(message): sent.append(message)
        scope={'type':'http','method':'POST','path':'/v1/ocr','headers':[(b'x-api-key',KEY.encode())], 'query_string':b'', 'http_version':'1.1','scheme':'http','server':('test',80),'client':('test',1),'root_path':''}
        await app(scope,receive,send)
        assert calls==3
        assert sent[0]['status']==413
    asyncio.run(scenario())


def test_upload_timeout_and_disconnect_release_admission():
    async def scenario():
        app=create_app(settings(max_pending=0,upload_timeout_seconds=.01),lambda _:Engine())
        scope={'type':'http','method':'POST','path':'/v1/ocr','headers':[(b'x-api-key',KEY.encode())], 'query_string':b'', 'http_version':'1.1','scheme':'http','server':('test',80),'client':('test',1),'root_path':''}
        sent=[]
        async def slow(): await asyncio.sleep(1)
        async def send(message): sent.append(message)
        await app(dict(scope),slow,send)
        assert sent[0]['status']==408
        async def disconnected(): return {'type':'http.disconnect'}
        sent.clear()
        await app(dict(scope),disconnected,send)
        assert sent==[]  # no queue_full; previous slot was released
    asyncio.run(scenario())


def test_shutdown_failure_still_joins_executor():
    async def scenario():
        class Broken(Engine):
            def close(self): raise RuntimeError('close failed')
        service=OCRService(settings(),lambda _:Broken())
        await service.start()
        with pytest.raises(RuntimeError,match='close failed'): await service.close()
        with pytest.raises(RuntimeError,match='shutdown'): service.executor.submit(lambda:None)
    asyncio.run(scenario())
