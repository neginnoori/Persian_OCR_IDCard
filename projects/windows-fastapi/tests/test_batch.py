import asyncio
from io import BytesIO
from threading import Event
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from app.batch import process_batch
from app.service import ServiceError
from test_api import Engine, HEADERS, KEY, picture, settings, create_app


def uploads(*images):
    # Duplicate filenames must not overwrite each other's results.
    return [('files', ('same.png', data, 'image/png')) for data in images]


def test_success_and_mixed_failure_preserve_order_and_single_api():
    engine=Engine()
    with TestClient(create_app(settings(), lambda _:engine)) as client:
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(picture(),b'bad',picture()))
        assert response.status_code==200
        result=response.json()
        assert (result['status'],result['completed'],result['failed'])==('partial_failure',2,1)
        assert [item['index'] for item in result['items']]==[0,1,2]
        assert [item['http_status'] for item in result['items']]==[200,400,200]
        assert len({item['item_id'] for item in result['items']})==3
        for item in (result['items'][0],result['items'][2]):
            assert item['result']['request_id']==item['item_id']
            assert len(item['result']['fields'])==6
            assert item['result']['fields']['national_id']['value']=='0012345678'
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(picture(),picture()))
        assert response.json()['status']=='completed'
        assert client.post('/v1/ocr',headers=HEADERS,files={'file':('one.png',picture())}).status_code==200
    assert engine.calls==5 and engine.closed


def test_all_fail_and_model_exception_do_not_abort_later_items():
    class FailOnce(Engine):
        failures=0
        def process(self,image):
            self.failures+=1
            if self.failures==1: raise RuntimeError('PRIVATE DATA')
            return super().process(image)
    with TestClient(create_app(settings(),lambda _:FailOnce())) as client:
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(picture(),picture()))
        assert [i['http_status'] for i in response.json()['items']]==[500,200]
        assert 'PRIVATE DATA' not in response.text
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(b'',b'bad'))
        assert response.json()['status']=='failed' and response.json()['failed']==2


def test_auth_missing_input_and_openapi():
    engine=Engine()
    with TestClient(create_app(settings(),lambda _:engine)) as client:
        assert client.post('/v1/ocr/batch',files=uploads(picture())).status_code==401
        response=client.post('/v1/ocr/batch',headers=HEADERS)
        assert response.status_code==422 and 'files' in response.json()['error']['message']
        doc=client.get('/openapi.json').json()
        endpoint=doc['paths']['/v1/ocr/batch']['post']
        assert endpoint['security']==[{'APIKeyHeader':[]}]
        ref=endpoint['requestBody']['content']['multipart/form-data']['schema']['$ref'].split('/')[-1]
        assert doc['components']['schemas'][ref]['properties']['files']['type']=='array'
        assert engine.calls==0


def test_batch_count_and_aggregate_limits_before_any_inference():
    engine=Engine()
    with TestClient(create_app(settings(max_batch_files=2,max_batch_upload_mb=1),lambda _:engine)) as client:
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(picture(),picture(),picture()))
        assert response.status_code==413 and response.json()['error']['code']=='too_many_files'
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(b'x'*600000,b'x'*600000))
        assert response.status_code==413 and response.json()['error']['code']=='batch_too_large'
        assert engine.calls==0


def test_per_file_limit_is_independent():
    with TestClient(create_app(settings(max_upload_mb=1,max_batch_upload_mb=3),lambda _:Engine())) as client:
        response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(b'x'*(1024*1024+1),picture()))
        assert response.status_code==200
        assert [i['http_status'] for i in response.json()['items']]==[413,200]


def test_batch_deadline_retains_results_and_stops_remaining_work():
    entered=Event();release=Event()
    class SlowSecond(Engine):
        def process(self,image):
            if self.calls==1:
                entered.set();release.wait(3)
            return super().process(image)
    engine=SlowSecond()
    with TestClient(create_app(settings(batch_timeout_seconds=.1,request_timeout_seconds=2,max_pending=0),lambda _:engine)) as client:
        try:
            response=client.post('/v1/ocr/batch',headers=HEADERS,files=uploads(picture(),picture(),picture()))
            assert response.status_code==200 and entered.is_set()
            items=response.json()['items']
            assert [i['http_status'] for i in items]==[200,504,504]
            assert items[2]['error']['code']=='batch_timeout'
            # Still-running GPU work occupies capacity even after batch response.
            assert client.post('/v1/ocr',headers=HEADERS,files={'file':('one.png',picture())}).status_code==503
            assert client.get('/health/live').status_code==200
        finally:release.set()
    assert engine.calls==2


def test_batch_and_single_share_one_nonoverlapping_queue():
    async def scenario():
        from app.service import OCRService
        from threading import Lock
        import time
        class Checked(Engine):
            active=0;peak=0
            lock=Lock()
            def process(self,image):
                with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
                try:
                    time.sleep(.01)
                    return super().process(image)
                finally:
                    with self.lock:self.active-=1
        engine=Checked();config=settings();service=OCRService(config,lambda _:engine)
        await service.start()
        files=[UploadFile(BytesIO(picture()),size=len(picture())) for _ in range(3)]
        try:
            batch,single=await asyncio.gather(process_batch(files,service,config,'batch'), service.process(picture(),'single'))
            assert batch.completed==3 and single.status=='ocr_candidate'
            assert engine.calls==4 and engine.peak==1
            assert all(file.file.closed for file in files)
        finally:await service.close()
    asyncio.run(scenario())


def test_batch_closes_all_files_on_invalid_count_and_cancellation():
    async def scenario():
        files=[UploadFile(BytesIO(b'x'),size=1) for _ in range(2)]
        with pytest.raises(ServiceError):await process_batch(files,None,settings(max_batch_files=1),'bad')
        assert all(f.file.closed for f in files)
        class Cancelled:
            async def process(self,*args,**kwargs):raise asyncio.CancelledError()
        files=[UploadFile(BytesIO(picture()),size=len(picture())) for _ in range(2)]
        with pytest.raises(asyncio.CancelledError):await process_batch(files,Cancelled(),settings(),'cancel')
        assert all(f.file.closed for f in files)
    asyncio.run(scenario())


def test_batch_chunked_body_limit_cannot_be_bypassed():
    async def scenario():
        app=create_app(settings(max_batch_upload_mb=1),lambda _:Engine())
        sent=[];calls=0
        async def receive():
            nonlocal calls
            calls+=1
            return {'type':'http.request','body':b'x'*1024*1024,'more_body':True}
        async def send(message):sent.append(message)
        scope={'type':'http','method':'POST','path':'/v1/ocr/batch','headers':[(b'x-api-key',KEY.encode())], 'query_string':b'', 'http_version':'1.1','scheme':'http','server':('test',80),'client':('test',1),'root_path':''}
        await app(scope,receive,send)
        assert calls==3 and sent[0]['status']==413
    asyncio.run(scenario())
