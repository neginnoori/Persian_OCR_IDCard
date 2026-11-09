import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from PIL import Image
from cardid import CardExtractor,ExtractorConfig,CardNotFoundError,ReviewRequiredError,GPURequiredError
from cardid.backend import Detection,require_cuda
import cardid.extractor as implementation
from test_geometry import scene


class EvidenceReader:
    """Injected OCR evidence tests orchestration, not recognition accuracy."""
    def __init__(self,reference): self.reference=reference
    def readtext(self,rgb,**kwargs):
        assert kwargs['rotation_info'] is None
        if np.mean(np.abs(rgb.astype(float)-self.reference.astype(float)))<3:
            return [([[300,100],[650,100],[650,130],[300,130]],'نام خانوادگی',.94),
                    ([[300,200],[650,200],[650,230],[300,230]],'تاریخ تولد',.92)]
        return []


def make_extractor(monkeypatch,mask,reader,no_detection=False):
    loads=[];calls=[]
    class Backend:
        def __init__(self,config):
            loads.append(1);self.reader=reader;self.face_detector=None
            self.info={'device':'injected-test-backend'}
        def detect_masks(self,pil):
            assert pil.mode=='RGB'
            calls.append(1)
            return Detection([],[],None) if no_detection else Detection([mask],[.95],.8)
    monkeypatch.setattr(implementation,'TorchBackend',Backend)
    return CardExtractor(),loads,calls


@pytest.mark.parametrize('grayscale',[False,True])
def test_full_orchestration_and_model_reuse(monkeypatch,tmp_path,grayscale):
    card,photo,mask,_=scene()
    rgb=cv2.cvtColor(photo,cv2.COLOR_BGR2RGB)
    reference=cv2.cvtColor(card,cv2.COLOR_BGR2RGB)
    if grayscale:
        rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        reference=cv2.cvtColor(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    extractor,loads,calls=make_extractor(monkeypatch,mask,EvidenceReader(reference))
    assert not extractor.loaded
    result=extractor.extract_result(rgb)
    assert result.ok and result.image.mode=='RGB' and result.image.size==(856,540)
    assert np.mean(np.abs(np.asarray(result.image).astype(float)-reference.astype(float)))<3
    assert json.loads(json.dumps(result.to_dict()))['status']=='ok'
    assert extractor.extract(rgb).size==(856,540)
    extractor.load()
    assert len(loads)==1 and len(calls)==2
    assert list(tmp_path.iterdir())==[]


def test_no_detection_is_explicit(monkeypatch):
    card,photo,mask,_=scene()
    extractor,_,_=make_extractor(monkeypatch,mask,EvidenceReader(card),True)
    result=extractor.extract_result(photo)
    assert result.status=='not_found' and result.image is None
    with pytest.raises(CardNotFoundError): extractor.extract(photo)


def test_review_candidate_is_not_silently_accepted(monkeypatch):
    _,photo,mask,_=scene()
    class Empty:
        def readtext(self,*args,**kwargs): return []
    extractor,_,_=make_extractor(monkeypatch,mask,Empty())
    result=extractor.extract_result(photo)
    assert result.status=='needs_review' and result.image is None
    assert result.reason=='no_supported_card_candidate'
    with pytest.raises(ReviewRequiredError) as error: extractor.extract(photo)
    assert error.value.result.image is None
    assert extractor.extract_result(photo,orientation_override=0).ok


def test_unrecoverable_mask_returns_review_without_crop(monkeypatch):
    _,photo,mask,_=scene()
    extractor,_,_=make_extractor(monkeypatch,np.zeros_like(mask),None)
    result=extractor.extract_result(photo)
    assert result.status=='needs_review' and result.image is None


def test_inference_errors_are_not_no_card(monkeypatch):
    class Broken:
        info={}
        def detect_masks(self,image): raise RuntimeError('test backend failure')
    extractor=CardExtractor();extractor._backend=Broken()
    with pytest.raises(RuntimeError,match='test backend failure'):
        extractor.extract(np.zeros((40,40,3),np.uint8))


def test_gpu_is_required_and_probed():
    with pytest.raises(GPURequiredError):
        require_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)),'cuda')
    def broken(*args,**kwargs): raise RuntimeError('driver error')
    with pytest.raises(GPURequiredError):
        require_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),ones=broken),'cuda')
    events=[]
    class Tensor:
        def sum(self): return self
        def item(self): events.append('computed')
    def probe(*args,**kwargs):
        assert kwargs['device']=='cuda';return Tensor()
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:events.append('sync')),ones=probe)
    require_cuda(torch,'cuda')
    assert events==['computed','sync']


@pytest.mark.parametrize('kwargs',[{'device':'cpu'},{'device':'cuda:1'},{'width':0},
    {'box_threshold':0},{'text_threshold':2},{'min_orientation_margin':float('nan')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): ExtractorConfig(**kwargs)


def test_cli_writes_only_accepted_image(monkeypatch,tmp_path):
    from cardid import cli,ExtractionResult
    state={'result':ExtractionResult('needs_review',image=Image.new('RGB',(40,40)),reason='ambiguous_orientation')}
    class Extractor:
        def __init__(self,*args): pass
        def extract_result(self,*args): return state['result']
    monkeypatch.setattr(cli,'CardExtractor',Extractor)
    output=tmp_path/'card.png';report=tmp_path/'report.json'
    assert cli.main(['input.jpg',str(output),'--report',str(report)])==2
    assert not output.exists() and json.loads(report.read_text())['status']=='needs_review'
    state['result']=ExtractionResult('ok',image=Image.new('RGB',(40,40)))
    assert cli.main(['input.jpg',str(output)])==0
    assert output.exists()



def test_stamp_candidate_is_rejected_before_field_assignment(monkeypatch):
    _, photo, mask, _ = scene()
    class StampReader:
        def readtext(self, *args, **kwargs):
            return [([[300,100],[650,100],[650,130],[300,130]], 'شرکت کپی برابر اصل', .99)]
    extractor, _, _ = make_extractor(monkeypatch, mask, StampReader())
    result = extractor.extract_result(photo)
    assert result.image is None
    assert result.reason == 'no_supported_card_candidate'
    assert not result.metadata['card_evidence_ok']


def test_card_evidence_beats_stamp_geometry(monkeypatch):
    _, photo, mask, _ = scene()
    extractor, _, _ = make_extractor(monkeypatch, mask, None)
    extractor.load()
    extractor._backend.detect_masks = lambda pil: Detection([mask, mask], [.99, .70], .9)
    count = iter([0, 100])
    def quad(*args, **kwargs):
        offset = next(count)
        return np.array([[0,0],[800,0],[800,500],[0,500]], dtype=np.float32)+offset, {'geometry_ok':True,'mask_quad_iou':.99}
    monkeypatch.setattr(implementation, 'quad_from_mask', quad)
    def orientation(candidates,*args,**kwargs):
        return candidates[0], {'orientation_ok':True}
    monkeypatch.setattr(implementation,'choose_orientation',orientation)
    evidence = iter([
        {'score':4.,'matched_keywords':[],'face_on_left':False},
        {'score':9.,'matched_keywords':['تولد','ملی'],'face_on_left':False},
    ])
    monkeypatch.setattr(implementation,'score_orientation',lambda *args:next(evidence))
    result = extractor.extract_result(photo)
    assert result.ok
    assert result.metadata['sam_predicted_iou'] == .70
    assert len(result.metadata['candidate_selection']) == 2
