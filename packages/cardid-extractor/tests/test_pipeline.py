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
        if np.mean(np.abs(rgb.astype(float)-self.reference.astype(float)))<2:
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
    photo = np.zeros_like(photo)
    result=extractor.extract_result(photo)
    assert result.status=='not_found' and result.image is None
    with pytest.raises(CardNotFoundError): extractor.extract(photo)


@pytest.mark.parametrize("include_card_mask", [False, True])
def test_stamp_detection_falls_through_to_full_card(monkeypatch, include_card_mask):
    card, photo, card_mask, _ = scene()
    stamp_mask = np.zeros_like(card_mask)
    cv2.rectangle(stamp_mask, (30, 790), (220, 846), 1, -1)
    photo[stamp_mask.astype(bool)] = 130
    masks = [stamp_mask, card_mask] if include_card_mask else [stamp_mask]
    scores = [.98, .9] if include_card_mask else [.98]

    class Backend:
        def __init__(self, config):
            self.reader = EvidenceReader(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
            self.face_detector = None
            self.info = {}

        def detect_masks(self, pil):
            return Detection(masks, scores, .95, [.95, .5] if include_card_mask else [.95])

    monkeypatch.setattr(implementation, "TorchBackend", Backend)
    result = CardExtractor().extract_result(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    assert result.ok
    assert result.metadata["proposal_source"] == ("dino_sam" if include_card_mask else "card_edges")
    assert result.metadata["dino_score"] == (.5 if include_card_mask else None)


def test_edge_crop_replaces_small_stamp_for_review(monkeypatch):
    card, photo, _, _ = scene()
    reference = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY).astype(float)
    stamp_mask = np.zeros(photo.shape[:2], np.uint8)
    cv2.rectangle(stamp_mask, (30, 790), (220, 846), 1, -1)
    photo[stamp_mask.astype(bool)] = 130
    reader = SimpleNamespace(readtext=lambda *args, **kwargs: [])
    extractor, _, _ = make_extractor(monkeypatch, stamp_mask, reader)
    extractor.load()

    class FaceOnCard:
        def empty(self):
            return False

        def detectMultiScale(self, gray, **kwargs):
            if np.mean(np.abs(gray.astype(float) - reference)) < 2:
                return [(100, 160, 120, 140)]
            return []

    extractor._backend.face_detector = FaceOnCard()
    result = extractor.extract_result(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))

    assert result.status == "needs_review"
    assert result.reason == "ambiguous_orientation"
    assert result.image is not None
    assert result.metadata["proposal_source"] == "card_edges"
    assert cv2.contourArea(np.array(result.metadata["source_corners"], np.float32)) > .4 * stamp_mask.size

    extractor._backend.face_detector = None
    result = extractor.extract_result(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    assert result.reason == "no_card_layout_evidence"
    assert result.image is None


def test_full_bleed_card_returns_review_only_with_layout_evidence(monkeypatch):
    card, _, _, _ = scene()
    reference = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY).astype(float)
    stamp_mask = np.zeros(card.shape[:2], np.uint8)
    cv2.rectangle(stamp_mask, (250, 340), (420, 410), 1, -1)

    class Reader:
        def readtext(self, rgb, **kwargs):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if np.mean(np.abs(gray.astype(float) - reference)) < 2:
                return [([[480,145],[650,145],[650,170],[480,170]],"1234567890",.91),
                        ([[500,235],[650,235],[650,260],[500,260]],"1234",.92)]
            return []

    class FaceOnCard:
        def empty(self):
            return False

        def detectMultiScale(self, gray, **kwargs):
            if np.mean(np.abs(gray.astype(float) - reference)) < 2:
                return [(100, 160, 120, 140)]
            return []

    extractor, _, _ = make_extractor(monkeypatch, stamp_mask, Reader())
    extractor.load()
    extractor._backend.face_detector = FaceOnCard()
    rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    result = extractor.extract_result(rgb)

    assert result.status == "needs_review" and result.image is not None
    assert result.metadata["proposal_source"] == "image_frame"
    assert result.metadata["touches_image_border"]
    assert "unreliable_geometry" in result.reason
    with pytest.raises(ReviewRequiredError):
        extractor.extract(rgb)

    extractor._backend.face_detector = None
    assert extractor.extract_result(rgb).image is None


def test_review_candidate_is_not_silently_accepted(monkeypatch):
    _,photo,mask,_=scene()
    class Empty:
        def readtext(self,*args,**kwargs): return []
    extractor,_,_=make_extractor(monkeypatch,mask,Empty())
    result=extractor.extract_result(photo)
    assert result.status=='needs_review' and result.image is not None
    assert result.reason=='ambiguous_orientation'
    with pytest.raises(ReviewRequiredError) as error: extractor.extract(photo)
    assert error.value.result.image is not None
    assert extractor.extract_result(photo,orientation_override=0).ok


def test_unrecoverable_mask_returns_review_without_crop(monkeypatch):
    _,photo,mask,_=scene()
    reader = SimpleNamespace(readtext=lambda *args, **kwargs: [])
    extractor,_,_=make_extractor(monkeypatch,np.zeros_like(mask),reader)
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
