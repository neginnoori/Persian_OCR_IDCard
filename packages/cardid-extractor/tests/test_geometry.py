import cv2
import numpy as np
import pytest
from cardid.geometry import order_points,quad_from_mask,warp_candidates,choose_orientation


def scene():
    w,h=856,540
    source=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    target=np.array([[185,100],[920,230],[760,780],[90,660]],np.float32)
    card=np.full((h,w,3),210,np.uint8)
    for point,color in [((60,60),(0,0,255)),((790,60),(0,255,0)),((790,480),(255,0,0)),((60,480),(0,255,255))]:
        cv2.circle(card,point,26,color,-1)
    cv2.putText(card,'UPRIGHT',(240,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
    matrix=cv2.getPerspectiveTransform(source,target)
    photo=cv2.warpPerspective(card,matrix,(1050,900))
    mask=cv2.warpPerspective(np.ones((h,w),np.uint8),matrix,(1050,900),flags=cv2.INTER_NEAREST)
    return card,photo,mask,source


@pytest.mark.parametrize('turns',range(4))
def test_perspective_and_inverse_coordinates(turns):
    card,photo,mask,source=scene()
    photo=np.ascontiguousarray(np.rot90(photo,turns));mask=np.ascontiguousarray(np.rot90(mask,turns))
    quad,metrics=quad_from_mask(mask)
    assert metrics['geometry_ok'] and metrics['mask_quad_iou']>.99
    candidates=warp_candidates(photo,quad)
    best=min(candidates,key=lambda c:np.mean(np.abs(c['card'].astype(float)-card.astype(float))))
    assert np.mean(np.abs(best['card'].astype(float)-card.astype(float)))<3
    assert np.max(np.abs(cv2.perspectiveTransform(best['quad'][None],best['H'])[0]-source))<.001
    roi=np.array([[[400,200],[780,200],[780,300],[400,300]]],np.float32)
    mapped=cv2.perspectiveTransform(roi,np.linalg.inv(best['H']))
    assert np.max(np.abs(cv2.perspectiveTransform(mapped,best['H'])-roi))<.001


def test_nonduplicated_diamond_corners():
    assert len(np.unique(order_points([[100,0],[200,100],[100,200],[0,100]]),axis=0))==4


def test_uncertainty_and_manual_orientation():
    class Reader:
        def readtext(self,*args,**kwargs): return []
    _,photo,mask,_=scene()
    quad,_=quad_from_mask(mask);candidates=warp_candidates(photo,quad)
    _,meta=choose_orientation(candidates,Reader())
    assert not meta['orientation_ok']
    for override in (0,90,180,270):
        chosen,meta=choose_orientation(candidates,None,override=override)
        assert meta['orientation_ok'] and chosen['corner_shift_degrees']==override


def test_invalid_and_clipped_masks():
    with pytest.raises(ValueError): quad_from_mask(np.zeros((30,30),np.uint8))
    mask=np.zeros((500,700),np.uint8);cv2.rectangle(mask,(0,50),(650,450),1,-1)
    assert not quad_from_mask(mask)[1]['geometry_ok']
    circle=np.zeros((500,500),np.uint8);cv2.circle(circle,(250,250),180,1,-1)
    try:
        assert not quad_from_mask(circle)[1]['geometry_ok']
    except ValueError:
        pass
