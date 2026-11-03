import numpy as np
from cardid.geometry import quad_from_mask


def test_full_frame_card_is_not_rejected_for_border_contact():
    _, metrics = quad_from_mask(np.ones((540, 856), np.uint8))
    assert metrics['geometry_ok']
    assert metrics['touches_image_border']
    assert metrics['border_status'] == 'tight_crop_candidate'


def test_single_cut_edge_still_requires_review():
    mask = np.zeros((700, 1000), np.uint8)
    mask[100:600, :800] = 1
    _, metrics = quad_from_mask(mask)
    assert not metrics['geometry_ok']
    assert metrics['border_status'] == 'clipped_or_uncertain'


def test_card_with_background_keeps_existing_acceptance():
    mask = np.zeros((700, 1000), np.uint8)
    mask[100:600, 100:900] = 1
    _, metrics = quad_from_mask(mask)
    assert metrics['geometry_ok']
    assert metrics['border_status'] == 'clear'


def test_tight_crop_with_small_top_and_bottom_margins():
    mask = np.zeros((515, 800), np.uint8)
    mask[5:514, :800] = 1
    _, metrics = quad_from_mask(mask)
    assert metrics['geometry_ok']
    assert metrics['border_status'] == 'tight_crop_candidate'


def test_near_frame_corners_do_not_accept_low_occupancy_clipped_card():
    mask = np.zeros((515, 800), np.uint8)
    mask[50:500, :800] = 1
    _, metrics = quad_from_mask(mask)
    assert not metrics['geometry_ok']


def test_valid_hull_wins_over_higher_iou_outside_edge_fit(monkeypatch):
    import cardid.geometry as geometry
    mask = np.zeros((200, 300), np.uint8)
    mask[30:170, 1:290] = 1
    def outside_fit(quad, contour):
        result = quad.copy()
        result[0, 0] = -2.75
        return result
    monkeypatch.setattr(geometry, '_refine_edges', outside_fit)
    monkeypatch.setattr(geometry, '_quad_metrics', lambda q, m: ((.995, 1.) if q[:,0].min() < 0 else (.98, 1.)))
    quad, metrics = geometry.quad_from_mask(mask)
    assert metrics['geometry_ok']
    assert metrics['quad_method'] == 'hull_quad'
    assert metrics['mask_quad_iou'] == .98
    assert metrics['geometry_failure_reasons'] == []
    assert (quad[:,0] >= 0).all()


def test_outside_refinement_does_not_waive_low_quality_hull(monkeypatch):
    import cardid.geometry as geometry
    mask = np.zeros((200, 300), np.uint8)
    mask[30:170, 1:290] = 1
    def outside_fit(quad, contour):
        result = quad.copy()
        result[0, 0] = -2.75
        return result
    monkeypatch.setattr(geometry, '_refine_edges', outside_fit)
    monkeypatch.setattr(geometry, '_quad_metrics', lambda q, m: ((.995, 1.) if q[:,0].min() < 0 else (.7, 1.)))
    _, metrics = geometry.quad_from_mask(mask)
    assert not metrics['geometry_ok']
    assert 'corners_outside_image' in metrics['geometry_failure_reasons']
