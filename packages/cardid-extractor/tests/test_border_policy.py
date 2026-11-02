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
