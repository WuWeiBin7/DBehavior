import numpy as np

from edge_app.geometry import clip_bbox, crop_resize_letterbox, expand_bbox, merge_bboxes


def test_bbox_clip_expand_and_merge():
    assert clip_bbox((-10, 5, 30, 80), 20, 40) == (0.0, 5.0, 19.0, 39.0)
    expanded = expand_bbox((10, 10, 20, 20), 2.0, 100, 100)
    assert expanded == (5.0, 5.0, 25.0, 25.0)
    merged = merge_bboxes([(10, 10, 20, 20), (18, 8, 30, 25)], 100, 100)
    assert merged == (10.0, 8.0, 30.0, 25.0)


def test_crop_resize_letterbox_handles_boundary():
    frame = np.full((50, 80, 3), 255, dtype=np.uint8)
    crop = crop_resize_letterbox(frame, (-20, -20, 30, 30), (224, 224))
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8
