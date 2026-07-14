import numpy as np

from edge_app.config import TrackingConfig
from edge_app.tracker import ByteTrack
from edge_app.types import PersonDetection


def _det(x1, y1, x2, y2, score=0.9):
    return PersonDetection((x1, y1, x2, y2), score, np.zeros((17, 3), dtype=np.float32))


def test_bytetrack_keeps_id_after_short_lost_period():
    cfg = TrackingConfig(track_high_thresh=0.45, track_low_thresh=0.1, match_thresh=0.7, max_age=30, min_hits=1)
    tracker = ByteTrack(cfg, frame_rate=15)

    tracks1 = tracker.update([_det(10, 10, 50, 100)], 1.0)
    tracks2 = tracker.update([_det(12, 12, 52, 102)], 1.1)
    tracks3 = tracker.update([], 1.2)
    tracks4 = tracker.update([_det(14, 14, 54, 104)], 1.3)

    assert len(tracks1) == 1
    assert len(tracks2) == 1
    assert tracks3 == []
    assert len(tracks4) == 1
    assert tracks1[0].track_id == tracks2[0].track_id == tracks4[0].track_id
