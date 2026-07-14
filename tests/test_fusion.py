import numpy as np

from edge_app.config import BehaviorEventConfig, EventsConfig
from edge_app.fusion import fuse_candidates
from edge_app.types import ActionScore, CandidateROI, Track


def test_single_person_fight_fallback_uses_high_model_probability():
    track = Track(
        track_id=7,
        bbox_xyxy=(10, 10, 60, 100),
        score=0.9,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        history=[(float(i), (10, 10, 60, 100), np.zeros((17, 3), dtype=np.float32)) for i in range(12)],
    )
    roi = CandidateROI("fall", "person", [7], track.bbox_xyxy, 0.1)
    cfg = EventsConfig(
        fight=BehaviorEventConfig(
            single_person_fallback=True,
            single_person_model_threshold=0.85,
            weights={"model": 1.0},
        )
    )

    fused = fuse_candidates(
        [roi],
        [ActionScore({"normal": 0.01, "fight": 0.98, "fall": 0.01})],
        [track],
        cfg,
        1.0,
    )

    assert [candidate.event_type for candidate in fused] == ["fall", "fight"]
    assert fused[1].score == 0.98
