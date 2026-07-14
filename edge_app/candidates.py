from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np

from edge_app.geometry import (
    body_angle_score,
    expand_bbox,
    keypoint_motion_score,
    low_posture_score,
    merge_bboxes,
    vertical_drop_score,
)
from edge_app.config import CandidateConfig
from edge_app.types import CandidateROI, Track


def build_candidates(
    tracks: List[Track], frame_shape: tuple[int, int, int], config: CandidateConfig | None = None
) -> List[CandidateROI]:
    config = config or CandidateConfig()
    height, width = frame_shape[:2]
    candidates: List[CandidateROI] = []

    for track in tracks:
        fall_hint = 0.35 * vertical_drop_score(track) + 0.35 * body_angle_score(track) + 0.30 * low_posture_score(track)
        if fall_hint >= config.fall_min_hint or len(track.history) >= config.fall_min_history:
            candidates.append(
                CandidateROI(
                    event_type="fall",
                    roi_type="person",
                    track_ids=[track.track_id],
                    bbox_xyxy=expand_bbox(track.bbox_xyxy, config.fall_roi_scale, width, height),
                    score_hint=float(fall_hint),
                    features={"fall_hint": float(fall_hint)},
                )
            )

    for a, b in combinations(tracks, 2):
        proximity = _proximity_score(a, b)
        motion = max(keypoint_motion_score(a), keypoint_motion_score(b))
        overlap = _overlap_score(a, b)
        fight_hint = 0.45 * proximity + 0.35 * motion + 0.20 * overlap
        if fight_hint >= config.fight_min_hint and proximity >= config.fight_min_proximity:
            candidates.append(
                CandidateROI(
                    event_type="fight",
                    roi_type="group",
                    track_ids=[a.track_id, b.track_id],
                    bbox_xyxy=merge_bboxes(
                        [a.bbox_xyxy, b.bbox_xyxy], width, height, scale=config.fight_roi_scale
                    ),
                    score_hint=float(fight_hint),
                    features={
                        "proximity": float(proximity),
                        "limb_motion": float(motion),
                        "overlap": float(overlap),
                    },
                )
            )
    return candidates


def _proximity_score(a: Track, b: Track) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ac = np.asarray([(ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5])
    bc = np.asarray([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5])
    distance = float(np.linalg.norm(ac - bc))
    scale = max(1.0, max(ay2 - ay1, by2 - by1))
    return float(np.clip(1.0 - distance / (2.0 * scale), 0.0, 1.0))


def _overlap_score(a: Track, b: Track) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(max(1.0, (ax2 - ax1) * (ay2 - ay1)), max(1.0, (bx2 - bx1) * (by2 - by1)))
    return float(np.clip(inter / small, 0.0, 1.0))
