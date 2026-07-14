from __future__ import annotations

from typing import Dict, List

import numpy as np

from edge_app.config import EventsConfig
from edge_app.geometry import (
    body_angle_score,
    keypoint_motion_score,
    low_posture_score,
    track_stability_score,
    vertical_drop_score,
)
from edge_app.types import ActionScore, CandidateROI, EventCandidate, Track


def fuse_candidates(
    rois: List[CandidateROI],
    action_scores: List[ActionScore],
    tracks: List[Track],
    events_cfg: EventsConfig,
    timestamp: float,
) -> List[EventCandidate]:
    by_id = {t.track_id: t for t in tracks}
    fused: List[EventCandidate] = []
    for roi, action in zip(rois, action_scores):
        if roi.event_type == "fall":
            track = by_id.get(roi.track_ids[0])
            if track is None:
                continue
            score = _weighted(
                events_cfg.fall.weights,
                {
                    "model": action.get("fall"),
                    "vertical_drop": vertical_drop_score(track),
                    "body_angle": body_angle_score(track),
                    "low_posture": low_posture_score(track),
                    "track_stability": track_stability_score(track),
                },
            )
            fused.append(EventCandidate("fall", float(score), roi.track_ids, roi.bbox_xyxy, timestamp))
            fight_cfg = events_cfg.fight
            if fight_cfg.single_person_fallback and action.get("fight") >= fight_cfg.single_person_model_threshold:
                fight_score = _weighted(
                    fight_cfg.weights,
                    {
                        "model": action.get("fight"),
                        "proximity": 0.0,
                        "limb_motion": keypoint_motion_score(track),
                        "overlap": 0.0,
                        "continuous": track_stability_score(track),
                    },
                )
                fused.append(
                    EventCandidate("fight", float(fight_score), roi.track_ids, roi.bbox_xyxy, timestamp)
                )
            continue
        elif roi.event_type == "fight":
            roi_tracks = [by_id[i] for i in roi.track_ids if i in by_id]
            continuous = min([track_stability_score(t) for t in roi_tracks] or [0.0])
            score = _weighted(
                events_cfg.fight.weights,
                {
                    "model": action.get("fight"),
                    "proximity": roi.features.get("proximity", roi.score_hint),
                    "limb_motion": roi.features.get(
                        "limb_motion", max([keypoint_motion_score(t) for t in roi_tracks] or [0.0])
                    ),
                    "overlap": roi.features.get("overlap", 0.0),
                    "continuous": continuous,
                },
            )
        else:
            continue
        fused.append(EventCandidate(roi.event_type, float(score), roi.track_ids, roi.bbox_xyxy, timestamp))
    return fused


def _weighted(weights: Dict[str, float], values: Dict[str, float]) -> float:
    if not weights:
        return float(np.mean(list(values.values())))
    total = sum(float(v) for v in weights.values()) or 1.0
    score = 0.0
    for name, weight in weights.items():
        score += float(weight) * float(values.get(name, 0.0))
    return float(np.clip(score / total, 0.0, 1.0))
