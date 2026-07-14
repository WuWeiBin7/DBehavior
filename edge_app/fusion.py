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
            fall_cfg = events_cfg.fall
            vertical_drop = vertical_drop_score(track)
            body_angle = body_angle_score(track)
            low_posture = low_posture_score(track)
            stability = track_stability_score(track)
            current_geometry = 0.35 * vertical_drop + 0.35 * body_angle + 0.30 * low_posture
            fall_geometry = max(
                float(current_geometry),
                float(roi.features.get("temporal_hint_max", roi.score_hint)),
            )
            model_gate = _geometry_gate(
                fall_geometry,
                fall_cfg.model_geometry_gate_threshold,
                fall_cfg.model_geometry_gate_floor,
            )
            fall_values = {
                "model": action.get("fall") * model_gate,
                "vertical_drop": vertical_drop,
                "body_angle": body_angle,
                "low_posture": low_posture,
                "track_stability": stability,
            }
            score = _weighted(fall_cfg.weights, fall_values)
            fall_features = {
                "model_probability": action.get("fall"),
                "model_gate": model_gate,
                "fall_geometry": fall_geometry,
                **fall_values,
            }
            fused.append(
                EventCandidate(
                    "fall", float(score), roi.track_ids, roi.bbox_xyxy, timestamp, fall_features
                )
            )
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
                    EventCandidate(
                        "fight",
                        float(fight_score),
                        roi.track_ids,
                        roi.bbox_xyxy,
                        timestamp,
                        {
                            "model_probability": action.get("fight"),
                            "single_person_fallback": 1.0,
                            "limb_motion": keypoint_motion_score(track),
                            "continuous": track_stability_score(track),
                        },
                    )
                )
            continue
        elif roi.event_type == "fight":
            roi_tracks = [by_id[i] for i in roi.track_ids if i in by_id]
            continuous = min([track_stability_score(t) for t in roi_tracks] or [0.0])
            fight_values = {
                "model": action.get("fight"),
                "proximity": roi.features.get("proximity", roi.score_hint),
                "limb_motion": roi.features.get(
                    "limb_motion", max([keypoint_motion_score(t) for t in roi_tracks] or [0.0])
                ),
                "overlap": roi.features.get("overlap", 0.0),
                "continuous": continuous,
            }
            score = _weighted(events_cfg.fight.weights, fight_values)
        else:
            continue
        fused.append(
            EventCandidate(
                roi.event_type,
                float(score),
                roi.track_ids,
                roi.bbox_xyxy,
                timestamp,
                {"model_probability": action.get("fight"), **fight_values},
            )
        )
    return fused


def _weighted(weights: Dict[str, float], values: Dict[str, float]) -> float:
    if not weights:
        return float(np.mean(list(values.values())))
    total = sum(float(v) for v in weights.values()) or 1.0
    score = 0.0
    for name, weight in weights.items():
        score += float(weight) * float(values.get(name, 0.0))
    return float(np.clip(score / total, 0.0, 1.0))


def _geometry_gate(geometry: float, threshold: float, floor: float) -> float:
    if threshold <= 0.0:
        return 1.0
    floor = float(np.clip(floor, 0.0, 1.0))
    return float(np.clip(max(floor, geometry / threshold), 0.0, 1.0))
