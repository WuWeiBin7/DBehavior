from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from edge_app.config import BehaviorEventConfig, EventsConfig
from edge_app.geometry import bbox_iou
from edge_app.types import EventCandidate, EventRecord, EventState


@dataclass
class _BehaviorState:
    state: EventState = EventState.NORMAL
    start_ts: float = 0.0
    last_ts: float = 0.0
    best_score: float = 0.0
    clip_count: int = 0
    cooldown_until: float = 0.0
    track_ids: List[int] = field(default_factory=list)
    bbox_xyxy: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class _ConfirmedContext:
    timestamp: float
    track_ids: List[int]
    bbox_xyxy: tuple[float, float, float, float]


class EventStateMachine:
    def __init__(self, config: EventsConfig, source: str) -> None:
        self.config = config
        self.source = source
        # Fight is evaluated first so a simultaneous weak Fall candidate can
        # be arbitrated against the newly confirmed Fight incident.
        self.states: Dict[str, _BehaviorState] = {
            "fight": _BehaviorState(),
            "fall": _BehaviorState(),
        }
        self.last_confirmed: Dict[str, _ConfirmedContext] = {}
        self.last_suppressions: List[Dict] = []

    def update(self, candidates: List[EventCandidate], timestamp: float | None = None) -> List[EventRecord]:
        records: List[EventRecord] = []
        self.last_suppressions = []
        latest_by_type = {}
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            latest_by_type.setdefault(candidate.event_type, candidate)

        now_values = [c.timestamp for c in candidates]
        now = max(now_values) if now_values else (float(timestamp) if timestamp is not None else 0.0)

        for event_type, state in self.states.items():
            cfg = self._cfg(event_type)
            candidate: Optional[EventCandidate] = latest_by_type.get(event_type)
            if (
                event_type == "fall"
                and candidate is not None
                and candidate.score >= cfg.suspect_threshold
                and self._suppress_fall_after_fight(candidate)
            ):
                candidate = None
            ts = candidate.timestamp if candidate else now

            if state.state == EventState.COOLDOWN:
                if ts >= state.cooldown_until:
                    state.state = EventState.NORMAL
                    state.clip_count = 0
                else:
                    continue

            if candidate is None:
                if state.state == EventState.SUSPECT and now - state.last_ts > cfg.candidate_gap_sec:
                    state.state = EventState.NORMAL
                    state.clip_count = 0
                continue

            state.last_ts = candidate.timestamp
            if state.state == EventState.NORMAL and candidate.score >= cfg.suspect_threshold:
                state.state = EventState.SUSPECT
                state.start_ts = candidate.timestamp
                state.best_score = candidate.score
                state.clip_count = 1
                state.track_ids = candidate.track_ids
                state.bbox_xyxy = candidate.bbox_xyxy
                continue

            if state.state == EventState.SUSPECT:
                if candidate.score < cfg.suspect_threshold:
                    state.state = EventState.NORMAL
                    state.clip_count = 0
                    continue
                state.best_score = max(state.best_score, candidate.score)
                state.clip_count += 1
                state.track_ids = candidate.track_ids
                state.bbox_xyxy = candidate.bbox_xyxy
                if self._confirmed(event_type, cfg, state, candidate):
                    record = EventRecord(
                        event_id=f"{event_type}-{uuid.uuid4().hex[:12]}",
                        event_type=event_type,
                        state=EventState.CONFIRMED.value,
                        start_ts=state.start_ts,
                        confirm_ts=candidate.timestamp,
                        score=state.best_score,
                        track_ids=state.track_ids,
                        bbox_xyxy=state.bbox_xyxy,
                        snapshot_path=None,
                        video_path=None,
                        source=self.source,
                    )
                    records.append(record)
                    self.last_confirmed[event_type] = _ConfirmedContext(
                        timestamp=candidate.timestamp,
                        track_ids=list(candidate.track_ids),
                        bbox_xyxy=candidate.bbox_xyxy,
                    )
                    state.state = EventState.COOLDOWN
                    state.cooldown_until = candidate.timestamp + cfg.cooldown_sec
                    state.clip_count = 0
                    state.best_score = 0.0
        return records

    def snapshot(self) -> Dict[str, str]:
        return {event_type: state.state.value for event_type, state in self.states.items()}

    def suppression_snapshot(self) -> List[Dict]:
        return list(self.last_suppressions)

    def _suppress_fall_after_fight(self, candidate: EventCandidate) -> bool:
        suppression_sec = self.config.cross_event_suppression_sec
        fight = self.last_confirmed.get("fight")
        if suppression_sec <= 0.0 or fight is None:
            return False
        elapsed = candidate.timestamp - fight.timestamp
        if elapsed < 0.0 or elapsed > suppression_sec:
            return False
        same_track = bool(set(candidate.track_ids) & set(fight.track_ids))
        overlap = bbox_iou(candidate.bbox_xyxy, fight.bbox_xyxy)
        same_incident = same_track or overlap >= self.config.cross_event_iou_threshold
        geometry = float(candidate.features.get("fall_geometry", 0.0))
        if not same_incident or geometry >= self.config.fall_geometry_override_threshold:
            return False
        self.last_suppressions.append(
            {
                "event_type": "fall",
                "reason": "recent_fight_same_incident",
                "elapsed_sec": elapsed,
                "same_track": same_track,
                "bbox_iou": overlap,
                "fall_geometry": geometry,
            }
        )
        return True

    def _cfg(self, event_type: str) -> BehaviorEventConfig:
        return self.config.fall if event_type == "fall" else self.config.fight

    @staticmethod
    def _confirmed(event_type: str, cfg: BehaviorEventConfig, state: _BehaviorState, candidate: EventCandidate) -> bool:
        if candidate.score < cfg.confirm_threshold:
            return False
        if event_type == "fight":
            return state.clip_count >= cfg.confirm_clip_count
        return (candidate.timestamp - state.start_ts) >= cfg.confirm_duration_sec
