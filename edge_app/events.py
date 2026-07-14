from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from edge_app.config import BehaviorEventConfig, EventsConfig
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


class EventStateMachine:
    def __init__(self, config: EventsConfig, source: str) -> None:
        self.config = config
        self.source = source
        self.states: Dict[str, _BehaviorState] = {
            "fall": _BehaviorState(),
            "fight": _BehaviorState(),
        }

    def update(self, candidates: List[EventCandidate], timestamp: float | None = None) -> List[EventRecord]:
        records: List[EventRecord] = []
        latest_by_type = {}
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            latest_by_type.setdefault(candidate.event_type, candidate)

        now_values = [c.timestamp for c in candidates]
        now = max(now_values) if now_values else (float(timestamp) if timestamp is not None else 0.0)

        for event_type, state in self.states.items():
            cfg = self._cfg(event_type)
            candidate: Optional[EventCandidate] = latest_by_type.get(event_type)
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
                    state.state = EventState.COOLDOWN
                    state.cooldown_until = candidate.timestamp + cfg.cooldown_sec
                    state.clip_count = 0
                    state.best_score = 0.0
        return records

    def snapshot(self) -> Dict[str, str]:
        return {event_type: state.state.value for event_type, state in self.states.items()}

    def _cfg(self, event_type: str) -> BehaviorEventConfig:
        return self.config.fall if event_type == "fall" else self.config.fight

    @staticmethod
    def _confirmed(event_type: str, cfg: BehaviorEventConfig, state: _BehaviorState, candidate: EventCandidate) -> bool:
        if candidate.score < cfg.confirm_threshold:
            return False
        if event_type == "fight":
            return state.clip_count >= cfg.confirm_clip_count
        return (candidate.timestamp - state.start_ts) >= cfg.confirm_duration_sec
