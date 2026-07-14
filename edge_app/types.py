from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


class EventState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray
    source: str


@dataclass
class PersonDetection:
    bbox_xyxy: BBox
    score: float
    keypoints: np.ndarray


@dataclass
class Track:
    track_id: int
    bbox_xyxy: BBox
    score: float
    keypoints: np.ndarray
    age: int = 0
    hits: int = 1
    missed: int = 0
    history: List[Tuple[float, BBox, np.ndarray]] = field(default_factory=list)


@dataclass
class CandidateROI:
    event_type: str
    roi_type: str
    track_ids: List[int]
    bbox_xyxy: BBox
    score_hint: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActionScore:
    scores: Dict[str, float]

    def get(self, label: str, default: float = 0.0) -> float:
        return float(self.scores.get(label, default))


@dataclass
class EventCandidate:
    event_type: str
    score: float
    track_ids: List[int]
    bbox_xyxy: BBox
    timestamp: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class EventRecord:
    event_id: str
    event_type: str
    state: str
    start_ts: float
    confirm_ts: float
    score: float
    track_ids: Sequence[int]
    bbox_xyxy: BBox
    snapshot_path: Optional[str]
    video_path: Optional[str]
    source: str
