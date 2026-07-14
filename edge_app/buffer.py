from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List

from edge_app.types import FramePacket


@dataclass
class RingFrameBuffer:
    max_seconds: float
    fps: float

    def __post_init__(self) -> None:
        maxlen = max(1, int(round(self.max_seconds * max(1.0, self.fps))))
        self._frames: deque[FramePacket] = deque(maxlen=maxlen)

    def append(self, packet: FramePacket) -> None:
        self._frames.append(packet)

    def recent(self, seconds: float, now_ts: float) -> List[FramePacket]:
        start = now_ts - seconds
        return [p for p in self._frames if p.timestamp >= start]

    def all(self) -> List[FramePacket]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)
