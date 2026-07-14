from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import DefaultDict, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from edge_app.geometry import crop_resize_letterbox
from edge_app.types import CandidateROI, FramePacket


@dataclass
class ClipBuilder:
    clip_length: int
    sample_stride: int
    crop_size: Tuple[int, int]
    infer_interval_frames: int = 1
    _roi_frames: DefaultDict[str, deque[np.ndarray]] = field(default_factory=lambda: defaultdict(deque))
    _roi_hints: DefaultDict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _last_emitted_frame: Dict[str, int] = field(default_factory=dict)
    _last_seen_frame: Dict[str, int] = field(default_factory=dict)

    def add(self, packet: FramePacket, rois: List[CandidateROI]) -> tuple[List[CandidateROI], np.ndarray]:
        ready_rois: List[CandidateROI] = []
        clips: List[np.ndarray] = []
        max_needed = self.clip_length * max(1, self.sample_stride)
        for roi in rois:
            key = self._key(roi)
            crop = crop_resize_letterbox(packet.frame_bgr, roi.bbox_xyxy, self.crop_size)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            queue = self._roi_frames[key]
            hints = self._roi_hints[key]
            self._last_seen_frame[key] = packet.frame_id
            queue.append(crop_rgb)
            hints.append(float(roi.score_hint))
            while len(queue) > max_needed:
                queue.popleft()
                hints.popleft()
            last_emitted = self._last_emitted_frame.get(key, -self.infer_interval_frames)
            can_emit = packet.frame_id - last_emitted >= max(1, self.infer_interval_frames)
            if len(queue) >= max_needed and can_emit:
                sampled = list(queue)[-max_needed:: self.sample_stride]
                if len(sampled) == self.clip_length:
                    temporal_features = dict(roi.features)
                    temporal_features["temporal_hint_max"] = float(max(hints))
                    temporal_features["temporal_hint_mean"] = float(np.mean(hints))
                    ready_rois.append(replace(roi, features=temporal_features))
                    clips.append(np.stack(sampled, axis=0))
                    self._last_emitted_frame[key] = packet.frame_id
        self._discard_stale(packet.frame_id, max_needed * 2)
        if not clips:
            return [], np.empty((0, self.clip_length, self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
        return ready_rois, np.stack(clips, axis=0).astype(np.uint8)

    @staticmethod
    def _key(roi: CandidateROI) -> str:
        ids = "-".join(str(i) for i in sorted(roi.track_ids))
        return f"{roi.event_type}:{roi.roi_type}:{ids}"

    def _discard_stale(self, frame_id: int, max_idle_frames: int) -> None:
        stale = [key for key, seen in self._last_seen_frame.items() if frame_id - seen > max_idle_frames]
        for key in stale:
            self._roi_frames.pop(key, None)
            self._roi_hints.pop(key, None)
            self._last_seen_frame.pop(key, None)
            self._last_emitted_frame.pop(key, None)
