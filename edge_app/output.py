from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import cv2

from edge_app.buffer import RingFrameBuffer
from edge_app.config import EventsConfig
from edge_app.geometry import clip_bbox
from edge_app.types import EventRecord, FramePacket


class EventWriter:
    def __init__(self, config: EventsConfig, fps: float) -> None:
        self.config = config
        self.fps = fps
        self.root = Path(config.output_dir)
        self.snapshot_dir = self.root / config.snapshot_dir
        self.video_dir = self.root / config.video_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / config.jsonl_name

    def write(self, record: EventRecord, packet: FramePacket, buffer: RingFrameBuffer) -> EventRecord:
        snapshot_path = self.snapshot_dir / f"{record.event_id}.jpg"
        video_path = self.video_dir / f"{record.event_id}.mp4"
        snapshot = packet.frame_bgr.copy()
        h, w = snapshot.shape[:2]
        x1, y1, x2, y2 = clip_bbox(record.bbox_xyxy, w, h)
        cv2.rectangle(snapshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            snapshot,
            f"{record.event_type}:{record.score:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(snapshot_path), snapshot)

        frames = [
            p
            for p in buffer.all()
            if record.confirm_ts - self.config.evidence_pre_sec <= p.timestamp <= record.confirm_ts + self.config.evidence_post_sec
        ]
        if not frames:
            frames = [packet]
        self._write_video(video_path, frames)

        record.snapshot_path = str(snapshot_path)
        record.video_path = str(video_path)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def _write_video(self, path: Path, frames: List[FramePacket]) -> None:
        if not frames:
            return
        first = frames[0].frame_bgr
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, max(1.0, self.fps), (w, h))
        for packet in frames:
            writer.write(packet.frame_bgr)
        writer.release()
