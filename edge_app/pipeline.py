from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2

from edge_app.buffer import RingFrameBuffer
from edge_app.candidates import build_candidates
from edge_app.clips import ClipBuilder
from edge_app.config import AppConfig
from edge_app.diagnostics import PipelineDiagnostics
from edge_app.events import EventStateMachine
from edge_app.fusion import fuse_candidates
from edge_app.inferencers import DryRunActionInferencer, build_action_inferencer, build_pose_inferencer
from edge_app.output import EventWriter
from edge_app.tracker import ByteTrack
from edge_app.types import EventRecord, FramePacket
from edge_app.video import VideoSource
from edge_app.visualization import draw_overlay


@dataclass
class PendingEvent:
    record: EventRecord
    packet: FramePacket


class BehaviorPipeline:
    def __init__(self, config: AppConfig, source: str, dry_run_models: bool = False, save_vis: bool = False) -> None:
        self.config = config
        self.source = source
        self.dry_run_models = dry_run_models
        self.save_vis = save_vis or config.runtime.save_vis
        self.pose = build_pose_inferencer(config.models.pose, dry_run_models, config.runtime.dry_run_mode)
        self.action_degraded = False
        self.action_degraded_reason = ""
        try:
            self.action = build_action_inferencer(config.models.action, dry_run_models, config.runtime.dry_run_mode)
        except Exception as exc:
            if not config.runtime.allow_action_fallback and not dry_run_models:
                raise
            self.action = DryRunActionInferencer(config.models.action.labels, "neutral")
            self.action_degraded = True
            self.action_degraded_reason = str(exc)
        self.tracker = ByteTrack(config.tracking, frame_rate=config.input.target_fps)
        self.clip_builder = ClipBuilder(
            clip_length=config.models.action.clip_length,
            sample_stride=config.models.action.sample_stride,
            crop_size=config.models.action.input_size,
            infer_interval_frames=config.models.action.infer_interval_frames,
        )
        cache_seconds = config.events.evidence_pre_sec + config.events.evidence_post_sec + 2.0
        self.frame_buffer = RingFrameBuffer(cache_seconds, config.input.target_fps)
        self.events = EventStateMachine(config.events, source)
        self.writer = EventWriter(config.events, config.input.target_fps)
        self.pending_events: List[PendingEvent] = []
        self.vis_writer = None
        self.diagnostics = PipelineDiagnostics(config.runtime, source)

    def run(self) -> None:
        video = VideoSource(self.source, self.config.input)
        max_frames = self.config.runtime.max_frames
        try:
            for packet in video.frames(max_frames=max_frames):
                self.process_packet(packet)
            self.flush_pending()
        finally:
            if self.vis_writer is not None:
                self.vis_writer.release()
            self.diagnostics.finish(self.action_degraded, self.action_degraded_reason)

    def process_packet(self, packet: FramePacket) -> List[EventRecord]:
        self.frame_buffer.append(packet)
        detections = self.pose.infer(packet.frame_bgr)
        tracks = self.tracker.update(detections, packet.timestamp)
        rois = build_candidates(tracks, packet.frame_bgr.shape, self.config.candidates)
        self.diagnostics.record_frame(packet.frame_id, len(detections), len(tracks), rois)
        ready_rois, clips = self.clip_builder.add(packet, rois)
        action_scores = self.action.infer(clips) if len(ready_rois) else []
        fused = fuse_candidates(ready_rois, action_scores, tracks, self.config.events, packet.timestamp)
        confirmed = self.events.update(fused, packet.timestamp)
        self.diagnostics.record_inference(
            packet.frame_id,
            packet.timestamp,
            ready_rois,
            action_scores,
            fused,
            self.events.snapshot(),
            self.events.suppression_snapshot(),
        )
        self.diagnostics.record_confirmed(confirmed)
        for record in confirmed:
            self.pending_events.append(PendingEvent(record=record, packet=packet))

        written = self._write_ready(packet.timestamp)
        if self.save_vis:
            self._write_visualization(packet, tracks, fused)
        return written

    def flush_pending(self) -> List[EventRecord]:
        written = []
        for pending in list(self.pending_events):
            written.append(self.writer.write(pending.record, pending.packet, self.frame_buffer))
            self.pending_events.remove(pending)
        return written

    def _write_ready(self, now_ts: float) -> List[EventRecord]:
        written = []
        for pending in list(self.pending_events):
            if now_ts >= pending.record.confirm_ts + self.config.events.evidence_post_sec:
                written.append(self.writer.write(pending.record, pending.packet, self.frame_buffer))
                self.pending_events.remove(pending)
        return written

    def _write_visualization(self, packet: FramePacket, tracks, candidates) -> None:
        frame = draw_overlay(packet.frame_bgr, tracks, candidates)
        if self.vis_writer is None:
            path = Path(self.config.runtime.vis_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.vis_writer = cv2.VideoWriter(str(path), fourcc, max(1.0, self.config.input.target_fps), (w, h))
        self.vis_writer.write(frame)
