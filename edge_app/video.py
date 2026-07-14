from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2

from edge_app.config import InputConfig
from edge_app.types import FramePacket


def build_rtsp_gstreamer_pipeline(rtsp_url: str, width: int, height: int, fps: int) -> str:
    return (
        f"rtspsrc location={rtsp_url} latency=120 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false"
    )


@dataclass
class VideoSource:
    source: str
    config: InputConfig

    def _open(self) -> cv2.VideoCapture:
        if self.source.startswith("rtsp://") and self.config.use_gstreamer:
            pipeline = build_rtsp_gstreamer_pipeline(
                self.source, self.config.width, self.config.height, self.config.target_fps
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
        cap = cv2.VideoCapture(self.source)
        if self.config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.target_fps:
            cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        return cap

    def frames(self, max_frames: int = 0) -> Iterator[FramePacket]:
        frame_id = 0
        decoded_frame_id = 0
        cap: Optional[cv2.VideoCapture] = None
        is_rtsp = self.source.startswith("rtsp://")
        source_fps = 0.0
        next_file_ts = 0.0
        while True:
            if cap is None or not cap.isOpened():
                cap = self._open()
                if not cap.isOpened():
                    if not self.source.startswith("rtsp://"):
                        raise RuntimeError(f"Failed to open video source: {self.source}")
                    time.sleep(self.config.reconnect_sec)
                    continue
                source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                if not self.source.startswith("rtsp://"):
                    break
                time.sleep(self.config.reconnect_sec)
                continue

            if is_rtsp:
                timestamp = time.time()
            else:
                media_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                timestamp = media_ms / 1000.0 if media_ms > 0 else decoded_frame_id / max(1.0, source_fps)
                decoded_frame_id += 1
                target_fps = float(self.config.target_fps or 0)
                if target_fps > 0 and source_fps > target_fps:
                    if timestamp + 1e-6 < next_file_ts:
                        continue
                    next_file_ts += 1.0 / target_fps

            yield FramePacket(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_bgr=frame,
                source=self.source,
            )
            frame_id += 1
            if max_frames and frame_id >= max_frames:
                break

        if cap is not None:
            cap.release()
