from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from edge_app.types import BBox, Track


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, scale: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1) * scale
    bh = max(1.0, y2 - y1) * scale
    return clip_bbox((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5), width, height)


def merge_bboxes(boxes: Sequence[BBox], width: int, height: int, scale: float = 1.0) -> BBox:
    if not boxes:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(boxes, dtype=np.float32)
    merged = float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())
    return expand_bbox(merged, scale, width, height)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def crop_resize_letterbox(frame_bgr: np.ndarray, bbox: BBox, size: Tuple[int, int]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    x1i, y1i = int(np.floor(x1)), int(np.floor(y1))
    x2i, y2i = int(np.ceil(x2)), int(np.ceil(y2))
    crop = frame_bgr[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    target_w, target_h = size
    ch, cw = crop.shape[:2]
    scale = min(target_w / max(1, cw), target_h / max(1, ch))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def body_angle_score(track: Track) -> float:
    kpts = track.keypoints
    if kpts is None or len(kpts) < 17:
        return 0.0
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_hip, right_hip = kpts[11], kpts[12]
    valid = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p[2] > 0.2]
    if len(valid) < 2:
        return 0.0
    shoulder = np.mean([p[:2] for p in (left_shoulder, right_shoulder) if p[2] > 0.2], axis=0)
    hip = np.mean([p[:2] for p in (left_hip, right_hip) if p[2] > 0.2], axis=0)
    dx = abs(float(shoulder[0] - hip[0]))
    dy = abs(float(shoulder[1] - hip[1]))
    if dx + dy <= 1e-6:
        return 0.0
    return float(dx / (dx + dy))


def vertical_drop_score(track: Track, lookback: int = 8) -> float:
    if len(track.history) < 2:
        return 0.0
    recent = track.history[-lookback:]
    y_old = (recent[0][1][1] + recent[0][1][3]) * 0.5
    y_new = (recent[-1][1][1] + recent[-1][1][3]) * 0.5
    h_old = max(1.0, recent[0][1][3] - recent[0][1][1])
    return float(np.clip((y_new - y_old) / h_old, 0.0, 1.0))


def low_posture_score(track: Track) -> float:
    x1, y1, x2, y2 = track.bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return float(np.clip((bw / bh - 0.6) / 1.0, 0.0, 1.0))


def track_stability_score(track: Track) -> float:
    return float(np.clip(len(track.history) / 12.0, 0.0, 1.0))


def keypoint_motion_score(track: Track, lookback: int = 4) -> float:
    if len(track.history) < 2:
        return 0.0
    recent = track.history[-lookback:]
    motions: List[float] = []
    for (_, _, prev), (_, _, cur) in zip(recent, recent[1:]):
        if prev is None or cur is None or len(prev) != len(cur):
            continue
        mask = (prev[:, 2] > 0.2) & (cur[:, 2] > 0.2)
        if mask.any():
            motions.append(float(np.linalg.norm(cur[mask, :2] - prev[mask, :2], axis=1).mean()))
    if not motions:
        return 0.0
    _, y1, _, y2 = track.bbox_xyxy
    return float(np.clip(np.mean(motions) / max(1.0, y2 - y1), 0.0, 1.0))
