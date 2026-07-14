from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np

from edge_app.types import EventCandidate, Track


def draw_overlay(frame: np.ndarray, tracks: List[Track], candidates: List[EventCandidate]) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"id={track.track_id}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        for x, y, s in track.keypoints:
            if s > 0.3:
                cv2.circle(out, (int(x), int(y)), 2, (255, 200, 0), -1)
    for cand in candidates:
        x1, y1, x2, y2 = [int(v) for v in cand.bbox_xyxy]
        color = (0, 0, 255) if cand.event_type == "fall" else (0, 128, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{cand.event_type}:{cand.score:.2f}",
            (x1, min(out.shape[0] - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return out
