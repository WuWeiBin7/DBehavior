from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from edge_app.config import ActionModelConfig, PoseModelConfig
from edge_app.geometry import clip_bbox
from edge_app.runtime import create_runtime_session
from edge_app.types import ActionScore, PersonDetection


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_dets: int) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_dets:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def _letterbox(frame: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - nw) / 2.0
    pad_y = (target_h - nh) / 2.0
    canvas[int(pad_y):int(pad_y) + nh, int(pad_x):int(pad_x) + nw] = resized
    return canvas, scale, (pad_x, pad_y)


class PoseInferencer:
    def __init__(self, config: PoseModelConfig) -> None:
        self.config = config
        self.session = create_runtime_session(_resolve_model_path(config.path))

    def infer(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        h, w = frame_bgr.shape[:2]
        image, scale, (pad_x, pad_y) = _letterbox(frame_bgr, self.config.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32)
        outputs = self.session.infer(tensor)
        boxes, scores, keypoints = _parse_ultralytics_pose_outputs(outputs, self.config.box_format)
        valid = scores >= self.config.conf_threshold
        boxes = boxes[valid]
        scores = scores[valid]
        keypoints = keypoints[valid]
        if len(scores) == 0:
            return []

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale

        keep = _nms(boxes, scores, self.config.iou_threshold, self.config.max_detections)
        detections: List[PersonDetection] = []
        for i in keep:
            detections.append(
                PersonDetection(
                    bbox_xyxy=clip_bbox(tuple(float(v) for v in boxes[i]), w, h),
                    score=float(scores[i]),
                    keypoints=keypoints[i].astype(np.float32),
                )
            )
        return detections


class ActionInferencer:
    def __init__(self, config: ActionModelConfig) -> None:
        self.config = config
        self.session = create_runtime_session(_resolve_model_path(config.path))
        self.labels = config.labels

    def infer(self, clips_rgb: np.ndarray) -> List[ActionScore]:
        if clips_rgb.size == 0:
            return []
        x = clips_rgb.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (0, 4, 1, 2, 3)).astype(np.float32)
        logits = np.concatenate([self._infer_one(sample[None, ...]) for sample in x], axis=0)
        if logits.ndim == 1:
            logits = logits[None, :]
        if logits.shape[1] != len(self.labels):
            raise RuntimeError(
                f"Action model outputs {logits.shape[1]} classes, but config declares "
                f"{len(self.labels)} labels: {self.labels}"
            )
        probs = _softmax(logits)
        return [ActionScore(dict(zip(self.labels, map(float, row)))) for row in probs]

    def _infer_one(self, x: np.ndarray) -> np.ndarray:
        x = _adapt_action_tensor_to_model(x, getattr(self.session, "input_shape", None))
        logits = np.asarray(self.session.infer(x)[0])
        if logits.ndim == 1:
            logits = logits[None, :]
        return logits.reshape(logits.shape[0], -1)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(1e-12, exp.sum(axis=1, keepdims=True))


def _adapt_action_tensor_to_model(x: np.ndarray, input_shape: List | None) -> np.ndarray:
    """Adapt N,C,T,H,W clips to common Video Swin ONNX layouts."""
    if input_shape is None:
        return x
    shape = list(input_shape)
    rank = len(shape)
    if rank not in {5, 6}:
        return x

    expected_t = _expected_temporal_length(shape)
    if expected_t:
        x = _resize_temporal_axis(x, expected_t)

    if rank == 6:
        # MMAction/Video-Swin exports often use N,num_clips,C,T,H,W.
        return x[:, None, :, :, :, :].astype(np.float32)
    return x.astype(np.float32)


def _expected_temporal_length(input_shape: List) -> int | None:
    numeric = [dim if isinstance(dim, int) else None for dim in input_shape]
    if len(numeric) == 6 and isinstance(numeric[3], int):
        return numeric[3]
    if len(numeric) == 5 and isinstance(numeric[2], int):
        return numeric[2]
    return None


def _resize_temporal_axis(x: np.ndarray, target_t: int) -> np.ndarray:
    current_t = x.shape[2]
    if current_t == target_t:
        return x
    if current_t <= 0:
        raise ValueError("Action clip has no temporal frames")
    if current_t > target_t:
        indices = np.linspace(0, current_t - 1, target_t).round().astype(int)
        return x[:, :, indices, :, :]
    pad_count = target_t - current_t
    pad = np.repeat(x[:, :, -1:, :, :], pad_count, axis=2)
    return np.concatenate([x, pad], axis=2)


def _parse_ultralytics_pose_outputs(outputs: List[np.ndarray], box_format: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse Ultralytics pose exports, including YOLOv8/11/26 style layouts."""
    if len(outputs) >= 3:
        boxes = np.squeeze(outputs[0]).astype(np.float32).reshape(-1, 4)
        scores = np.squeeze(outputs[1]).astype(np.float32).reshape(-1)
        keypoints = np.squeeze(outputs[2]).astype(np.float32)
        if keypoints.ndim == 2:
            keypoints = keypoints.reshape(-1, 17, 3)
        elif keypoints.ndim == 3 and keypoints.shape[-1] != 3:
            keypoints = keypoints.reshape(-1, 17, 3)
        boxes = _boxes_to_xyxy(boxes, box_format)
        return boxes, scores[: len(boxes)], keypoints[: len(boxes)]

    output = np.squeeze(outputs[0])
    preds = np.asarray(output, dtype=np.float32)
    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.ndim == 2 and preds.shape[0] in {56, 57, 58} and preds.shape[0] < preds.shape[1]:
        preds = preds.T
    if preds.ndim != 2 or preds.shape[1] < 56:
        raise RuntimeError(f"Unexpected Ultralytics pose output shape: {outputs[0].shape}")

    keypoint_start = preds.shape[1] - 51
    boxes = preds[:, :4]
    scores = preds[:, 4]
    keypoints = preds[:, keypoint_start:keypoint_start + 51].reshape(-1, 17, 3)
    boxes = _boxes_to_xyxy(boxes, box_format)
    return boxes, scores, keypoints


def _boxes_to_xyxy(boxes: np.ndarray, box_format: str) -> np.ndarray:
    boxes = boxes.astype(np.float32).copy()
    fmt = box_format.lower()
    if fmt == "auto":
        if len(boxes) == 0:
            fmt = "xyxy"
        else:
            xyxy_like = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).mean()
            fmt = "xyxy" if xyxy_like > 0.90 else "xywh"
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out
    raise ValueError(f"Unsupported pose box_format={box_format!r}; expected auto, xywh, or xyxy")


def _resolve_model_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    alternatives = []
    if candidate.suffix == ".engine":
        alternatives.append(candidate.with_suffix(".onnx"))
    if "yolo26s-pose" in candidate.name:
        alternatives.append(candidate.with_name(candidate.name.replace("yolo26s-pose", "yolo26-pose")))
        alternatives.append(candidate.with_name(candidate.name.replace("yolo26s-pose", "yolo26-pose")).with_suffix(".onnx"))

    for alt in alternatives:
        if alt.exists():
            return str(alt)
    return str(candidate)


class DryRunPoseInferencer:
    def __init__(self, mode: str = "none") -> None:
        self.mode = mode

    def infer(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        if self.mode == "none":
            return []
        h, w = frame_bgr.shape[:2]
        boxes = [(w * 0.35, h * 0.25, w * 0.60, h * 0.85)]
        if self.mode == "fight":
            boxes = [
                (w * 0.25, h * 0.25, w * 0.52, h * 0.85),
                (w * 0.45, h * 0.25, w * 0.72, h * 0.85),
            ]
        detections: List[PersonDetection] = []
        for box_i, bbox in enumerate(boxes):
            kpts = np.zeros((17, 3), dtype=np.float32)
            x1, y1, x2, y2 = bbox
            for idx in range(17):
                kpts[idx] = [
                    x1 + (x2 - x1) * (0.35 + 0.08 * (idx % 3)),
                    y1 + (y2 - y1) * (0.10 + 0.04 * idx),
                    0.9,
                ]
            detections.append(PersonDetection(bbox, 0.9, kpts))
        return detections


class DryRunActionInferencer:
    def __init__(self, labels: List[str], mode: str = "neutral") -> None:
        self.labels = labels
        self.mode = mode

    def infer(self, clips_rgb: np.ndarray) -> List[ActionScore]:
        out = []
        for _ in range(len(clips_rgb)):
            scores = {label: 0.02 for label in self.labels}
            scores["normal"] = 0.92
            if self.mode == "fall":
                scores.update({"normal": 0.05, "fall": 0.90})
            elif self.mode == "fight":
                scores.update({"normal": 0.05, "fight": 0.90})
            out.append(ActionScore(scores))
        return out


def build_pose_inferencer(config: PoseModelConfig, dry_run: bool, dry_mode: str):
    if dry_run:
        return DryRunPoseInferencer(dry_mode if dry_mode in {"synthetic", "fall", "fight"} else "none")
    return PoseInferencer(config)


def build_action_inferencer(config: ActionModelConfig, dry_run: bool, dry_mode: str):
    if dry_run:
        action_mode = dry_mode if dry_mode in {"fall", "fight"} else "neutral"
        return DryRunActionInferencer(config.labels, action_mode)
    return ActionInferencer(config)
