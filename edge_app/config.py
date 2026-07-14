from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class InputConfig:
    rtsp_url: str = ""
    width: int = 1280
    height: int = 720
    target_fps: int = 15
    reconnect_sec: float = 3.0
    use_gstreamer: bool = True


@dataclass
class PoseModelConfig:
    path: str = "weights/yolo26s-pose.engine"
    format: str = "ultralytics_pose"
    box_format: str = "auto"
    input_size: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 50


@dataclass
class ActionModelConfig:
    path: str = "weights/video_swin_tiny.engine"
    input_size: Tuple[int, int] = (224, 224)
    clip_length: int = 32
    sample_stride: int = 1
    infer_interval_frames: int = 4
    labels: List[str] = field(default_factory=lambda: ["normal", "fight", "fall"])


@dataclass
class ModelsConfig:
    pose: PoseModelConfig = field(default_factory=PoseModelConfig)
    action: ActionModelConfig = field(default_factory=ActionModelConfig)


@dataclass
class TrackingConfig:
    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.10
    match_thresh: float = 0.30
    max_age: int = 30
    min_hits: int = 2


@dataclass
class CandidateConfig:
    fall_min_hint: float = 0.20
    fall_min_history: int = 4
    fall_roi_scale: float = 1.45
    fight_min_hint: float = 0.10
    fight_min_proximity: float = 0.10
    fight_roi_scale: float = 1.45


@dataclass
class BehaviorEventConfig:
    suspect_threshold: float = 0.55
    confirm_threshold: float = 0.70
    confirm_duration_sec: float = 1.2
    confirm_clip_count: int = 2
    cooldown_sec: float = 8.0
    candidate_gap_sec: float = 1.0
    single_person_fallback: bool = False
    single_person_model_threshold: float = 0.85
    model_geometry_gate_threshold: float = 0.0
    model_geometry_gate_floor: float = 1.0
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class EventsConfig:
    evidence_pre_sec: float = 5.0
    evidence_post_sec: float = 5.0
    output_dir: str = "outputs/edge_events"
    snapshot_dir: str = "snapshots"
    video_dir: str = "videos"
    jsonl_name: str = "events.jsonl"
    cross_event_suppression_sec: float = 0.0
    cross_event_iou_threshold: float = 0.30
    fall_geometry_override_threshold: float = 1.0
    fall: BehaviorEventConfig = field(default_factory=BehaviorEventConfig)
    fight: BehaviorEventConfig = field(default_factory=lambda: BehaviorEventConfig(cooldown_sec=10.0))


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    fp16: bool = True
    dry_run_mode: str = "none"
    save_vis: bool = False
    vis_path: str = "outputs/edge_events/vis.mp4"
    max_frames: int = 0
    allow_action_fallback: bool = False
    diagnostics_enabled: bool = False
    diagnostics_path: str = "outputs/edge_events/diagnostics.json"
    scores_jsonl_path: str = "outputs/edge_events/action_scores.jsonl"
    log_interval_frames: int = 30


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _tuple2(value) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected a two-item size, got {value!r}")


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files. Install with: pip install -r requirements-edge.txt") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    input_cfg = InputConfig(**data.get("input", {}))

    pose_raw = data.get("models", {}).get("pose", {})
    action_raw = data.get("models", {}).get("action", {})
    if "input_size" in pose_raw:
        pose_raw["input_size"] = _tuple2(pose_raw["input_size"])
    if "input_size" in action_raw:
        action_raw["input_size"] = _tuple2(action_raw["input_size"])
    models_cfg = ModelsConfig(
        pose=PoseModelConfig(**pose_raw),
        action=ActionModelConfig(**action_raw),
    )

    tracking_cfg = TrackingConfig(**data.get("tracking", {}))
    candidates_cfg = CandidateConfig(**data.get("candidates", {}))

    events_raw = data.get("events", {})
    fall_raw = events_raw.pop("fall", {}) if "fall" in events_raw else {}
    fight_raw = events_raw.pop("fight", {}) if "fight" in events_raw else {}
    events_cfg = EventsConfig(
        **events_raw,
        fall=BehaviorEventConfig(**fall_raw),
        fight=BehaviorEventConfig(**fight_raw),
    )

    runtime_cfg = RuntimeConfig(**data.get("runtime", {}))
    return AppConfig(
        input=input_cfg,
        models=models_cfg,
        tracking=tracking_cfg,
        candidates=candidates_cfg,
        events=events_cfg,
        runtime=runtime_cfg,
    )
