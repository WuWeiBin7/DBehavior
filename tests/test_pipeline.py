import numpy as np

from edge_app.config import AppConfig, BehaviorEventConfig, EventsConfig
from edge_app.pipeline import BehaviorPipeline
from edge_app.types import FramePacket


def test_pipeline_dry_run_processes_without_weights(tmp_path):
    cfg = AppConfig()
    cfg.runtime.dry_run_mode = "none"
    cfg.events = EventsConfig(output_dir=str(tmp_path))
    pipe = BehaviorPipeline(cfg, "synthetic.mp4", dry_run_models=True)
    for i in range(5):
        packet = FramePacket(i, float(i), np.zeros((64, 64, 3), dtype=np.uint8), "synthetic.mp4")
        written = pipe.process_packet(packet)
        assert written == []


def test_pipeline_synthetic_fight_can_confirm(tmp_path):
    cfg = AppConfig()
    cfg.runtime.dry_run_mode = "fight"
    cfg.models.action.clip_length = 2
    cfg.models.action.sample_stride = 1
    cfg.models.action.infer_interval_frames = 1
    cfg.tracking.min_hits = 1
    cfg.events = EventsConfig(
        output_dir=str(tmp_path),
        evidence_post_sec=0.0,
        fight=BehaviorEventConfig(
            suspect_threshold=0.1,
            confirm_threshold=0.1,
            confirm_clip_count=2,
            cooldown_sec=1.0,
            weights={"model": 1.0},
        ),
    )
    pipe = BehaviorPipeline(cfg, "synthetic.mp4", dry_run_models=True)
    written_total = []
    for i in range(6):
        packet = FramePacket(i, float(i), np.zeros((128, 128, 3), dtype=np.uint8), "synthetic.mp4")
        written_total.extend(pipe.process_packet(packet))
    written_total.extend(pipe.flush_pending())
    assert any(r.event_type == "fight" for r in written_total)
