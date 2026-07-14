import json
import numpy as np

from edge_app.buffer import RingFrameBuffer
from edge_app.config import BehaviorEventConfig, EventsConfig
from edge_app.events import EventStateMachine
from edge_app.output import EventWriter
from edge_app.types import EventCandidate, FramePacket


def test_fight_state_machine_confirms_after_clip_count():
    cfg = EventsConfig(
        fight=BehaviorEventConfig(
            suspect_threshold=0.5,
            confirm_threshold=0.7,
            confirm_clip_count=2,
            cooldown_sec=1.0,
            weights={},
        )
    )
    sm = EventStateMachine(cfg, "test.mp4")
    assert sm.update([EventCandidate("fight", 0.8, [1, 2], (0, 0, 10, 10), 1.0)]) == []
    records = sm.update([EventCandidate("fight", 0.85, [1, 2], (0, 0, 10, 10), 1.5)])
    assert len(records) == 1
    assert records[0].event_type == "fight"
    assert records[0].state == "CONFIRMED"


def test_event_writer_outputs_json_snapshot_and_video(tmp_path):
    cfg = EventsConfig(output_dir=str(tmp_path), evidence_pre_sec=1.0, evidence_post_sec=1.0)
    writer = EventWriter(cfg, fps=5)
    buffer = RingFrameBuffer(max_seconds=3, fps=5)
    for i in range(5):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        buffer.append(FramePacket(i, float(i) * 0.2, frame, "test"))

    record = EventStateMachine(cfg, "test").update(
        [
            EventCandidate("fight", 0.8, [1, 2], (2, 2, 20, 20), 1.0),
            EventCandidate("fight", 0.9, [1, 2], (2, 2, 20, 20), 1.5),
        ]
    )
    assert record == []
    sm = EventStateMachine(
        EventsConfig(
            output_dir=str(tmp_path),
            fight=BehaviorEventConfig(suspect_threshold=0.5, confirm_threshold=0.7, confirm_clip_count=2),
        ),
        "test",
    )
    sm.update([EventCandidate("fight", 0.8, [1, 2], (2, 2, 20, 20), 1.0)])
    records = sm.update([EventCandidate("fight", 0.9, [1, 2], (2, 2, 20, 20), 1.5)])
    packet = FramePacket(99, 1.5, np.zeros((32, 32, 3), dtype=np.uint8), "test")
    written = writer.write(records[0], packet, buffer)
    assert written.snapshot_path is not None
    assert written.video_path is not None
    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "fight"
    assert payload["state"] == "CONFIRMED"


def test_recent_fight_suppresses_weak_fall_for_same_track():
    cfg = EventsConfig(
        cross_event_suppression_sec=10.0,
        fall_geometry_override_threshold=0.35,
        fight=BehaviorEventConfig(
            suspect_threshold=0.5,
            confirm_threshold=0.5,
            confirm_clip_count=2,
        ),
        fall=BehaviorEventConfig(
            suspect_threshold=0.5,
            confirm_threshold=0.5,
            confirm_duration_sec=0.5,
        ),
    )
    sm = EventStateMachine(cfg, "test")
    sm.update([EventCandidate("fight", 0.9, [10], (0, 0, 100, 100), 0.0)])
    fight = sm.update([EventCandidate("fight", 0.9, [10], (0, 0, 100, 100), 0.2)])
    assert [record.event_type for record in fight] == ["fight"]

    weak_fall = EventCandidate(
        "fall", 0.9, [10], (0, 0, 100, 100), 1.0, {"fall_geometry": 0.10}
    )
    assert sm.update([weak_fall]) == []
    assert sm.snapshot()["fall"] == "NORMAL"
    assert sm.suppression_snapshot()[0]["reason"] == "recent_fight_same_incident"


def test_strong_fall_geometry_can_override_recent_fight():
    cfg = EventsConfig(
        cross_event_suppression_sec=10.0,
        fall_geometry_override_threshold=0.35,
        fight=BehaviorEventConfig(
            suspect_threshold=0.5,
            confirm_threshold=0.5,
            confirm_clip_count=2,
        ),
        fall=BehaviorEventConfig(
            suspect_threshold=0.5,
            confirm_threshold=0.5,
            confirm_duration_sec=0.5,
        ),
    )
    sm = EventStateMachine(cfg, "test")
    sm.update([EventCandidate("fight", 0.9, [10], (0, 0, 100, 100), 0.0)])
    sm.update([EventCandidate("fight", 0.9, [10], (0, 0, 100, 100), 0.2)])
    strong = {"fall_geometry": 0.60}
    sm.update([EventCandidate("fall", 0.9, [10], (0, 0, 100, 100), 1.0, strong)])
    records = sm.update([EventCandidate("fall", 0.9, [10], (0, 0, 100, 100), 1.6, strong)])

    assert [record.event_type for record in records] == ["fall"]
