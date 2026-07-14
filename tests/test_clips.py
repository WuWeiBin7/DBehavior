import numpy as np

from edge_app.clips import ClipBuilder
from edge_app.types import CandidateROI, FramePacket


def test_clip_builder_emits_after_enough_frames():
    builder = ClipBuilder(clip_length=4, sample_stride=2, crop_size=(32, 32))
    roi = CandidateROI("fall", "person", [1], (0, 0, 20, 20), 0.5)
    emitted = 0
    for i in range(8):
        frame = np.full((40, 40, 3), i, dtype=np.uint8)
        rois, clips = builder.add(FramePacket(i, float(i), frame, "test"), [roi])
        if len(rois):
            emitted += 1
            assert clips.shape == (1, 4, 32, 32, 3)
    assert emitted == 1


def test_clip_builder_preserves_max_geometry_hint_over_window():
    builder = ClipBuilder(clip_length=4, sample_stride=1, crop_size=(16, 16))
    ready = []
    for i, hint in enumerate([0.05, 0.40, 0.10, 0.08]):
        roi = CandidateROI("fall", "person", [1], (0, 0, 20, 20), hint)
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        ready, _ = builder.add(FramePacket(i, float(i), frame, "test"), [roi])

    assert ready[0].features["temporal_hint_max"] == 0.40
    assert ready[0].features["temporal_hint_mean"] == 0.1575
