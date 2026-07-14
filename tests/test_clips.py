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
