# Fight Frame Extraction

This script does two things:

1. Extracts all frames from the videos and stores them by split:
   - `out_root/frames/train/<video_name>/...`
   - `out_root/frames/val/<video_name>/...`
   - `out_root/frames/test/<video_name>/...`
2. Collects all frames that fall inside `Fight` time segments into one flat folder:
   - `out_root/fight_frames/`

## WSL command

```bash
cd /root/WWB/DBehavior
python3 -m pip install opencv-python-headless

python3 tools/extract_fight_frames_for_annotation.py \
  --src-root /root/WWB/DBehavior/dataset/打架/dataset \
  --gt-json /root/WWB/DBehavior/dataset/打架/dataset/ground-truth.json \
  --out-root /root/WWB/DBehavior/dataset/打架_frames
```

## Notes

- The dataset keeps the original `training / validation / testing` split from `ground-truth.json`.
- By default, every frame inside a fight segment is copied to `fight_frames`.
- If the folder becomes too large, you can set `--fight-only-fps 1.0` to downsample fight frames.
