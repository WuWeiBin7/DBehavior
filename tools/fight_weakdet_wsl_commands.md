# Fight Weak Detection In WSL

## Files prepared

- Converter: `tools/fight_video_to_coco_weakdet.py`
- Config: `Swin-Transformer-Object-Detection/configs/swin/mask_reppointsv2_swin_tiny_patch4_window7_fight_weak_3x.py`
- Pretrained weight: `mask_reppointsv2_swin_tiny_patch4_window7_3x.pth`

## Important note

This dataset only contains temporal fight segments per video, not detection boxes.
The converter therefore builds a weak detection dataset:

- positive frames: one full-image pseudo box with class `fight`
- negative frames: empty annotation

This is enough to run the Swin detection repo, but the learned detector is weakly supervised.

## 1. Go to workspace

```bash
cd /root/WWB/DBehavior
```

## 2. Install OpenCV for frame extraction

```bash
python3 -m pip install opencv-python-headless
```

## 3. Optional dry run

```bash
python3 tools/fight_video_to_coco_weakdet.py \
  --src-root /root/WWB/DBehavior/dataset/打架/dataset \
  --gt-json /root/WWB/DBehavior/dataset/打架/dataset/ground-truth.json \
  --out-root /root/WWB/DBehavior/dataset/fight_det_weak \
  --dry-run
```

## 4. Convert videos to weak COCO detection dataset

```bash
python3 tools/fight_video_to_coco_weakdet.py \
  --src-root /root/WWB/DBehavior/dataset/打架/dataset \
  --gt-json /root/WWB/DBehavior/dataset/打架/dataset/ground-truth.json \
  --out-root /root/WWB/DBehavior/dataset/fight_det_weak \
  --positive-fps 1.0 \
  --negative-fps 0.25 \
  --negative-margin 0.5
```

## 5. Train

```bash
cd /root/WWB/DBehavior/Swin-Transformer-Object-Detection

python3 tools/train.py \
  configs/swin/mask_reppointsv2_swin_tiny_patch4_window7_fight_weak_3x.py \
  --work-dir /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny \
  --cfg-options model.pretrained=/root/WWB/DBehavior/mask_reppointsv2_swin_tiny_patch4_window7_3x.pth
```

## 6. Validate on val split

```bash
python3 tools/test.py \
  configs/swin/mask_reppointsv2_swin_tiny_patch4_window7_fight_weak_3x.py \
  /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny/latest.pth \
  --eval bbox \
  --cfg-options \
  data.test.ann_file=/root/WWB/DBehavior/dataset/fight_det_weak/annotations/val.json \
  data.test.img_prefix=/root/WWB/DBehavior/dataset/fight_det_weak/images/ \
  data.test.classes="('fight',)"
```

## 7. Test on test split

```bash
python3 tools/test.py \
  configs/swin/mask_reppointsv2_swin_tiny_patch4_window7_fight_weak_3x.py \
  /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny/latest.pth \
  --eval bbox
```

## 8. Export prediction visualizations on test split

```bash
mkdir -p /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny/test_vis

python3 tools/test.py \
  configs/swin/mask_reppointsv2_swin_tiny_patch4_window7_fight_weak_3x.py \
  /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny/latest.pth \
  --show-score-thr 0.3 \
  --show-dir /root/WWB/DBehavior/work_dirs/fight_weak_reppointsv2_swin_tiny/test_vis
```

## Expected dry-run frame counts

Using the current defaults:

- train: `12801` positive, `5443` negative
- val: `5671` positive, `2846` negative
- test: `5934` positive, `2735` negative
