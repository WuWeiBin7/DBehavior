import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_MAP = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert fight video temporal annotations to a weak COCO detection dataset."
    )
    parser.add_argument(
        "--src-root",
        default="dataset/打架/dataset",
        help="Root directory that contains CCTV_DATA/NON_CCTV_DATA and ground-truth.json.",
    )
    parser.add_argument(
        "--gt-json",
        default="dataset/打架/dataset/ground-truth.json",
        help="Path to ground-truth.json.",
    )
    parser.add_argument(
        "--out-root",
        default="dataset/fight_det_weak",
        help="Output directory for extracted images and COCO annotations.",
    )
    parser.add_argument(
        "--positive-fps",
        type=float,
        default=1.0,
        help="Sampling rate for fight segments in frames per second.",
    )
    parser.add_argument(
        "--negative-fps",
        type=float,
        default=0.25,
        help="Sampling rate for non-fight segments in frames per second.",
    )
    parser.add_argument(
        "--negative-margin",
        type=float,
        default=0.5,
        help="Seconds to exclude around fight boundaries when sampling negatives.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Only process the first N videos. 0 means process all videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print expected split counts without extracting frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images/annotations.",
    )
    return parser.parse_args()


def load_database(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["database"]


def index_videos(src_root):
    indexed = {}
    for path in Path(src_root).rglob("*.mpeg"):
        indexed[path.stem] = path
    return indexed


def merge_segments(segments):
    if not segments:
        return []
    segments = sorted((float(a), float(b)) for a, b in segments if b > a)
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def expand_and_clip_segments(segments, duration, margin):
    expanded = []
    for start, end in segments:
        expanded.append((max(0.0, start - margin), min(duration, end + margin)))
    return merge_segments(expanded)


def invert_segments(segments, duration):
    if not segments:
        return [(0.0, duration)]
    intervals = []
    cursor = 0.0
    for start, end in segments:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        intervals.append((cursor, duration))
    return intervals


def sample_times(intervals, fps):
    if fps <= 0:
        return []
    step = 1.0 / fps
    sampled = []
    for start, end in intervals:
        if end <= start:
            continue
        t = start
        while t <= end + 1e-6:
            sampled.append(t)
            t += step
        if not sampled or sampled[-1] < start or sampled[-1] > end:
            sampled.append((start + end) / 2.0)
    return sampled


def sample_frame_indices(video_info, positive_fps, negative_fps, negative_margin):
    duration = float(video_info["duration"])
    frame_rate = float(video_info["frame_rate"])
    nb_frames = int(video_info["nb_frames"])
    positive_segments = merge_segments(
        [tuple(ann["segment"]) for ann in video_info.get("annotations", [])]
    )
    negative_segments = invert_segments(
        expand_and_clip_segments(positive_segments, duration, negative_margin), duration
    )

    pos_times = sample_times(positive_segments, positive_fps)
    neg_times = sample_times(negative_segments, negative_fps)

    labels = {}
    for t in neg_times:
        idx = min(max(int(round(t * frame_rate)), 0), max(nb_frames - 1, 0))
        labels[idx] = "negative"
    for t in pos_times:
        idx = min(max(int(round(t * frame_rate)), 0), max(nb_frames - 1, 0))
        labels[idx] = "positive"
    return labels


def ensure_output_dirs(out_root):
    out_root = Path(out_root)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)


def coco_stub():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fight"}],
    }


def full_frame_annotation(image_id, ann_id, width, height):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [0.0, 0.0, float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [
            [0.0, 0.0, float(width - 1), 0.0, float(width - 1), float(height - 1), 0.0, float(height - 1)]
        ],
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    gt_json = Path(args.gt_json)
    out_root = Path(args.out_root)

    database = load_database(gt_json)
    video_index = index_videos(src_root)
    video_items = sorted(database.items())
    if args.limit_videos > 0:
        video_items = video_items[:args.limit_videos]

    if not args.dry_run:
        import cv2

        ensure_output_dirs(out_root)

    split_coco = {split: coco_stub() for split in ("train", "val", "test")}
    split_stats = defaultdict(Counter)
    missing_videos = []
    image_id = 1
    ann_id = 1

    for video_name, video_info in video_items:
        split = SPLIT_MAP[video_info["subset"]]
        video_path = video_index.get(video_name)
        if video_path is None:
            missing_videos.append(video_name)
            continue

        labels = sample_frame_indices(
            video_info,
            positive_fps=args.positive_fps,
            negative_fps=args.negative_fps,
            negative_margin=args.negative_margin,
        )
        split_stats[split]["videos"] += 1
        split_stats[split]["positive_frames_expected"] += sum(
            1 for label in labels.values() if label == "positive"
        )
        split_stats[split]["negative_frames_expected"] += sum(
            1 for label in labels.values() if label == "negative"
        )

        if args.dry_run:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            missing_videos.append(video_name)
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_dir = out_root / "images" / split
        target_indices = set(labels.keys())
        current_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if current_idx not in target_indices:
                current_idx += 1
                continue

            label = labels[current_idx]
            file_name = f"{video_name}_{current_idx:06d}.jpg"
            rel_path = f"{split}/{file_name}"
            img_path = output_dir / file_name
            if img_path.exists() and not args.overwrite:
                pass
            else:
                cv2.imwrite(
                    str(img_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
                )

            split_coco[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": rel_path,
                    "width": width,
                    "height": height,
                }
            )
            if label == "positive":
                split_coco[split]["annotations"].append(
                    full_frame_annotation(image_id, ann_id, width, height)
                )
                ann_id += 1
                split_stats[split]["positive_frames_written"] += 1
            else:
                split_stats[split]["negative_frames_written"] += 1
            image_id += 1
            current_idx += 1

        cap.release()

    report = {
        "source_root": str(src_root),
        "gt_json": str(gt_json),
        "out_root": str(out_root),
        "positive_fps": args.positive_fps,
        "negative_fps": args.negative_fps,
        "negative_margin": args.negative_margin,
        "limit_videos": args.limit_videos,
        "dry_run": args.dry_run,
        "missing_videos": missing_videos,
        "split_stats": {k: dict(v) for k, v in split_stats.items()},
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    for split, payload in split_coco.items():
        ann_path = out_root / "annotations" / f"{split}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    report_path = out_root / "annotations" / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
