import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all video frames by split and collect fight-segment frames into one folder."
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
        default="dataset/打架_frames",
        help="Output directory.",
    )
    parser.add_argument(
        "--fight-only-fps",
        type=float,
        default=0.0,
        help="If > 0, downsample fight frames to this fps. Default 0 means keep every frame in fight segments.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep one frame every N frames for both split folders and fight frame export. Default is 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Only process the first N videos. 0 means all videos.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N processed videos.",
    )
    return parser.parse_args()


def load_database(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["database"]


def index_videos(src_root):
    indexed = {}
    for ext in ("*.mpeg", "*.mp4", "*.avi", "*.mov", "*.mkv"):
        for path in Path(src_root).rglob(ext):
            indexed[path.stem] = path
    return indexed


def merge_segments(segments):
    merged = []
    for start, end in sorted((float(a), float(b)) for a, b in segments if b > a):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def belongs_to_segments(t, segments):
    for start, end in segments:
        if start <= t <= end:
            return True
        if t < start:
            return False
    return False


def ensure_dirs(out_root):
    out_root = Path(out_root)
    for split in ("train", "val", "test"):
        (out_root / "frames" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "fight_frames").mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)


def save_jpeg(path, frame, cv2, quality=95):
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def main():
    args = parse_args()

    import cv2

    src_root = Path(args.src_root)
    gt_json = Path(args.gt_json)
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    database = load_database(gt_json)
    video_index = index_videos(src_root)
    items = sorted(database.items())
    if args.limit_videos > 0:
        items = items[: args.limit_videos]

    split_map = {"training": "train", "validation": "val", "testing": "test"}
    manifest_rows = []
    summary = {
        "videos": 0,
        "frames_written": {"train": 0, "val": 0, "test": 0},
        "fight_frames_written": 0,
        "missing_videos": [],
    }

    total_videos = len(items)

    for video_idx, (video_name, info) in enumerate(items, start=1):
        video_path = video_index.get(video_name)
        if video_path is None:
            summary["missing_videos"].append(video_name)
            continue

        split = split_map[info["subset"]]
        split_dir = out_root / "frames" / split / video_name
        split_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            summary["missing_videos"].append(video_name)
            continue

        fps = float(info["frame_rate"])
        fight_segments = merge_segments([tuple(ann["segment"]) for ann in info.get("annotations", [])])
        fight_out_dir = out_root / "fight_frames"
        fight_fps_step = None if args.fight_only_fps <= 0 else 1.0 / args.fight_only_fps
        next_fight_sample_time = 0.0

        frame_idx = 0
        written_this_video = 0
        fight_written_this_video = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / fps if fps > 0 else 0.0
            keep_frame = (frame_idx % args.frame_step) == 0
            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = split_dir / frame_name
            if keep_frame:
                if not frame_path.exists() or args.overwrite:
                    save_jpeg(frame_path, frame, cv2)
                written_this_video += 1

            if keep_frame and belongs_to_segments(timestamp, fight_segments):
                should_write = True
                if fight_fps_step is not None:
                    should_write = timestamp + 1e-9 >= next_fight_sample_time
                    if should_write:
                        next_fight_sample_time = timestamp + fight_fps_step

                if should_write:
                    fight_name = f"{split}__{video_name}__frame_{frame_idx:06d}.jpg"
                    fight_path = fight_out_dir / fight_name
                    if not fight_path.exists() or args.overwrite:
                        save_jpeg(fight_path, frame, cv2)
                    fight_written_this_video += 1
                    manifest_rows.append(
                        {
                            "file_name": fight_name,
                            "split": split,
                            "video": video_name,
                            "frame_idx": frame_idx,
                            "timestamp": round(timestamp, 6),
                        }
                    )

            frame_idx += 1

        cap.release()
        summary["videos"] += 1
        summary["frames_written"][split] += written_this_video
        summary["fight_frames_written"] += fight_written_this_video

        if args.progress_every > 0 and (
            video_idx % args.progress_every == 0 or video_idx == total_videos
        ):
            print(
                json.dumps(
                    {
                        "processed_videos": video_idx,
                        "total_videos": total_videos,
                        "frames_written": summary["frames_written"],
                        "fight_frames_written": summary["fight_frames_written"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest_path = out_root / "annotations" / "fight_frames_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "split", "video", "frame_idx", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = out_root / "annotations" / "extract_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
