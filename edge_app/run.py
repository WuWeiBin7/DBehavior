from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run edge behavior recognition pipeline.")
    parser.add_argument("--config", default="configs/edge_orin.yaml", help="Path to YAML config.")
    parser.add_argument("--source", default="", help="RTSP URL or local video path. Overrides input.rtsp_url.")
    parser.add_argument("--dry-run-models", action="store_true", help="Run without model weights.")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means no limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from edge_app.config import load_config
        from edge_app.pipeline import BehaviorPipeline
    except ImportError as exc:
        print(
            f"Missing runtime dependency: {exc}. Install with: python -m pip install -r requirements-edge.txt",
            file=sys.stderr,
        )
        return 3

    try:
        config = load_config(args.config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.max_frames:
        config.runtime.max_frames = args.max_frames
    source = args.source or config.input.rtsp_url
    if not source:
        print("No source provided. Use --source rtsp://... or set input.rtsp_url in config.", file=sys.stderr)
        return 2
    pipeline = BehaviorPipeline(config, source=source, dry_run_models=args.dry_run_models, save_vis=args.save_vis)
    if pipeline.action_degraded:
        print(
            f"Warning: action model fallback enabled. Reason: {pipeline.action_degraded_reason}",
            file=sys.stderr,
        )
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
