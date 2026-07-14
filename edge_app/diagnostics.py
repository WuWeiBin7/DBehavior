from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List

from edge_app.config import RuntimeConfig
from edge_app.types import ActionScore, CandidateROI, EventCandidate, EventRecord


class PipelineDiagnostics:
    def __init__(self, config: RuntimeConfig, source: str) -> None:
        self.enabled = config.diagnostics_enabled
        self.source = source
        self.summary_path = Path(config.diagnostics_path)
        self.scores_path = Path(config.scores_jsonl_path)
        self.log_interval = max(0, config.log_interval_frames)
        self.counts: Counter = Counter()
        self.prob_sum: DefaultDict[str, float] = defaultdict(float)
        self.prob_max: DefaultDict[str, float] = defaultdict(float)
        self.fused_max: DefaultDict[str, float] = defaultdict(float)
        if self.enabled:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.scores_path.parent.mkdir(parents=True, exist_ok=True)
            self.scores_path.unlink(missing_ok=True)

    def record_frame(self, frame_id: int, detections: int, tracks: int, rois: List[CandidateROI]) -> None:
        if not self.enabled:
            return
        self.counts["frames"] += 1
        self.counts["detections"] += detections
        self.counts["tracks"] += tracks
        self.counts["fall_rois"] += sum(roi.event_type == "fall" for roi in rois)
        self.counts["fight_rois"] += sum(roi.event_type == "fight" for roi in rois)
        if self.log_interval and self.counts["frames"] % self.log_interval == 0:
            print(
                "[edge] "
                f"frame={frame_id} det={detections} tracks={tracks} rois={len(rois)} "
                f"clips={self.counts['ready_clips']} max_fight={self.prob_max['fight']:.3f} "
                f"max_fused_fight={self.fused_max['fight']:.3f}"
            )

    def record_inference(
        self,
        frame_id: int,
        timestamp: float,
        rois: List[CandidateROI],
        scores: List[ActionScore],
        fused: List[EventCandidate],
        states: Dict[str, str],
        suppressions: List[Dict] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.counts["action_batches"] += int(bool(scores))
        self.counts["ready_clips"] += len(scores)
        if suppressions:
            self.counts["cross_event_suppressions"] += len(suppressions)
        fused_by_key = {(item.event_type, tuple(sorted(item.track_ids))): item for item in fused}
        for item in fused:
            self.fused_max[item.event_type] = max(self.fused_max[item.event_type], item.score)
        lines = []
        for roi, action in zip(rois, scores):
            for label, probability in action.scores.items():
                self.prob_sum[label] += probability
                self.prob_max[label] = max(self.prob_max[label], probability)
            candidate = fused_by_key.get((roi.event_type, tuple(sorted(roi.track_ids))))
            fused_score = candidate.score if candidate else None
            same_tracks = {
                item.event_type: item.score
                for item in fused
                if tuple(sorted(item.track_ids)) == tuple(sorted(roi.track_ids))
            }
            fusion_features = {
                item.event_type: item.features
                for item in fused
                if tuple(sorted(item.track_ids)) == tuple(sorted(roi.track_ids))
            }
            lines.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "event_type": roi.event_type,
                    "track_ids": roi.track_ids,
                    "bbox_xyxy": roi.bbox_xyxy,
                    "roi_hint": roi.score_hint,
                    "features": roi.features,
                    "probabilities": action.scores,
                    "fused_score": fused_score,
                    "fused_scores": same_tracks,
                    "fusion_features": fusion_features,
                    "states": states,
                    "event_suppressions": suppressions or [],
                }
            )
        if lines:
            with self.scores_path.open("a", encoding="utf-8") as stream:
                for line in lines:
                    stream.write(json.dumps(line, ensure_ascii=False) + "\n")

    def record_confirmed(self, records: Iterable[EventRecord]) -> None:
        if not self.enabled:
            return
        for record in records:
            self.counts[f"confirmed_{record.event_type}"] += 1

    def finish(self, degraded: bool = False, degraded_reason: str = "") -> Dict:
        if not self.enabled:
            return {}
        clip_count = max(1, self.counts["ready_clips"])
        payload = {
            "source": self.source,
            "action_degraded": degraded,
            "action_degraded_reason": degraded_reason,
            "counts": dict(self.counts),
            "action_probability_max": dict(self.prob_max),
            "action_probability_mean": {
                label: value / clip_count for label, value in self.prob_sum.items()
            },
            "fused_score_max": dict(self.fused_max),
            "scores_jsonl_path": str(self.scores_path),
        }
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[edge] diagnostics written to {self.summary_path}")
        return payload
