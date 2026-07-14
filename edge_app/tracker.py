from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from edge_app.config import TrackingConfig
from edge_app.types import PersonDetection, Track


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    def __init__(self) -> None:
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class KalmanFilter:
    """Kalman filter from ByteTrack, adapted to avoid scipy dependency."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(mean) == 0:
            return mean, covariance
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.asarray([np.diag(s) for s in sqr])
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = np.linalg.multi_dot((covariance, self._update_mat.T, np.linalg.inv(projected_cov)))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: Sequence[float], score: float, keypoints: np.ndarray | None = None) -> None:
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.score = float(score)
        self.keypoints = np.asarray(keypoints, dtype=np.float32) if keypoints is not None else np.zeros((17, 3), dtype=np.float32)
        self.tracklet_len = 0
        self.hits = 0
        self.history: List[Tuple[float, Tuple[float, float, float, float], np.ndarray]] = []

    def predict(self) -> None:
        if self.mean is None or self.covariance is None or self.kalman_filter is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: Sequence["STrack"]) -> None:
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks if st.mean is not None])
        multi_covariance = np.asarray([st.covariance for st in stracks if st.covariance is not None])
        if len(multi_mean) == 0:
            return
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int, timestamp: float) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.hits = 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._append_history(timestamp)

    def re_activate(self, new_track: "STrack", frame_id: int, timestamp: float, new_id: bool = False) -> None:
        assert self.kalman_filter is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.hits += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.keypoints = new_track.keypoints
        self._append_history(timestamp)

    def update(self, new_track: "STrack", frame_id: int, timestamp: float) -> None:
        assert self.kalman_filter is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.hits += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.keypoints = new_track.keypoints
        self._append_history(timestamp)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: Sequence[float]) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(1e-6, ret[3])
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: Sequence[float]) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    def _append_history(self, timestamp: float) -> None:
        self.history.append((timestamp, tuple(float(v) for v in self.tlbr), self.keypoints.copy()))
        if len(self.history) > 120:
            self.history = self.history[-120:]


class ByteTrack:
    """Official ByteTrack-style tracker adapted for pose detections."""

    def __init__(self, config: TrackingConfig, frame_rate: int = 30) -> None:
        self.config = config
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self.det_thresh = config.track_high_thresh
        self.buffer_size = int(frame_rate / 30.0 * config.max_age)
        self.max_time_lost = max(1, self.buffer_size)
        self.kalman_filter = KalmanFilter()

    def update(self, detections: List[PersonDetection], timestamp: float) -> List[Track]:
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        scores = np.asarray([det.score for det in detections], dtype=np.float32)
        bboxes = np.asarray([det.bbox_xyxy for det in detections], dtype=np.float32)
        keypoints = [det.keypoints for det in detections]

        remain_inds = scores > self.config.track_high_thresh
        inds_low = scores > self.config.track_low_thresh
        inds_high = scores < self.config.track_high_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds] if len(bboxes) else np.empty((0, 4), dtype=np.float32)
        scores_keep = scores[remain_inds] if len(scores) else np.empty((0,), dtype=np.float32)
        kpts_keep = [kp for kp, keep in zip(keypoints, remain_inds) if keep]

        dets_second = bboxes[inds_second] if len(bboxes) else np.empty((0, 4), dtype=np.float32)
        scores_second = scores[inds_second] if len(scores) else np.empty((0,), dtype=np.float32)
        kpts_second = [kp for kp, keep in zip(keypoints, inds_second) if keep]

        detections_high = [
            STrack(STrack.tlbr_to_tlwh(tlbr), score, kp)
            for tlbr, score, kp in zip(dets, scores_keep, kpts_keep)
        ]

        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = fuse_score(iou_distance(strack_pool, detections_high), detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.config.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[int(itracked)]
            det = detections_high[int(idet)]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, timestamp)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, timestamp, new_id=False)
                refind_stracks.append(track)

        detections_second = [
            STrack(STrack.tlbr_to_tlwh(tlbr), score, kp)
            for tlbr, score, kp in zip(dets_second, scores_second, kpts_second)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[int(itracked)]
            det = detections_second[int(idet)]
            track.update(det, self.frame_id, timestamp)
            activated_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[int(it)]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections_unmatched = [detections_high[i] for i in u_detection]
        dists = fuse_score(iou_distance(unconfirmed, detections_unmatched), detections_unmatched)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[int(itracked)].update(detections_unmatched[int(idet)], self.frame_id, timestamp)
            activated_stracks.append(unconfirmed[int(itracked)])
        for it in u_unconfirmed:
            track = unconfirmed[int(it)]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections_unmatched[int(inew)]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, timestamp)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [
            track
            for track in self.tracked_stracks
            if track.is_activated and track.hits >= self.config.min_hits and track.state == TrackState.Tracked
        ]
        return [_to_external_track(track) for track in output_stracks]


def _to_external_track(track: STrack) -> Track:
    return Track(
        track_id=track.track_id,
        bbox_xyxy=tuple(float(v) for v in track.tlbr),
        score=float(track.score),
        keypoints=track.keypoints.copy(),
        age=max(0, track.frame_id - track.start_frame),
        hits=track.hits,
        missed=0,
        history=list(track.history),
    )


def joint_stracks(tlista: Sequence[STrack], tlistb: Sequence[STrack]) -> List[STrack]:
    exists: Dict[int, bool] = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = True
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, False):
            exists[t.track_id] = True
            res.append(t)
    return res


def sub_stracks(tlista: Sequence[STrack], tlistb: Sequence[STrack]) -> List[STrack]:
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: Sequence[STrack], stracksb: Sequence[STrack]) -> tuple[List[STrack], List[STrack]]:
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


def iou_distance(atracks: Sequence[STrack], btracks: Sequence[STrack]) -> np.ndarray:
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    return 1.0 - bbox_ious(atlbrs, btlbrs)


def bbox_ious(a_boxes: Sequence[Sequence[float]], b_boxes: Sequence[Sequence[float]]) -> np.ndarray:
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    a = np.asarray(a_boxes, dtype=np.float32)
    b = np.asarray(b_boxes, dtype=np.float32)
    ious = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box_a in enumerate(a):
        ax1, ay1, ax2, ay2 = box_a
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j, box_b in enumerate(b):
            bx1, by1, bx2, by2 = box_b
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            denom = a_area + b_area - inter
            ious[i, j] = 0.0 if denom <= 0 else inter / denom
    return ious


def fuse_score(cost_matrix: np.ndarray, detections: Sequence[STrack]) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.asarray([det.score for det in detections], dtype=np.float32)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    return 1.0 - iou_sim * det_scores


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        return np.asarray(matches, dtype=int), np.where(x < 0)[0], np.where(y < 0)[0]
    except ImportError:
        return greedy_assignment(cost_matrix, thresh)


def greedy_assignment(cost_matrix: np.ndarray, thresh: float):
    matches = []
    used_rows = set()
    used_cols = set()
    rows, cols = cost_matrix.shape
    candidates = [(cost_matrix[r, c], r, c) for r in range(rows) for c in range(cols) if cost_matrix[r, c] <= thresh]
    for _, r, c in sorted(candidates, key=lambda item: item[0]):
        if r in used_rows or c in used_cols:
            continue
        matches.append([r, c])
        used_rows.add(r)
        used_cols.add(c)
    unmatched_rows = tuple(r for r in range(rows) if r not in used_rows)
    unmatched_cols = tuple(c for c in range(cols) if c not in used_cols)
    return np.asarray(matches, dtype=int), unmatched_rows, unmatched_cols


ByteTrackLite = ByteTrack
