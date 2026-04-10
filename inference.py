"""
Inference engine for GaitCNNv3Soft — 3-class fall-risk (LOW / MEDIUM / HIGH).

Key differences from the v6 binary engine:
  - WINDOW_SIZE = 90 frames (3 seconds at 30 fps)
  - 3 output classes: LOW=0, MEDIUM=1, HIGH=2
  - Global channel-wise z-score normalization using checkpoint's norm_mu / norm_sd
  - Threshold logic: P(HIGH) >= high_th → HIGH; else P(MEDIUM) >= med_th → MEDIUM; else LOW
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# YOLOv8-pose setup
# ---------------------------------------------------------------------------

_YOLO_MODEL_CACHE: Optional[YOLO] = None

# COCO skeleton connections for drawing
_YOLO_CONNECTIONS = [
    (0, 5), (0, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# COCO indices of keypoints required for fall-risk features
_REQUIRED_KP = [0, 5, 6, 11, 12, 13, 14, 15, 16]


def get_yolo_model() -> YOLO:
    """Load YOLOv8n-pose once and cache at module level."""
    global _YOLO_MODEL_CACHE
    if _YOLO_MODEL_CACHE is None:
        _YOLO_MODEL_CACHE = YOLO("yolov8n-pose.pt")
    return _YOLO_MODEL_CACHE


def detect_pose_yolo(model: YOLO, frame: np.ndarray) -> Optional[np.ndarray]:
    """Run YOLO pose on a BGR frame.
    Returns (17, 3) array [x_px, y_px, conf] for the first person, or None.
    """
    results = model(frame, verbose=False, device="cpu")
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints.data
    if len(kps) == 0:
        return None
    kp = kps[0].cpu().numpy()  # (17, 3)
    if any(kp[i, 2] < 0.3 for i in _REQUIRED_KP):
        return None
    return kp


def lm_to_dict(kp: np.ndarray, frame_w: int, frame_h: int) -> Dict[str, float]:
    """Convert YOLO COCO keypoint array (17, 3) to normalised landmark dict.
    COCO→feature mapping: nose=0, l_shoulder=5, r_shoulder=6,
    l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ankle=15, r_ankle=16.
    """
    def nx(i: int) -> float: return float(kp[i, 0]) / frame_w
    def ny(i: int) -> float: return float(kp[i, 1]) / frame_h
    return {
        "nose_x": nx(0),  "nose_y": ny(0),
        "ls_x":   nx(5),  "ls_y":   ny(5),
        "rs_x":   nx(6),  "rs_y":   ny(6),
        "lh_x":   nx(11), "lh_y":   ny(11),
        "rh_x":   nx(12), "rh_y":   ny(12),
        "lk_x":   nx(13), "lk_y":   ny(13),
        "rk_x":   nx(14), "rk_y":   ny(14),
        "la_x":   nx(15), "la_y":   ny(15),
        "ra_x":   nx(16), "ra_y":   ny(16),
    }


def draw_pose_landmarks(frame: np.ndarray, kp: np.ndarray) -> None:
    """Draw YOLO skeleton on a BGR frame in-place."""
    pts = []
    for i in range(17):
        x, y, conf = kp[i]
        pts.append((int(x), int(y)) if conf > 0.3 else None)
    for i, j in _YOLO_CONNECTIONS:
        if pts[i] and pts[j]:
            cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2, cv2.LINE_AA)
    for pt in pts:
        if pt:
            cv2.circle(frame, pt, 3, (0, 200, 0), -1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 90       # 3 seconds at 30 fps
WINDOW_STEP = 15       # stride for live / video (every 0.5 s)
MIN_DETECTIONS = 15    # out of 90 frames
SEQ_CHANNELS = 10      # position-only channels
INPUT_CHANNELS = 20    # 10 position + 10 velocity
N_CLASSES = 3

DEFAULT_HIGH_THRESHOLD = 0.64   # tuned on best fold (fold 5)
DEFAULT_MED_THRESHOLD = 0.33    # fixed during training
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

# ---------------------------------------------------------------------------
# Feature extraction (identical to train_occu_v3_soft.py)
# ---------------------------------------------------------------------------


def angle_at_b(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    ab = np.array([ax - bx, ay - by])
    cb = np.array([cx - bx, cy - by])
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))))


def lm_to_dict(lm) -> Dict[str, float]:
    return {
        "nose_x": lm[0].x,  "nose_y": lm[0].y,
        "ls_x": lm[11].x,   "ls_y": lm[11].y,
        "rs_x": lm[12].x,   "rs_y": lm[12].y,
        "lh_x": lm[23].x,   "lh_y": lm[23].y,
        "rh_x": lm[24].x,   "rh_y": lm[24].y,
        "lk_x": lm[25].x,   "lk_y": lm[25].y,
        "rk_x": lm[26].x,   "rk_y": lm[26].y,
        "la_x": lm[27].x,   "la_y": lm[27].y,
        "ra_x": lm[28].x,   "ra_y": lm[28].y,
    }


def extract_frame_vec(lm: Optional[Dict[str, float]]) -> Optional[List[float]]:
    if lm is None:
        return None
    com_x = (lm["lh_x"] + lm["rh_x"]) / 2
    com_y = (lm["lh_y"] + lm["rh_y"]) / 2
    l_ka = angle_at_b(lm["lh_x"], lm["lh_y"], lm["lk_x"], lm["lk_y"], lm["la_x"], lm["la_y"]) / 180.0
    r_ka = angle_at_b(lm["rh_x"], lm["rh_y"], lm["rk_x"], lm["rk_y"], lm["ra_x"], lm["ra_y"]) / 180.0
    trunk_tilt = (lm["ls_x"] + lm["rs_x"]) / 2 - com_x
    l_hip = angle_at_b(lm["ls_x"], lm["ls_y"], lm["lh_x"], lm["lh_y"], lm["lk_x"], lm["lk_y"]) / 180.0
    r_hip = angle_at_b(lm["rs_x"], lm["rs_y"], lm["rh_x"], lm["rh_y"], lm["rk_x"], lm["rk_y"]) / 180.0
    shoulder_tilt = lm["ls_y"] - lm["rs_y"]
    body_width = abs(lm["ls_x"] - lm["rs_x"]) + 1e-6
    lateral_foot_spread = abs(lm["la_x"] - lm["ra_x"]) / body_width
    s_mid_x = (lm["ls_x"] + lm["rs_x"]) / 2
    s_mid_y = (lm["ls_y"] + lm["rs_y"]) / 2
    trunk_len = abs(s_mid_y - com_y) + 1e-6
    body_lean = (s_mid_x - com_x) / trunk_len
    return [com_x, com_y, l_ka, r_ka, trunk_tilt, l_hip, r_hip, shoulder_tilt, lateral_foot_spread, body_lean]


def _add_velocity(pos_arr: np.ndarray) -> np.ndarray:
    """(T, 10) → (T, 20): concatenate position with frame-to-frame delta."""
    vel = np.concatenate(
        [np.zeros((1, pos_arr.shape[1]), dtype=np.float32),
         np.diff(pos_arr, axis=0).astype(np.float32)],
        axis=0,
    )
    return np.concatenate([pos_arr, vel], axis=1).astype(np.float32)


def build_window_tensor(
    window_lm: List[Optional[Dict]],
    norm_mu: np.ndarray,
    norm_sd: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Build a normalised (WINDOW_SIZE, 20) tensor from a window of landmark dicts.

    Parameters
    ----------
    window_lm : list of length WINDOW_SIZE, each item is a landmark dict or None
    norm_mu   : shape (1, 1, 20) — global channel means from training
    norm_sd   : shape (1, 1, 20) — global channel stds from training

    Returns
    -------
    np.ndarray of shape (WINDOW_SIZE, 20) or None if too few detections
    """
    vecs = [extract_frame_vec(lm) for lm in window_lm]
    valid = [v for v in vecs if v is not None]
    if len(valid) < MIN_DETECTIONS:
        return None
    mean_v = np.mean(valid, axis=0).astype(np.float32)
    pos = np.array([v if v is not None else mean_v for v in vecs], dtype=np.float32)  # (T, 10)
    feat = _add_velocity(pos)                                                          # (T, 20)
    # norm_mu/norm_sd are (1,1,20); reshape to (1,20) so broadcasting gives (T,20)
    mu = norm_mu.reshape(1, 20)
    sd = norm_sd.reshape(1, 20)
    feat = (feat - mu) / (sd + 1e-8)                                                  # z-score
    return feat.astype(np.float32)


def classify_probs(probs: np.ndarray, high_th: float, med_th: float) -> str:
    """Apply threshold logic: HIGH → MEDIUM → LOW."""
    if probs[2] >= high_th:
        return "HIGH"
    if probs[1] >= med_th:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Overlay drawing (3-class)
# ---------------------------------------------------------------------------

_RISK_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 140, 255),
    "LOW": (0, 190, 0),
    "WARMING UP": (180, 180, 180),
}


def _draw_overlay(
    frame: np.ndarray,
    risk_label: str,
    probs: Optional[np.ndarray],
    high_th: float,
    frame_idx: int,
) -> np.ndarray:
    color = _RISK_COLORS.get(risk_label, (180, 180, 180))
    lines = [f"Risk: {risk_label}"]
    if probs is not None:
        lines += [
            f"P(LOW):  {probs[0]:.3f}",
            f"P(MED):  {probs[1]:.3f}",
            f"P(HIGH): {probs[2]:.3f}",
        ]
    else:
        lines.append("P: --")
    lines += [f"Thr(H): {high_th:.2f}", f"Frame: {frame_idx}"]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 30
    return frame


# ---------------------------------------------------------------------------
# Model definition (must match train_occu_v3_soft.py exactly)
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class ResConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class GaitCNNv3Soft(nn.Module):
    """
    Input : (B, 90, 20) — 90 frames, 20 channels
    Output: (B, 3)      — logits [LOW, MEDIUM, HIGH]
    """
    def __init__(self, in_ch: int = INPUT_CHANNELS, n_classes: int = N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.enc1 = ResConv1d(32, 64, stride=2)
        self.se1 = SEBlock(64)
        self.enc2 = ResConv1d(64, 128, stride=2)
        self.se2 = SEBlock(128)
        self.enc3 = ResConv1d(128, 256, stride=1)
        self.se3 = SEBlock(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)   # (B, T, C) → (B, C, T)
        x = self.stem(x)
        x = self.se1(self.enc1(x))
        x = self.se2(self.enc2(x))
        x = self.se3(self.enc3(x))
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class FallRiskV3InferenceEngine:
    """
    Stateless inference engine for GaitCNNv3Soft.

    Loads the checkpoint once; norm_mu / norm_sd from the checkpoint are used
    for global channel-wise z-score normalisation at inference time.
    """

    def __init__(
        self,
        model_path: Path,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        med_threshold: float = DEFAULT_MED_THRESHOLD,
        aggregation: str = "p90",
        min_high_windows: int = 2,
    ):
        self.high_threshold = high_threshold
        self.med_threshold = med_threshold
        self.aggregation = aggregation
        self.min_high_windows = min_high_windows
        self.window_size = WINDOW_SIZE
        self.window_step = WINDOW_STEP
        self.score_history_len = 12

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=self.device)

        self.norm_mu = np.array(ckpt["norm_mu"], dtype=np.float32)  # (1, 1, 20)
        self.norm_sd = np.array(ckpt["norm_sd"], dtype=np.float32)  # (1, 1, 20)

        self.model = GaitCNNv3Soft().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _predict_window(self, arr: np.ndarray) -> np.ndarray:
        """(WINDOW_SIZE, 20) → softmax probs shape (3,)"""
        with torch.no_grad():
            logits = self.model(
                torch.tensor(arr[None], dtype=torch.float32, device=self.device)
            )
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    def _aggregate_high_scores(self, scores: List[float]) -> float:
        arr = np.array(scores, dtype=np.float32)
        if arr.size == 0:
            return -1.0
        if self.aggregation == "max":
            return float(np.max(arr))
        if self.aggregation == "p90":
            return float(np.percentile(arr, 90))
        if self.aggregation == "p75":
            return float(np.percentile(arr, 75))
        return float(np.mean(arr))

    def process_video(self, video_path: str) -> dict:
        """
        Analyse a video file frame-by-frame.
        Returns structured JSON-serialisable dict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        lm_buffer: Deque = deque(maxlen=self.window_size)
        high_score_history: List[float] = []
        window_results = []
        frame_count = 0
        detected_frames = 0
        t0 = time.time()

        model = get_yolo_model()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                kp = detect_pose_yolo(model, frame)
                lm_dict = lm_to_dict(kp, w, h) if kp is not None else None
                if lm_dict is not None:
                    detected_frames += 1
                lm_buffer.append(lm_dict)
                frame_count += 1

                if len(lm_buffer) == self.window_size and (frame_count % self.window_step == 0):
                    arr = build_window_tensor(list(lm_buffer), self.norm_mu, self.norm_sd)
                    if arr is not None:
                        probs = self._predict_window(arr)
                        high_score_history.append(float(probs[2]))
                        risk = classify_probs(probs, self.high_threshold, self.med_threshold)
                        window_results.append({
                            "frame_idx": frame_count,
                            "prob_low": round(float(probs[0]), 4),
                            "prob_medium": round(float(probs[1]), 4),
                            "prob_high": round(float(probs[2]), 4),
                            "risk": risk,
                        })
        finally:
            cap.release()
        elapsed = time.time() - t0

        if not high_score_history:
            return {
                "final_risk": "INSUFFICIENT_DATA",
                "aggregated_high_score": None,
                "high_threshold": self.high_threshold,
                "med_threshold": self.med_threshold,
                "window_results": [],
                "total_frames": frame_count,
                "windows_processed": 0,
                "detection_rate": 0.0,
                "processing_time_s": round(elapsed, 2),
            }

        agg_high = self._aggregate_high_scores(high_score_history)
        high_hits = sum(1 for s in high_score_history if s >= self.high_threshold)
        if agg_high >= self.high_threshold and high_hits >= self.min_high_windows:
            final_risk = "HIGH"
        else:
            # Check if MEDIUM is prevalent in results
            med_hits = sum(1 for w in window_results if w["risk"] == "MEDIUM")
            final_risk = "MEDIUM" if med_hits >= self.min_high_windows else "LOW"

        return {
            "final_risk": final_risk,
            "aggregated_high_score": round(agg_high, 4),
            "high_threshold": self.high_threshold,
            "med_threshold": self.med_threshold,
            "window_results": window_results,
            "total_frames": frame_count,
            "windows_processed": len(window_results),
            "detection_rate": round(detected_frames / max(1, frame_count), 3),
            "processing_time_s": round(elapsed, 2),
        }

    def process_video_annotated(
        self,
        video_path: str,
        output_path: str,
        show_pose: bool = True,
    ) -> dict:
        """
        Analyse video and write an annotated H.264 MP4 to output_path.
        Returns same dict as process_video().
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        lm_buffer: Deque = deque(maxlen=self.window_size)
        score_history: Deque[float] = deque(maxlen=self.score_history_len)
        window_results = []
        annotated_frames: List[np.ndarray] = []
        frame_count = 0
        detected_frames = 0
        t0 = time.time()

        current_risk = "WARMING UP"
        current_probs: Optional[np.ndarray] = None

        model = get_yolo_model()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                kp = detect_pose_yolo(model, frame)
                lm_dict = lm_to_dict(kp, w, h) if kp is not None else None
                if lm_dict is not None:
                    detected_frames += 1
                lm_buffer.append(lm_dict)
                frame_count += 1

                if show_pose and kp is not None:
                    draw_pose_landmarks(frame, kp)

                if len(lm_buffer) == self.window_size and (frame_count % self.window_step == 0):
                        arr = build_window_tensor(list(lm_buffer), self.norm_mu, self.norm_sd)
                        if arr is not None:
                            probs = self._predict_window(arr)
                            score_history.append(float(probs[2]))
                            current_probs = probs
                            risk = classify_probs(probs, self.high_threshold, self.med_threshold)
                            window_results.append({
                                "frame_idx": frame_count,
                                "prob_low": round(float(probs[0]), 4),
                                "prob_medium": round(float(probs[1]), 4),
                                "prob_high": round(float(probs[2]), 4),
                                "risk": risk,
                            })

                    if len(score_history) > 0:
                        agg = self._aggregate_high_scores(list(score_history))
                        high_hits = sum(1 for s in score_history if s >= self.high_threshold)
                        if agg >= self.high_threshold and high_hits >= self.min_high_windows:
                            current_risk = "HIGH"
                        elif current_probs is not None and current_probs[1] >= self.med_threshold:
                            current_risk = "MEDIUM"
                        else:
                            current_risk = "LOW"

                    frame = _draw_overlay(frame, current_risk, current_probs, self.high_threshold, frame_count)
                    annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        elapsed = time.time() - t0

        if annotated_frames:
            with imageio.get_writer(
                str(output_path),
                format="ffmpeg",
                fps=fps_in,
                codec="libx264",
                ffmpeg_params=["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"],
            ) as writer:
                for f in annotated_frames:
                    writer.append_data(f)

        if not window_results:
            return {
                "final_risk": "INSUFFICIENT_DATA",
                "aggregated_high_score": None,
                "high_threshold": self.high_threshold,
                "med_threshold": self.med_threshold,
                "window_results": [],
                "total_frames": frame_count,
                "windows_processed": 0,
                "detection_rate": 0.0,
                "processing_time_s": round(elapsed, 2),
            }

        agg_high = self._aggregate_high_scores(list(score_history))
        high_hits = sum(1 for s in score_history if s >= self.high_threshold)
        if agg_high >= self.high_threshold and high_hits >= self.min_high_windows:
            final_risk = "HIGH"
        else:
            med_hits = sum(1 for w in window_results if w["risk"] == "MEDIUM")
            final_risk = "MEDIUM" if med_hits >= self.min_high_windows else "LOW"

        return {
            "final_risk": final_risk,
            "aggregated_high_score": round(agg_high, 4),
            "high_threshold": self.high_threshold,
            "med_threshold": self.med_threshold,
            "window_results": window_results,
            "total_frames": frame_count,
            "windows_processed": len(window_results),
            "detection_rate": round(detected_frames / max(1, frame_count), 3),
            "processing_time_s": round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Stateful frame processor (webcam / streamlit-webrtc)
# ---------------------------------------------------------------------------


class FrameProcessorV3:
    """
    Stateful single-frame processor for real-time webcam use.
    Buffers 90 frames before making the first prediction (~3 s at 30 fps).
    """

    def __init__(
        self,
        model: GaitCNNv3Soft,
        device: str,
        norm_mu: np.ndarray,
        norm_sd: np.ndarray,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        med_threshold: float = DEFAULT_MED_THRESHOLD,
        window_size: int = WINDOW_SIZE,
        window_step: int = WINDOW_STEP,
        score_history_len: int = 12,
        aggregation: str = "p90",
        min_high_windows: int = 2,
    ):
        self.model = model
        self.device = device
        self.norm_mu = norm_mu
        self.norm_sd = norm_sd
        self.high_threshold = high_threshold
        self.med_threshold = med_threshold
        self.window_size = window_size
        self.window_step = window_step
        self.aggregation = aggregation
        self.min_high_windows = min_high_windows

        self.lm_buffer: Deque = deque(maxlen=window_size)
        self.score_history: Deque[float] = deque(maxlen=score_history_len)
        self.frame_count = 0
        self.current_risk = "WARMING UP"
        self.current_probs: Optional[np.ndarray] = None

    def update(self, lm_dict: Optional[Dict]) -> None:
        self.lm_buffer.append(lm_dict)
        self.frame_count += 1

        if len(self.lm_buffer) == self.window_size and (self.frame_count % self.window_step == 0):
            arr = build_window_tensor(list(self.lm_buffer), self.norm_mu, self.norm_sd)
            if arr is not None:
                with torch.no_grad():
                    logits = self.model(
                        torch.tensor(arr[None], dtype=torch.float32, device=self.device)
                    )
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                self.current_probs = probs
                self.score_history.append(float(probs[2]))

        if len(self.score_history) > 0:
            arr_s = np.array(list(self.score_history))
            if self.aggregation == "max":
                agg = float(np.max(arr_s))
            elif self.aggregation == "p90":
                agg = float(np.percentile(arr_s, 90))
            else:
                agg = float(np.mean(arr_s))
            high_hits = sum(1 for s in self.score_history if s >= self.high_threshold)
            if agg >= self.high_threshold and high_hits >= self.min_high_windows:
                self.current_risk = "HIGH"
            elif self.current_probs is not None and self.current_probs[1] >= self.med_threshold:
                self.current_risk = "MEDIUM"
            else:
                self.current_risk = "LOW"

    def reset(self) -> None:
        self.lm_buffer.clear()
        self.score_history.clear()
        self.frame_count = 0
        self.current_risk = "WARMING UP"
        self.current_probs = None
