"""
Frame extraction, pose estimation, racket localisation, and ball detection.

All pose/angle measurements produced here are 2D projections from the camera
plane. They are estimates only; true 3D angles require multi-camera triangulation.

MediaPipe compatibility note
-----------------------------
This module targets MediaPipe >= 0.10 (Tasks API).  The legacy ``mp.solutions``
namespace was removed in 0.10.x.  Pose landmarking via ``extract_pose_keypoints``
requires the ``pose_landmarker_lite.task`` model file to be placed in
``models/pose_landmarker_lite.task``.  Download it from the MediaPipe model
card on ai.google.dev (search: "Pose Landmarker").  The function stubs to None
per frame if the model file is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config import INTERIM_DATA_DIR, MODELS_DIR, RAW_DATA_DIR

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

# BlazePose 33-keypoint indices (stable across MediaPipe versions).
KEYPOINT_INDICES: dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

_POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"


def extract_frames(video_path: str | Path) -> list[np.ndarray]:
    """
    Read every frame from a video file and return as a list of BGR uint8 arrays.

    TODO: Add a ``stride`` parameter for high-frame-rate footage once real video
    confirms the required temporal resolution.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _build_pose_landmarker():
    """
    Construct a MediaPipe Tasks PoseLandmarker, or return None if the model
    file is missing.
    """
    if not _POSE_MODEL_PATH.exists():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
            num_poses=1,
        )
        return PoseLandmarker.create_from_options(options)
    except Exception:
        return None


def extract_pose_keypoints(frames: list[np.ndarray]) -> list[dict | None]:
    """
    Run MediaPipe Pose on each frame and return per-frame keypoint dicts.

    Each dict maps keypoint name -> {"x": float, "y": float, "visibility": float}
    in normalised image coordinates [0, 1].  Returns None for frames where pose
    detection fails or the model file is absent.

    NOTE: x/y are 2D image coordinates — depth is not captured.
    TODO: Place ``pose_landmarker_lite.task`` in ``models/`` to enable live
    detection (see module docstring for the download source).
    """
    import mediapipe as mp

    landmarker = _build_pose_landmarker()
    if landmarker is None:
        # Model file absent: return None stubs rather than crashing.
        return [None] * len(frames)

    results: list[dict | None] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = landmarker.detect(mp_image)
        if not detection.pose_landmarks:
            results.append(None)
            continue
        lm = detection.pose_landmarks[0]
        results.append(
            {
                name: {
                    "x": lm[idx].x,
                    "y": lm[idx].y,
                    "visibility": lm[idx].visibility,
                }
                for name, idx in KEYPOINT_INDICES.items()
            }
        )
    landmarker.close()
    return results


def localize_racket(frames: list[np.ndarray]) -> list[dict | None]:
    """
    Detect the racket head per frame.

    Returns a list of dicts with keys:
        ``"bbox"``: [x, y, w, h] in pixel coordinates
        ``"orientation_deg"``: estimated racket-face angle from horizontal (2D projection)
    or None when no racket is detected.

    TODO: Replace this stub with a fine-tuned object-detection model (e.g. YOLO
    or RT-DETR) once a labelled racket bounding-box dataset is assembled.
    Colour-based heuristics are too fragile across different court surfaces and
    racket colours.

    NOTE: ``orientation_deg`` will be a 2D projection estimate only.
    """
    # Stub: returns None for every frame until a detector is implemented.
    return [None] * len(frames)


def detect_ball(frames: list[np.ndarray]) -> list[dict | None]:
    """
    Detect the tennis ball per frame using Hough circle detection on an HSV
    yellow-green mask.

    Returns a list of dicts ``{"x": int, "y": int, "radius": int}`` in pixel
    coordinates, or None if no ball is found in that frame.

    TODO: Tune HSV thresholds and Hough parameters for specific court lighting.
    A learned detector (e.g. TrackNet) would give more robust trajectory data.
    """
    results: list[dict | None] = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Yellow-green range typical for pressurised tennis balls under daylight
        mask = cv2.inRange(hsv, (25, 80, 80), (65, 255, 255))
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=20,
        )
        if circles is not None:
            x, y, r = np.round(circles[0, 0]).astype(int)
            results.append({"x": int(x), "y": int(y), "radius": int(r)})
        else:
            results.append(None)
    return results


def find_contact_frame(
    ball_positions: list[dict | None],
    racket_positions: list[dict | None],
) -> int | None:
    """
    Estimate the contact frame as the first frame where the ball centre falls
    inside the racket bounding box.

    Returns None when either detector has no data (both are stubs for now).

    TODO: Add velocity-reversal detection on the ball trajectory as a more
    reliable contact signal once smooth ball tracks are available.
    """
    for i, (ball, racket) in enumerate(zip(ball_positions, racket_positions)):
        if ball is None or racket is None:
            continue
        bx, by = ball["x"], ball["y"]
        rx, ry, rw, rh = racket["bbox"]
        if rx <= bx <= rx + rw and ry <= by <= ry + rh:
            return i
    return None


def process_video(video_path: str | Path | None = None) -> Path:
    """
    Full pipeline for a single video:
        1. Extract frames
        2. Run pose estimation (requires ``models/pose_landmarker_lite.task``)
        3. Localise racket (stub — returns None per frame)
        4. Detect ball via Hough circles
        5. Identify contact frame

    Persists results to ``data/interim/<video_stem>.json`` and returns that path.

    If *video_path* is None, scans ``data/raw/`` for the first supported video
    file.  Raises ``FileNotFoundError`` if none is found.
    """
    if video_path is None:
        candidates = [
            p for p in RAW_DATA_DIR.iterdir()
            if p.suffix.lower() in _VIDEO_SUFFIXES
        ]
        if not candidates:
            raise FileNotFoundError(
                "No video files found in data/raw/. "
                "Add a video or pass an explicit path to process_video()."
            )
        video_path = candidates[0]

    video_path = Path(video_path)
    frames = extract_frames(video_path)

    keypoints = extract_pose_keypoints(frames)
    racket_positions = localize_racket(frames)
    ball_positions = detect_ball(frames)
    contact_frame = find_contact_frame(ball_positions, racket_positions)

    payload = {
        "video": str(video_path),
        "n_frames": len(frames),
        "contact_frame": contact_frame,
        "keypoints": keypoints,
        "racket_positions": racket_positions,
        "ball_positions": ball_positions,
    }

    out_path = INTERIM_DATA_DIR / f"{video_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return out_path
