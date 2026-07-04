"""
Feature engineering for forehand tennis stroke analysis.

All angle computations are 2D projections from the camera plane and should be
treated as estimates only. True 3D angles require multi-camera triangulation.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import yaml

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REFERENCES_DIR

_REF_PATH = REFERENCES_DIR / "forehand_angle_reference.md"


# ---------------------------------------------------------------------------
# Reference file loading
# ---------------------------------------------------------------------------

def load_classification_ranges(ref_path: Path = _REF_PATH) -> dict:
    """
    Parse the yaml fenced block inside ``forehand_angle_reference.md`` and
    return the nested classification dict.

    Reads the file at call time so edits to the reference doc propagate without
    restarting Python.  Raises ValueError if no YAML block is found.
    """
    text = Path(ref_path).read_text(encoding="utf-8")
    match = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    if match is None:
        raise ValueError(f"No ```yaml``` block found in {ref_path}")
    return yaml.safe_load(match.group(1))


# ---------------------------------------------------------------------------
# Per-frame / per-swing angle computations
# ---------------------------------------------------------------------------

def compute_swing_path_angle(
    racket_positions: list[dict | None],
    contact_frame: int | None,
    lookback: int = 3,
) -> float | None:
    """
    Angle (degrees) of racket-head travel from *lookback* frames before contact
    to the contact frame, measured from horizontal.

    Positive = upward swing (low-to-high topspin motion).
    Returns None if either endpoint lacks a detected racket position.

    NOTE: 2D projection estimate — the measured angle depends on camera angle.
    """
    if contact_frame is None:
        return None
    pre_idx = max(0, contact_frame - lookback)
    pre = racket_positions[pre_idx]
    contact = racket_positions[contact_frame]
    if pre is None or contact is None:
        return None

    # Use top-centre of the bbox as the racket-head reference point
    px = pre["bbox"][0] + pre["bbox"][2] / 2
    py = pre["bbox"][1]
    cx = contact["bbox"][0] + contact["bbox"][2] / 2
    cy = contact["bbox"][1]

    dx = cx - px
    # Image y increases downward, so upward motion means cy < py → flip sign
    dy = py - cy
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def compute_racket_face_angle(
    racket_positions: list[dict | None],
    ball_positions: list[dict | None],
    contact_frame: int | None,
) -> float | None:
    """
    Angle (degrees) between the racket-face normal and the incoming ball
    trajectory at the contact frame.

    Positive = face closed relative to ball direction (topspin orientation).
    Returns None when racket orientation or ball trajectory cannot be determined.

    NOTE: 2D projection estimate — true face angle requires depth information.
    TODO: Implement once ``localize_racket`` returns reliable ``orientation_deg``
    values from a trained detector.
    """
    if contact_frame is None:
        return None
    racket = racket_positions[contact_frame]
    if racket is None or "orientation_deg" not in racket:
        return None

    # Approximate incoming ball direction from a few frames before contact
    pre_ball: dict | None = None
    for i in range(1, 4):
        idx = contact_frame - i
        if idx >= 0 and ball_positions[idx] is not None:
            pre_ball = ball_positions[idx]
            break
    contact_ball = ball_positions[contact_frame]
    if pre_ball is None or contact_ball is None:
        return None

    ball_dx = contact_ball["x"] - pre_ball["x"]
    ball_dy = pre_ball["y"] - contact_ball["y"]  # flip image y
    ball_angle = math.degrees(math.atan2(ball_dy, ball_dx))
    return racket["orientation_deg"] - ball_angle


def compute_contact_point_relative(
    keypoints: list[dict | None],
    racket_positions: list[dict | None],
    contact_frame: int | None,
    handedness: str = "right",
) -> dict | None:
    """
    Return racket-head centre offset (pixels) from the front hip and shoulder.

    "Front" is the hitting-arm side: right for right-handed players.
    Returns a dict with keys hip_dx, hip_dy, shoulder_dx, shoulder_dy where
    positive dy means the racket is above the landmark.

    Returns None if any required landmark is missing.

    NOTE: Pixel distances are not normalised for camera distance or focal length.
    """
    if contact_frame is None:
        return None
    kp = keypoints[contact_frame]
    racket = racket_positions[contact_frame]
    if kp is None or racket is None:
        return None

    side = "right" if handedness == "right" else "left"
    hip = kp.get(f"{side}_hip")
    shoulder = kp.get(f"{side}_shoulder")
    if hip is None or shoulder is None:
        return None

    rx = racket["bbox"][0] + racket["bbox"][2] / 2
    ry = racket["bbox"][1] + racket["bbox"][3] / 2

    return {
        "hip_dx": rx - hip["x"],
        "hip_dy": hip["y"] - ry,        # positive = racket above hip
        "shoulder_dx": rx - shoulder["x"],
        "shoulder_dy": shoulder["y"] - ry,  # positive = racket above shoulder
    }


# ---------------------------------------------------------------------------
# Shot classification
# ---------------------------------------------------------------------------

def classify_shot(
    swing_path_angle: float | None,
    racket_face_angle: float | None,
    ranges: dict,
) -> str:
    """
    Assign a shot label using ranges loaded from the reference doc.

    Uses nearest-centroid assignment across whichever angles are available.
    Returns "Unknown" when both angles are None.
    """
    if swing_path_angle is None and racket_face_angle is None:
        return "Unknown"

    classifications = ranges.get("classifications", {})
    best_label = "Unknown"
    best_distance = float("inf")

    for _key, cls in classifications.items():
        sp_range = cls.get("swing_path_angle_deg", {})
        rf_range = cls.get("racket_face_angle_deg", {})

        distances: list[float] = []
        if swing_path_angle is not None and sp_range:
            mid = (sp_range["min"] + sp_range["max"]) / 2
            distances.append(abs(swing_path_angle - mid))
        if racket_face_angle is not None and rf_range:
            mid = (rf_range["min"] + rf_range["max"]) / 2
            distances.append(abs(racket_face_angle - mid))

        if not distances:
            continue
        d = sum(distances) / len(distances)
        if d < best_distance:
            best_distance = d
            best_label = cls["label"]

    return best_label


# ---------------------------------------------------------------------------
# Main feature-computation entry point
# ---------------------------------------------------------------------------

def compute_features(
    interim_path: Path,
    ref_path: Path = _REF_PATH,
    handedness: str = "right",
) -> pd.DataFrame:
    """
    Load an interim JSON produced by ``dataset.process_video()``, compute all
    per-swing features, and persist to ``data/processed/<stem>_features.csv``.

    Returns a one-row DataFrame (one row per detected swing — currently the
    whole video is treated as a single swing until multi-swing segmentation is
    implemented).

    TODO: Segment video into individual swing episodes before calling this.
    """
    ranges = load_classification_ranges(ref_path)

    with open(interim_path, encoding="utf-8") as fh:
        data = json.load(fh)

    keypoints = data["keypoints"]
    racket_positions = data["racket_positions"]
    ball_positions = data["ball_positions"]
    contact_frame = data["contact_frame"]

    swing_path = compute_swing_path_angle(racket_positions, contact_frame)
    face_angle = compute_racket_face_angle(racket_positions, ball_positions, contact_frame)
    contact_rel = compute_contact_point_relative(
        keypoints, racket_positions, contact_frame, handedness
    )
    label = classify_shot(swing_path, face_angle, ranges)

    row: dict = {
        "video": data["video"],
        "contact_frame": contact_frame,
        "swing_path_angle_deg": swing_path,
        "racket_face_angle_deg": face_angle,
        "shot_label": label,
        "contact_hip_dx": contact_rel["hip_dx"] if contact_rel else None,
        "contact_hip_dy": contact_rel["hip_dy"] if contact_rel else None,
        "contact_shoulder_dx": contact_rel["shoulder_dx"] if contact_rel else None,
        "contact_shoulder_dy": contact_rel["shoulder_dy"] if contact_rel else None,
    }

    df = pd.DataFrame([row])
    stem = Path(data["video"]).stem
    out_path = PROCESSED_DATA_DIR / f"{stem}_features.csv"
    df.to_csv(out_path, index=False)
    return df
