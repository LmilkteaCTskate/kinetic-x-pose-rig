"""
Kinematics & Biomechanical Telemetry Engine
Computes joint angles, posture classification, and skeleton topology for
COCO-17 (YOLO-Pose) and MediaPipe-33 pose models.
"""

from __future__ import annotations

import math
from typing import Any

# COCO 17 keypoint names (YOLO11-Pose / YOLOv8-Pose)
COCO_KEYPOINT_NAMES = [
    "NOSE", "L_EYE", "R_EYE", "L_EAR", "R_EAR",
    "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
    "L_WRIST", "R_WRIST", "L_HIP", "R_HIP",
    "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE",
]

# Connections: (start_idx, end_idx, group)
COCO_SKELETON = [
    (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"),
    (5, 6, "torso"), (5, 11, "torso"), (6, 12, "torso"), (11, 12, "torso"),
    (5, 7, "left_arm"), (7, 9, "left_arm"),
    (6, 8, "right_arm"), (8, 10, "right_arm"),
    (11, 13, "left_leg"), (13, 15, "left_leg"),
    (12, 14, "right_leg"), (14, 16, "right_leg"),
]

# MediaPipe 33 keypoint names (subset mapped to canonical indices for angles)
MEDIAPIPE_KEYPOINT_NAMES = [
    "NOSE", "L_EYE_IN", "L_EYE", "L_EYE_OUT", "R_EYE_IN", "R_EYE", "R_EYE_OUT",
    "L_EAR", "R_EAR", "MOUTH_L", "MOUTH_R",
    "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST",
    "L_PINKY", "R_PINKY", "L_INDEX", "R_INDEX", "L_THUMB", "R_THUMB",
    "L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE",
    "L_HEEL", "R_HEEL", "L_FOOT", "R_FOOT",
]

MEDIAPIPE_SKELETON = [
    (0, 2, "head"), (0, 5, "head"), (2, 7, "head"), (5, 8, "head"),
    (11, 12, "torso"), (11, 23, "torso"), (12, 24, "torso"), (23, 24, "torso"),
    (11, 13, "left_arm"), (13, 15, "left_arm"), (15, 19, "left_arm"),
    (12, 14, "right_arm"), (14, 16, "right_arm"), (16, 20, "right_arm"),
    (23, 25, "left_leg"), (25, 27, "left_leg"), (27, 29, "left_leg"), (27, 31, "left_leg"),
    (24, 26, "right_leg"), (26, 28, "right_leg"), (28, 30, "right_leg"), (28, 32, "right_leg"),
]

# MediaPipe 21 Hand Landmark Names
HAND_KEYPOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Hand skeleton edges: (start_idx, end_idx, group)
HAND_SKELETON = [
    (0, 1, "thumb"), (1, 2, "thumb"), (2, 3, "thumb"), (3, 4, "thumb"),
    (0, 5, "palm"), (5, 9, "palm"), (9, 13, "palm"), (13, 17, "palm"), (0, 17, "palm"),
    (5, 6, "index"), (6, 7, "index"), (7, 8, "index"),
    (9, 10, "middle"), (10, 11, "middle"), (11, 12, "middle"),
    (13, 14, "ring"), (14, 15, "ring"), (15, 16, "ring"),
    (17, 18, "pinky"), (18, 19, "pinky"), (19, 20, "pinky"),
]

# Canonical joint index map for biomechanical angle calculations
JOINT_INDEX_MAP = {
    "coco17": {
        "l_shoulder": 5, "r_shoulder": 6,
        "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10,
        "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14,
        "l_ankle": 15, "r_ankle": 16,
    },
    "mediapipe33": {
        "l_shoulder": 11, "r_shoulder": 12,
        "l_elbow": 13, "r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16,
        "l_hip": 23, "r_hip": 24,
        "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28,
    },
}


def _angle_between(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Calculate angle ABC in degrees (0-180)."""
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0
    cos_val = (ba_x * bc_x + ba_y * bc_y) / (mag_ba * mag_bc)
    cos_val = max(-1.0, min(1.0, cos_val))
    return round(math.degrees(math.acos(cos_val)), 1)


def compute_kinematics(
    keypoints: list[dict[str, Any]],
    topology: str = "coco17",
    min_conf: float = 0.35,
) -> dict[str, Any]:
    """
    Compute biomechanical angles and posture classification for a single detected person.
    Each keypoint dict has normalized x, y in [0, 1] and conf in [0, 1].
    """
    idx_map = JOINT_INDEX_MAP.get(topology, JOINT_INDEX_MAP["coco17"])

    def pt(name: str) -> tuple[float, float] | None:
        i = idx_map[name]
        if i >= len(keypoints):
            return None
        kp = keypoints[i]
        if kp["conf"] < min_conf:
            return None
        return (kp["x"], kp["y"])

    def calc(a_name: str, b_name: str, c_name: str) -> float | None:
        a, b, c = pt(a_name), pt(b_name), pt(c_name)
        if a is None or b is None or c is None:
            return None
        return _angle_between(a, b, c)

    angles: dict[str, float | None] = {
        "L_ELBOW": calc("l_shoulder", "l_elbow", "l_wrist"),
        "R_ELBOW": calc("r_shoulder", "r_elbow", "r_wrist"),
        "L_SHOULDER": calc("l_hip", "l_shoulder", "l_elbow"),
        "R_SHOULDER": calc("r_hip", "r_shoulder", "r_elbow"),
        "L_KNEE": calc("l_hip", "l_knee", "l_ankle"),
        "R_KNEE": calc("r_hip", "r_knee", "r_ankle"),
        "L_HIP": calc("l_shoulder", "l_hip", "l_knee"),
        "R_HIP": calc("r_shoulder", "r_hip", "r_knee"),
    }

    # Spine inclination from vertical (0° = perfectly upright)
    ls, rs = pt("l_shoulder"), pt("r_shoulder")
    lh, rh = pt("l_hip"), pt("r_hip")
    spine_tilt: float | None = None
    if ls and rs and lh and rh:
        mid_sh = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
        mid_hp = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)
        dx = mid_sh[0] - mid_hp[0]
        dy = mid_hp[1] - mid_sh[1]  # positive when shoulders above hips
        spine_tilt = round(abs(math.degrees(math.atan2(dx, max(abs(dy), 1e-6)))), 1)
    angles["SPINE_TILT"] = spine_tilt

    # Posture classification
    posture = _classify_posture(keypoints, idx_map, angles, min_conf)

    # Active keypoints ratio
    valid_count = sum(1 for kp in keypoints if kp["conf"] >= min_conf)
    integrity = round((valid_count / max(len(keypoints), 1)) * 100.0, 1)

    return {
        "angles": angles,
        "posture": posture,
        "integrity_pct": integrity,
        "valid_keypoints": valid_count,
        "total_keypoints": len(keypoints),
    }


def _classify_posture(
    keypoints: list[dict[str, Any]],
    idx_map: dict[str, int],
    angles: dict[str, float | None],
    min_conf: float,
) -> str:
    def pt(name: str) -> tuple[float, float] | None:
        i = idx_map[name]
        if i >= len(keypoints) or keypoints[i]["conf"] < min_conf:
            return None
        return (keypoints[i]["x"], keypoints[i]["y"])

    lw, rw = pt("l_wrist"), pt("r_wrist")
    ls, rs = pt("l_shoulder"), pt("r_shoulder")

    l_knee, r_knee = angles.get("L_KNEE"), angles.get("R_KNEE")
    l_sh, r_sh = angles.get("L_SHOULDER"), angles.get("R_SHOULDER")
    l_el, r_el = angles.get("L_ELBOW"), angles.get("R_ELBOW")
    spine = angles.get("SPINE_TILT")

    # Arms raised above shoulders (lower y = higher on screen)
    if lw and rw and ls and rs and lw[1] < ls[1] - 0.05 and rw[1] < rs[1] - 0.05:
        return "ARMS_RAISED"

    # T-Pose detection
    if (
        l_sh is not None and r_sh is not None
        and 70 <= l_sh <= 115 and 70 <= r_sh <= 115
        and (l_el is None or l_el >= 145)
        and (r_el is None or r_el >= 145)
    ):
        return "T_POSE_CALIB"

    # Squatting / knee flexion
    if (l_knee is not None and l_knee < 125) and (r_knee is not None and r_knee < 125):
        return "SQUAT_FLEX"

    # Torso tilt
    if spine is not None and spine > 20.0:
        return "TORSO_INCLINED"

    # Default nominal standing if core joints visible
    if ls and rs:
        return "NOMINAL_UPRIGHT"

    return "PARTIAL_LOCK"


def compute_hand_kinematics(keypoints: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute finger extension states, pinch distance, and industrial gesture
    classification for a 21-keypoint MediaPipe hand.
    """
    if len(keypoints) < 21:
        return {
            "gesture": "PARTIAL_HAND",
            "extended_count": 0,
            "pinch_ratio": 1.0,
            "fingers": {},
        }

    def dist(i: int, j: int) -> float:
        return math.hypot(
            keypoints[i]["x"] - keypoints[j]["x"],
            keypoints[i]["y"] - keypoints[j]["y"],
        )

    palm_scale = max(dist(0, 9), 1e-4)

    # Distance-from-wrist test works regardless of hand rotation
    thumb_ext = dist(4, 17) > dist(3, 17) * 1.08 and dist(4, 5) > palm_scale * 0.32
    index_ext = dist(8, 0) > dist(6, 0) * 1.12
    middle_ext = dist(12, 0) > dist(10, 0) * 1.12
    ring_ext = dist(16, 0) > dist(14, 0) * 1.12
    pinky_ext = dist(20, 0) > dist(18, 0) * 1.12

    fingers = {
        "THUMB": thumb_ext,
        "INDEX": index_ext,
        "MIDDLE": middle_ext,
        "RING": ring_ext,
        "PINKY": pinky_ext,
    }
    ext_count = sum(1 for v in fingers.values() if v)
    pinch_ratio = round(dist(4, 8) / palm_scale, 2)

    if pinch_ratio < 0.28 and (middle_ext or ring_ext or pinky_ext):
        gesture = "PINCH_GRIP"
    elif ext_count == 0:
        gesture = "FIST_LOCK"
    elif ext_count == 5:
        gesture = "OPEN_PALM"
    elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        gesture = "THUMBS_UP"
    elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
        gesture = "POINT_INDEX"
    elif index_ext and middle_ext and not ring_ext and not pinky_ext:
        gesture = "VICTORY_V"
    elif ext_count == 3:
        gesture = "THREE_FINGER"
    elif ext_count == 4:
        gesture = "FOUR_FINGER"
    else:
        gesture = "CUSTOM_GRIP"

    return {
        "gesture": gesture,
        "extended_count": ext_count,
        "pinch_ratio": pinch_ratio,
        "fingers": fingers,
    }

