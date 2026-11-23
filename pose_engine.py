"""
Unified Human Pose & Hand Estimation Engine (Server & Local Adaptive)
Automatically detects host hardware (NVIDIA CUDA / Apple MPS / Multi-Core CPU)
and selects optimal inference device, tensor resolution, and network stream profile.
"""

from __future__ import annotations

import os
import platform
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from kinematics import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    HAND_KEYPOINT_NAMES,
    HAND_SKELETON,
    MEDIAPIPE_KEYPOINT_NAMES,
    MEDIAPIPE_SKELETON,
    compute_hand_kinematics,
    compute_kinematics,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _check_mediapipe_pose_available() -> bool:
    try:
        import mediapipe as mp  # noqa: F401
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except Exception:
        return False


def _ensure_hand_model_file() -> bool:
    """Ensure hand_landmarker.task exists; auto-download on fresh server deploys."""
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 100_000:
        return True
    try:
        print(f"[KINETIC-X] Downloading MediaPipe hand model -> {HAND_MODEL_PATH}")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        return HAND_MODEL_PATH.exists()
    except Exception as exc:
        print(f"[KINETIC-X] Warning: Could not download hand_landmarker.task: {exc}")
        return False


def _check_hand_engine_available() -> bool:
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return _ensure_hand_model_file()
    except Exception:
        return False


def _detect_hardware_profile() -> dict[str, Any]:
    """
    Inspect host environment (CUDA GPU, Apple MPS, CPU cores, RAM) and
    return optimal device + performance preset.
    Supports override via env KINETIC_DEVICE=auto|cuda|cpu.
    """
    forced = os.environ.get("KINETIC_DEVICE", "auto").strip().lower()
    cpu_cores = os.cpu_count() or 2
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    cpu_name = platform.processor() or platform.machine() or "CPU"

    cuda_ok = torch.cuda.is_available() and forced != "cpu"
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and forced not in ("cpu", "cuda")
    )

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        try:
            _, total_b = torch.cuda.mem_get_info(0)
            vram_total_mb = round(total_b / (1024 * 1024), 1)
        except Exception:
            vram_total_mb = 0.0

        if vram_total_mb >= 7500:
            tier = "CUDA_HIGH"
            rec_imgsz = 640
            rec_stream = {"capture_width": 640, "jpeg_quality": 0.78, "target_fps": 30}
        else:
            tier = "CUDA_STANDARD"
            rec_imgsz = 480
            rec_stream = {"capture_width": 480, "jpeg_quality": 0.72, "target_fps": 25}

        return {
            "device": "cuda:0",
            "compute_mode": "CUDA",
            "tier": tier,
            "cuda_available": True,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "cpu_name": cpu_name,
            "recommended_imgsz": rec_imgsz,
            "recommended_stream": rec_stream,
        }

    if mps_ok:
        return {
            "device": "mps",
            "compute_mode": "MPS",
            "tier": "APPLE_MPS",
            "cuda_available": False,
            "gpu_name": f"Apple MPS ({cpu_cores}C)",
            "vram_total_mb": 0.0,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
            "cpu_name": cpu_name,
            "recommended_imgsz": 480,
            "recommended_stream": {"capture_width": 480, "jpeg_quality": 0.70, "target_fps": 20},
        }

    # CPU-only server / workstation profile
    optimal_threads = max(1, min(cpu_cores, 8))
    try:
        torch.set_num_threads(optimal_threads)
    except Exception:
        pass

    if cpu_cores >= 6:
        tier = "CPU_MULTI"
        rec_imgsz = 320
        rec_stream = {"capture_width": 384, "jpeg_quality": 0.65, "target_fps": 15}
    else:
        tier = "CPU_LITE"
        rec_imgsz = 320
        rec_stream = {"capture_width": 320, "jpeg_quality": 0.60, "target_fps": 12}

    return {
        "device": "cpu",
        "compute_mode": "CPU",
        "tier": tier,
        "cuda_available": False,
        "gpu_name": f"CPU AUTO ({cpu_cores}C / {ram_gb}GB)",
        "vram_total_mb": 0.0,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "cpu_name": cpu_name,
        "recommended_imgsz": rec_imgsz,
        "recommended_stream": rec_stream,
    }


class PoseEngine:
    """Thread-safe hardware-adaptive pose & hand estimation engine."""

    AVAILABLE_ENGINES = {
        "yolo11n-pose": {
            "label": "YOLO11-NANO POSE (FAST)",
            "topology": "coco17",
            "keypoints": 17,
            "multi_person": True,
            "backend": "ultralytics",
            "weights": "yolo11n-pose.pt",
        },
        "yolo11s-pose": {
            "label": "YOLO11-SMALL POSE (PRECISION)",
            "topology": "coco17",
            "keypoints": 17,
            "multi_person": True,
            "backend": "ultralytics",
            "weights": "yolo11s-pose.pt",
        },
        "mediapipe": {
            "label": "MEDIAPIPE 33-PT (CPU)",
            "topology": "mediapipe33",
            "keypoints": 33,
            "multi_person": False,
            "backend": "mediapipe",
            "weights": None,
        },
    }

    def __init__(self, default_engine: str = "yolo11n-pose") -> None:
        self._lock = threading.Lock()
        self.profile = _detect_hardware_profile()
        self.device: str = self.profile["device"]
        self.compute_mode: str = self.profile["compute_mode"]
        self.cuda_available: bool = self.profile["cuda_available"]
        self.gpu_name: str = self.profile["gpu_name"]
        self.mediapipe_ready = _check_mediapipe_pose_available()
        self.hand_ready = _check_hand_engine_available()

        self._active_engine_name = default_engine
        self._yolo_models: dict[str, Any] = {}
        self._mp_pose: Any = None
        self._hand_landmarker: Any = None

    def get_system_info(self) -> dict[str, Any]:
        vram_used_mb = 0.0
        vram_total_mb = self.profile["vram_total_mb"]
        if self.cuda_available:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                vram_total_mb = round(total_bytes / (1024 * 1024), 1)
                vram_used_mb = round((total_bytes - free_bytes) / (1024 * 1024), 1)
            except Exception:
                vram_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)

        engines_meta = {}
        for k, v in self.AVAILABLE_ENGINES.items():
            available = True if v["backend"] == "ultralytics" else self.mediapipe_ready
            engines_meta[k] = {**v, "available": available}

        return {
            "device": self.device,
            "compute_mode": self.compute_mode,
            "hardware_tier": self.profile["tier"],
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "cpu_cores": self.profile["cpu_cores"],
            "ram_gb": self.profile["ram_gb"],
            "cpu_load_pct": psutil.cpu_percent(interval=None),
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "recommended_imgsz": self.profile["recommended_imgsz"],
            "recommended_stream": self.profile["recommended_stream"],
            "active_engine": self._active_engine_name,
            "hand_ready": self.hand_ready,
            "engines": engines_meta,
            "skeletons": {
                "coco17": {
                    "names": COCO_KEYPOINT_NAMES,
                    "edges": COCO_SKELETON,
                },
                "mediapipe33": {
                    "names": MEDIAPIPE_KEYPOINT_NAMES,
                    "edges": MEDIAPIPE_SKELETON,
                },
                "hand21": {
                    "names": HAND_KEYPOINT_NAMES,
                    "edges": HAND_SKELETON,
                },
            },
        }

    def _ensure_yolo(self, engine_name: str) -> Any:
        if engine_name in self._yolo_models:
            return self._yolo_models[engine_name]

        from ultralytics import YOLO

        weights_name = self.AVAILABLE_ENGINES[engine_name]["weights"]
        weights_path = MODELS_DIR / weights_name
        model = YOLO(str(weights_path))
        model.to(self.device)

        warmup_sz = self.profile["recommended_imgsz"]
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        model.predict(dummy, device=self.device, imgsz=warmup_sz, verbose=False)

        self._yolo_models[engine_name] = model
        return model

    def _ensure_mediapipe(self) -> Any:
        if self._mp_pose is not None:
            return self._mp_pose
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        return self._mp_pose

    def set_engine(self, engine_name: str) -> dict[str, Any]:
        if engine_name not in self.AVAILABLE_ENGINES:
            raise ValueError(f"Unknown engine: {engine_name}")
        if engine_name == "mediapipe" and not self.mediapipe_ready:
            raise RuntimeError("MediaPipe Pose solutions API is not available in this environment.")

        with self._lock:
            self._active_engine_name = engine_name
            if self.AVAILABLE_ENGINES[engine_name]["backend"] == "ultralytics":
                self._ensure_yolo(engine_name)
            else:
                self._ensure_mediapipe()
        return self.get_system_info()

    def _ensure_hand_landmarker(self) -> Any:
        if self._hand_landmarker is not None:
            return self._hand_landmarker
        if not self.hand_ready:
            return None
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Load via model_asset_buffer to support non-ASCII paths across Windows & Linux
        model_bytes = HAND_MODEL_PATH.read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)
        return self._hand_landmarker

    def preload_default(self) -> None:
        print(
            f"[KINETIC-X] Hardware Profile: {self.profile['tier']} | "
            f"Device: {self.device} ({self.gpu_name}) | "
            f"Rec Tensor: {self.profile['recommended_imgsz']}px"
        )
        with self._lock:
            if self.AVAILABLE_ENGINES[self._active_engine_name]["backend"] == "ultralytics":
                self._ensure_yolo(self._active_engine_name)
            elif self.mediapipe_ready:
                self._ensure_mediapipe()
            if self.hand_ready:
                self._ensure_hand_landmarker()

    def infer(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float = 0.35,
        engine_name: str | None = None,
        imgsz: int | None = None,
        enable_pose: bool = True,
        enable_hands: bool = True,
    ) -> dict[str, Any]:
        """Run body pose and/or hand landmark inference on a BGR numpy image."""
        t0 = time.perf_counter()
        target_engine = engine_name or self._active_engine_name
        if target_engine not in self.AVAILABLE_ENGINES:
            target_engine = self._active_engine_name
        if target_engine == "mediapipe" and not self.mediapipe_ready:
            target_engine = "yolo11n-pose"

        use_imgsz = imgsz or self.profile["recommended_imgsz"]
        h, w = frame_bgr.shape[:2]
        meta = self.AVAILABLE_ENGINES[target_engine]

        persons: list[dict[str, Any]] = []
        hands: list[dict[str, Any]] = []

        with self._lock:
            if enable_pose:
                if meta["backend"] == "ultralytics":
                    persons = self._infer_yolo(frame_bgr, target_engine, conf_threshold, use_imgsz)
                else:
                    persons = self._infer_mediapipe(frame_bgr, conf_threshold)

            if enable_hands and self.hand_ready:
                hands = self._infer_hands(frame_bgr)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        fps_est = round(1000.0 / max(elapsed_ms, 0.1), 1)

        vram_used_mb = 0.0
        if self.cuda_available:
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                vram_used_mb = round((total_b - free_b) / (1024 * 1024), 1)
            except Exception:
                pass

        return {
            "engine": target_engine,
            "compute_mode": self.compute_mode,
            "topology": meta["topology"],
            "frame_size": {"w": w, "h": h},
            "inference_ms": elapsed_ms,
            "theoretical_fps": fps_est,
            "enable_pose": enable_pose,
            "enable_hands": enable_hands,
            "person_count": len(persons),
            "hand_count": len(hands),
            "vram_used_mb": vram_used_mb,
            "persons": persons,
            "hands": hands,
        }

    def _infer_yolo(
        self,
        frame_bgr: np.ndarray,
        engine_name: str,
        conf_threshold: float,
        imgsz: int,
    ) -> list[dict[str, Any]]:
        model = self._ensure_yolo(engine_name)
        results = model.predict(
            frame_bgr,
            device=self.device,
            conf=max(0.15, min(0.95, conf_threshold)),
            imgsz=imgsz,
            verbose=False,
        )

        persons: list[dict[str, Any]] = []
        if not results:
            return persons

        res = results[0]
        if res.keypoints is None or res.boxes is None:
            return persons

        kpts_xy = res.keypoints.xyn.cpu().numpy()
        kpts_conf = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones((len(kpts_xy), 17), dtype=np.float32)
        )
        boxes_xyxyn = res.boxes.xyxyn.cpu().numpy()
        boxes_conf = res.boxes.conf.cpu().numpy()

        for idx in range(len(kpts_xy)):
            kp_list = []
            for j in range(min(len(kpts_xy[idx]), len(COCO_KEYPOINT_NAMES))):
                x_val = float(kpts_xy[idx][j][0])
                y_val = float(kpts_xy[idx][j][1])
                c_val = float(kpts_conf[idx][j])
                if x_val == 0.0 and y_val == 0.0:
                    c_val = 0.0
                kp_list.append({
                    "id": j,
                    "name": COCO_KEYPOINT_NAMES[j],
                    "x": round(x_val, 4),
                    "y": round(y_val, 4),
                    "conf": round(c_val, 3),
                })

            bbox = boxes_xyxyn[idx]
            kin = compute_kinematics(kp_list, topology="coco17", min_conf=conf_threshold)
            persons.append({
                "unit_id": f"UNIT-{idx + 1:02d}",
                "det_conf": round(float(boxes_conf[idx]), 3),
                "bbox": {
                    "x1": round(float(bbox[0]), 4),
                    "y1": round(float(bbox[1]), 4),
                    "x2": round(float(bbox[2]), 4),
                    "y2": round(float(bbox[3]), 4),
                },
                "keypoints": kp_list,
                "kinematics": kin,
            })

        return persons

    def _infer_mediapipe(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
    ) -> list[dict[str, Any]]:
        pose = self._ensure_mediapipe()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return []

        kp_list = []
        xs, ys = [], []
        for j, lm in enumerate(res.pose_landmarks.landmark):
            vis = float(getattr(lm, "visibility", 1.0))
            x_val = max(0.0, min(1.0, float(lm.x)))
            y_val = max(0.0, min(1.0, float(lm.y)))
            if vis >= conf_threshold:
                xs.append(x_val)
                ys.append(y_val)
            kp_list.append({
                "id": j,
                "name": MEDIAPIPE_KEYPOINT_NAMES[j] if j < len(MEDIAPIPE_KEYPOINT_NAMES) else f"KP_{j}",
                "x": round(x_val, 4),
                "y": round(y_val, 4),
                "conf": round(vis, 3),
            })

        if not xs:
            return []

        pad = 0.04
        bbox = {
            "x1": round(max(0.0, min(xs) - pad), 4),
            "y1": round(max(0.0, min(ys) - pad), 4),
            "x2": round(min(1.0, max(xs) + pad), 4),
            "y2": round(min(1.0, max(ys) + pad), 4),
        }
        kin = compute_kinematics(kp_list, topology="mediapipe33", min_conf=conf_threshold)
        return [{
            "unit_id": "UNIT-01",
            "det_conf": round(kin["integrity_pct"] / 100.0, 3),
            "bbox": bbox,
            "keypoints": kp_list,
            "kinematics": kin,
        }]

    def _infer_hands(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        landmarker = self._ensure_hand_landmarker()
        if landmarker is None:
            return []

        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return []

        hands_out: list[dict[str, Any]] = []
        for idx, lm_list in enumerate(res.hand_landmarks):
            handedness = "UNKNOWN"
            score = 0.9
            if res.handedness and idx < len(res.handedness) and res.handedness[idx]:
                cat = res.handedness[idx][0]
                handedness = (cat.category_name or "UNKNOWN").upper()
                score = round(float(cat.score), 3)

            kp_list = []
            xs, ys = [], []
            for j, lm in enumerate(lm_list):
                x_val = max(0.0, min(1.0, float(lm.x)))
                y_val = max(0.0, min(1.0, float(lm.y)))
                xs.append(x_val)
                ys.append(y_val)
                kp_list.append({
                    "id": j,
                    "name": HAND_KEYPOINT_NAMES[j] if j < len(HAND_KEYPOINT_NAMES) else f"H_{j}",
                    "x": round(x_val, 4),
                    "y": round(y_val, 4),
                    "z": round(float(lm.z), 4),
                    "conf": score,
                })

            pad = 0.025
            bbox = {
                "x1": round(max(0.0, min(xs) - pad), 4),
                "y1": round(max(0.0, min(ys) - pad), 4),
                "x2": round(min(1.0, max(xs) + pad), 4),
                "y2": round(min(1.0, max(ys) + pad), 4),
            }
            kin = compute_hand_kinematics(kp_list)
            side_code = "L" if handedness.startswith("L") else "R" if handedness.startswith("R") else "H"
            hands_out.append({
                "hand_id": f"HAND-{side_code}{idx + 1}",
                "handedness": handedness,
                "score": score,
                "bbox": bbox,
                "keypoints": kp_list,
                "kinematics": kin,
            })

        return hands_out
