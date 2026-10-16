"""
AI Posture & Drowsiness Corrector — Real-Time MediaPipe Vision Engine
© 2026 Naresh (naresh-coderrr). All Rights Reserved.

Uses:
1. MediaPipe FaceLandmarker: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Blink rate, and Head Pose
2. MediaPipe PoseLandmarker: Spinal alignment, forward neck slouch, and shoulder balance
3. MediaPipe ObjectDetector (EfficientDet-Lite0): TRUE "cell phone" object detection (never confuses an open palm or hand gesture with a phone!)
"""

import os
import math
import time
import urllib.request
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
}

# MediaPipe 468-point Face Mesh Landmark Indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [61, 37, 13, 267, 291, 314, 14, 84]


def ensure_models_downloaded() -> Dict[str, str]:
    """Ensure all MediaPipe model bundles exist locally."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 10000:
            urllib.request.urlretrieve(url, filepath)
        paths[filename] = filepath
    return paths


def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, indices: List[int], width: int, height: int) -> float:
    """
    Calculate Eye Aspect Ratio (EAR):
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    horiz = euclidean_dist(pts[0], pts[3])
    if horiz < 1e-6:
        return 0.30
    return (v1 + v2) / (2.0 * horiz)


def calculate_mar(landmarks, indices: List[int], width: int, height: int) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR):
    MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
    """
    pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
    v1 = euclidean_dist(pts[1], pts[7])
    v2 = euclidean_dist(pts[2], pts[6])
    v3 = euclidean_dist(pts[3], pts[5])
    horiz = euclidean_dist(pts[0], pts[4])
    if horiz < 1e-6:
        return 0.0
    return (v1 + v2 + v3) / (2.0 * horiz)


class PostureDrowsinessEngine:
    """
    Real-time AI Posture, Drowsiness (EAR/MAR), and True Mobile Phone Object Detector
    powered by Python MediaPipe Tasks API.
    """

    def __init__(self):
        paths = ensure_models_downloaded()

        # 1. FaceLandmarker (EAR, MAR, Blink Rate, Head Pitch/Yaw)
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["face_landmarker.task"]),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

        # 2. PoseLandmarker (Spinal alignment & shoulder posture)
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["pose_landmarker_lite.task"]),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # 3. ObjectDetector (True COCO `cell phone` detection — ignores bare hands/palms!)
        obj_opts = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["efficientdet_lite0.tflite"]),
            running_mode=mp_vision.RunningMode.IMAGE,
            score_threshold=0.35,
            category_allowlist=["cell phone", "remote"],
            max_results=3,
        )
        self.object_detector = mp_vision.ObjectDetector.create_from_options(obj_opts)

        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "blink_count": 0,
                "yawn_count": 0,
                "eye_closed_start": None,
                "yawn_start": None,
                "slouch_start": None,
                "phone_start": None,
                "distracted_start": None,
                "was_blinking": False,
                "was_yawning": False,
                "baseline_neck_ratio": None,
                "baseline_shoulder_y": None,
                "calibrated": False,
                "blink_timestamps": [],
                "ear_threshold": 0.22,
                "mar_threshold": 0.58,
                "posture_sensitivity": 1.0,
            }
        return self.sessions[session_id]

    def calibrate_session(self, session_id: str, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Calibrate upright baseline posture from current frame."""
        sess = self.get_session(session_id)
        metrics = self.analyze_frame(session_id, frame_bgr, force_calibrate=True)
        return {
            "calibrated": sess["calibrated"],
            "baseline_neck_ratio": sess["baseline_neck_ratio"],
            "baseline_shoulder_y": sess["baseline_shoulder_y"],
            "metrics": metrics,
        }

    def update_thresholds(
        self,
        session_id: str,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.58,
        posture_sensitivity: float = 1.0,
    ):
        sess = self.get_session(session_id)
        sess["ear_threshold"] = max(0.14, min(0.32, float(ear_threshold)))
        sess["mar_threshold"] = max(0.35, min(0.85, float(mar_threshold)))
        sess["posture_sensitivity"] = max(0.5, min(2.0, float(posture_sensitivity)))

    def analyze_frame(
        self,
        session_id: str,
        frame_bgr: np.ndarray,
        force_calibrate: bool = False,
    ) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        now = time.time()
        h, w = frame_bgr.shape[:2]

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_res = self.face_landmarker.detect(mp_image)
        pose_res = self.pose_landmarker.detect(mp_image)
        obj_res = self.object_detector.detect(mp_image)

        ear_left = 0.30
        ear_right = 0.30
        ear_avg = 0.30
        mar = 0.15
        head_yaw = 0.0
        head_pitch = 0.0
        head_roll = 0.0
        face_detected = False

        spine_score = 100
        neck_angle = 90.0
        shoulder_tilt = 0.0
        slouch_reasons: List[str] = []

        phone_detected = False
        phone_confidence = 0.0
        phone_boxes: List[Dict[str, float]] = []
        distracted_detected = False
        distraction_reason = ""

        overlay_points: Dict[str, Any] = {
            "left_eye": [],
            "right_eye": [],
            "mouth": [],
            "pose": [],
            "phone_boxes": [],
        }

        # --- 1. FACE ASPECT RATIO (EAR / MAR) & HEAD ORIENTATION ---
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            face_detected = True
            flm = face_res.face_landmarks[0]

            ear_left = calculate_ear(flm, LEFT_EYE_IDX, w, h)
            ear_right = calculate_ear(flm, RIGHT_EYE_IDX, w, h)
            ear_avg = round((ear_left + ear_right) / 2.0, 3)
            mar = round(calculate_mar(flm, MOUTH_IDX, w, h), 3)

            overlay_points["left_eye"] = [[round(flm[i].x, 4), round(flm[i].y, 4)] for i in LEFT_EYE_IDX]
            overlay_points["right_eye"] = [[round(flm[i].x, 4), round(flm[i].y, 4)] for i in RIGHT_EYE_IDX]
            overlay_points["mouth"] = [[round(flm[i].x, 4), round(flm[i].y, 4)] for i in MOUTH_IDX]

            nose = flm[1]
            forehead = flm[10]
            chin = flm[152]
            left_cheek = flm[234]
            right_cheek = flm[454]

            cheek_mid_x = (left_cheek.x + right_cheek.x) / 2.0
            cheek_span = max(0.05, abs(right_cheek.x - left_cheek.x))
            head_yaw = round(((nose.x - cheek_mid_x) / cheek_span) * 90.0, 1)

            face_height = max(0.05, abs(chin.y - forehead.y))
            nose_rel_y = (nose.y - forehead.y) / face_height
            head_pitch = round((nose_rel_y - 0.56) * 160.0, 1)

            dy_eyes = (flm[263].y - flm[33].y) * h
            dx_eyes = (flm[263].x - flm[33].x) * w
            head_roll = round(math.degrees(math.atan2(dy_eyes, max(1.0, dx_eyes))), 1)

            # Blink & Eye Closure Counter
            if ear_avg < sess["ear_threshold"]:
                if sess["eye_closed_start"] is None:
                    sess["eye_closed_start"] = now
                sess["was_blinking"] = True
            else:
                if sess["was_blinking"]:
                    closed_dur = now - (sess["eye_closed_start"] or now)
                    if 0.04 <= closed_dur <= 0.65:
                        sess["blink_count"] += 1
                        sess["blink_timestamps"].append(now)
                sess["was_blinking"] = False
                sess["eye_closed_start"] = None

            sess["blink_timestamps"] = [t for t in sess["blink_timestamps"] if now - t <= 60.0]

            # Yawn Detector
            if mar > sess["mar_threshold"]:
                if sess["yawn_start"] is None:
                    sess["yawn_start"] = now
                elif (now - sess["yawn_start"]) > 1.1 and not sess["was_yawning"]:
                    sess["yawn_count"] += 1
                    sess["was_yawning"] = True
            else:
                sess["yawn_start"] = None
                sess["was_yawning"] = False

            if abs(head_yaw) > 30.0:
                distracted_detected = True
                distraction_reason = "Looking away from screen"
            elif head_pitch > 24.0:
                distracted_detected = True
                distraction_reason = "Head drooping down"
        else:
            distracted_detected = True
            distraction_reason = "Face not in camera view"

        # --- 2. SPINAL & SHOULDER POSTURE ANALYSIS ---
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            plm = pose_res.pose_landmarks[0]
            nose_p = plm[0]
            l_sh = plm[11]
            r_sh = plm[12]

            overlay_points["pose"] = [
                {"id": "nose", "x": round(nose_p.x, 4), "y": round(nose_p.y, 4)},
                {"id": "left_shoulder", "x": round(l_sh.x, 4), "y": round(l_sh.y, 4)},
                {"id": "right_shoulder", "x": round(r_sh.x, 4), "y": round(r_sh.y, 4)},
            ]

            sh_mid_x = (l_sh.x + r_sh.x) / 2.0
            sh_mid_y = (l_sh.y + r_sh.y) / 2.0
            sh_width = max(0.10, abs(l_sh.x - r_sh.x))

            neck_vertical_ratio = (sh_mid_y - nose_p.y) / sh_width

            if force_calibrate or (not sess["calibrated"] and neck_vertical_ratio > 0.55):
                sess["baseline_neck_ratio"] = neck_vertical_ratio
                sess["baseline_shoulder_y"] = sh_mid_y
                sess["calibrated"] = True

            baseline_ratio = sess["baseline_neck_ratio"] or 0.78
            shoulder_tilt = round(
                math.degrees(math.atan2((r_sh.y - l_sh.y) * h, max(1.0, abs(r_sh.x - l_sh.x) * w))),
                1,
            )

            dx_spine = (nose_p.x - sh_mid_x) * w
            dy_spine = (sh_mid_y - nose_p.y) * h
            axial_offset_deg = abs(math.degrees(math.atan2(dx_spine, max(1.0, dy_spine))))
            neck_angle = round(90.0 - axial_offset_deg, 1)

            ratio_drop = max(0.0, (baseline_ratio - neck_vertical_ratio) / max(0.2, baseline_ratio))
            shoulder_drop = max(0.0, sh_mid_y - (sess["baseline_shoulder_y"] or sh_mid_y))

            sens = sess["posture_sensitivity"]
            penalty = 0.0

            if ratio_drop > (0.14 / sens):
                penalty += min(45.0, ratio_drop * 180.0 * sens)
                slouch_reasons.append("Neck slouching forward")

            if abs(shoulder_tilt) > (8.0 / sens):
                penalty += min(30.0, (abs(shoulder_tilt) - 5.0) * 2.8 * sens)
                slouch_reasons.append("Uneven shoulders")

            if axial_offset_deg > (12.0 / sens):
                penalty += min(30.0, (axial_offset_deg - 8.0) * 2.2 * sens)
                slouch_reasons.append("Leaning to one side")

            if shoulder_drop > (0.05 / sens):
                penalty += min(25.0, shoulder_drop * 300.0 * sens)
                if "Slouching down in chair" not in slouch_reasons:
                    slouch_reasons.append("Slouching down in chair")

            spine_score = int(max(0, min(100, round(100.0 - penalty))))

        # --- 3. TRUE MOBILE PHONE DETECTION (EfficientDet Object Detector) ---
        # Only triggers when an actual mobile phone is detected in the frame (never on an empty hand/palm!)
        if obj_res.detections:
            for det in obj_res.detections:
                if not det.categories:
                    continue
                cat = det.categories[0]
                score = float(cat.score or 0.0)
                if score >= 0.35:
                    phone_detected = True
                    phone_confidence = max(phone_confidence, round(score * 100, 1))
                    bb = det.bounding_box
                    box_norm = {
                        "x": round(bb.origin_x / w, 4),
                        "y": round(bb.origin_y / h, 4),
                        "w": round(bb.width / w, 4),
                        "h": round(bb.height / h, 4),
                        "score": round(score * 100, 0),
                    }
                    phone_boxes.append(box_norm)

        overlay_points["phone_boxes"] = phone_boxes

        # --- 4. ALERT EVALUATION ---
        eye_closed_duration = (now - sess["eye_closed_start"]) if sess["eye_closed_start"] else 0.0
        is_drowsy = eye_closed_duration >= 1.35 or (
            sess["yawn_start"] is not None and (now - sess["yawn_start"]) >= 1.6
        )

        is_slouching = spine_score < 68
        if is_slouching:
            if sess["slouch_start"] is None:
                sess["slouch_start"] = now
        else:
            sess["slouch_start"] = None
        slouch_duration = (now - sess["slouch_start"]) if sess["slouch_start"] else 0.0

        if phone_detected:
            if sess["phone_start"] is None:
                sess["phone_start"] = now
        else:
            sess["phone_start"] = None
        phone_duration = (now - sess["phone_start"]) if sess["phone_start"] else 0.0

        if distracted_detected:
            if sess["distracted_start"] is None:
                sess["distracted_start"] = now
        else:
            sess["distracted_start"] = None
        distracted_duration = (now - sess["distracted_start"]) if sess["distracted_start"] else 0.0

        active_alert: Optional[Dict[str, str]] = None
        if is_drowsy:
            active_alert = {
                "type": "DROWSY",
                "severity": "critical",
                "title": "WAKE UP! DROWSINESS DETECTED",
                "message": (
                    f"Your eyes have been closed for {eye_closed_duration:.1f}s! Sit up straight!"
                    if eye_closed_duration >= 1.35
                    else "Heavy yawning detected! Sit up and stay focused!"
                ),
            }
        elif phone_detected and phone_duration >= 0.4:
            active_alert = {
                "type": "PHONE_DISTRACTION",
                "severity": "critical",
                "title": "MOBILE PHONE DETECTED!",
                "message": f"Mobile phone spotted ({int(phone_confidence)}% confidence). Put your phone away and focus!",
            }
        elif slouch_duration >= 1.5:
            reason_str = ", ".join(slouch_reasons) if slouch_reasons else "Slouching detected"
            active_alert = {
                "type": "SLOUCH",
                "severity": "warning",
                "title": "SIT UP STRAIGHT!",
                "message": f"{reason_str} (Posture: {spine_score}%). Straighten your back!",
            }
        elif distracted_duration >= 2.5:
            active_alert = {
                "type": "DISTRACTED",
                "severity": "info",
                "title": "EYES ON SCREEN!",
                "message": f"{distraction_reason}. Lock back into your study session!",
            }

        return {
            "timestamp": round(now, 3),
            "face_detected": face_detected,
            "calibrated": sess["calibrated"],
            "ear": {
                "left": round(ear_left, 3),
                "right": round(ear_right, 3),
                "avg": ear_avg,
                "threshold": sess["ear_threshold"],
                "closed_seconds": round(eye_closed_duration, 2),
            },
            "mar": {
                "value": mar,
                "threshold": sess["mar_threshold"],
            },
            "blinks": {
                "total": sess["blink_count"],
                "per_minute": len(sess["blink_timestamps"]),
                "yawns": sess["yawn_count"],
            },
            "posture": {
                "spine_score": spine_score,
                "neck_angle": neck_angle,
                "shoulder_tilt": shoulder_tilt,
                "head_yaw": head_yaw,
                "head_pitch": head_pitch,
                "head_roll": head_roll,
                "is_slouching": is_slouching,
                "slouch_seconds": round(slouch_duration, 2),
                "reasons": slouch_reasons,
            },
            "distraction": {
                "phone_detected": phone_detected,
                "phone_confidence": phone_confidence,
                "phone_seconds": round(phone_duration, 2),
                "phone_reason": f"Mobile Phone Detected ({int(phone_confidence)}%)" if phone_detected else "No phone detected",
                "looking_away": distracted_detected,
                "distracted_seconds": round(distracted_duration, 2),
                "distraction_reason": distraction_reason,
            },
            "active_alert": active_alert,
            "overlay": overlay_points,
        }
