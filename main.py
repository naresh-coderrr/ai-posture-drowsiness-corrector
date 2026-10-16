"""
AI Posture & Drowsiness Corrector — FastAPI Server (Local Python MediaPipe + Vercel Serverless Ready)
© 2026 Naresh (naresh-coderrr). All Rights Reserved.
"""

import os
import json
import base64
from typing import Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase_auth import SupabaseAuthManager

# Optional local Python MediaPipe + OpenCV engine (gracefully skipped on Vercel Serverless <500MB limit)
try:
    import cv2
    import numpy as np
    from vision_engine import PostureDrowsinessEngine
    vision_engine: Optional[Any] = PostureDrowsinessEngine()
except Exception:
    cv2 = None
    np = None
    vision_engine = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="SpineGuard AI — Real-Time Posture & Drowsiness Corrector",
    description="Real-time MediaPipe Face & Pose Aspect Ratio tracker with Supabase Unique-Password Auth. © 2026 Naresh (naresh-coderrr).",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth_manager = SupabaseAuthManager()


# --- Pydantic Request Schemas ---
class FrameAnalyzeRequest(BaseModel):
    session_id: str
    image_base64: str
    force_calibrate: bool = False


class ThresholdUpdateRequest(BaseModel):
    session_id: str
    ear_threshold: float = 0.22
    mar_threshold: float = 0.58
    posture_sensitivity: float = 1.0


class PasswordCheckRequest(BaseModel):
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


class SupabaseConnectRequest(BaseModel):
    supabase_url: str
    supabase_key: str


def decode_base64_frame(data_uri: str):
    """Decode base64 JPEG/PNG data URI into OpenCV BGR numpy array."""
    if cv2 is None or np is None:
        return None
    try:
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        img_bytes = base64.b64decode(data_uri)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# --- API Endpoints ---
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "python_vision_available": vision_engine is not None,
        "engine": "MediaPipe Tasks (Face + Pose + ObjectDetector)",
        "copyright": "© 2026 Naresh (naresh-coderrr). All Rights Reserved.",
        "supabase": auth_manager.get_status(),
    }


@app.get("/api/supabase/status")
async def supabase_status():
    return auth_manager.get_status()


@app.post("/api/supabase/connect")
async def connect_supabase(req: SupabaseConnectRequest):
    if not req.supabase_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Supabase URL must start with https://")
    status = auth_manager.configure_supabase(req.supabase_url, req.supabase_key)
    return status


@app.post("/api/auth/check-password")
async def check_password_unique(req: PasswordCheckRequest):
    is_unique, message = auth_manager.is_password_unique(req.password)
    return {"unique": is_unique, "message": message}


@app.post("/api/auth/register")
async def register_user(req: RegisterRequest):
    res = auth_manager.register_user(req.username, req.email, req.password)
    if not res["ok"]:
        return JSONResponse(status_code=400, content=res)
    return res


@app.post("/api/auth/login")
async def login_user(req: LoginRequest):
    res = auth_manager.login_user(req.identifier, req.password)
    if not res["ok"]:
        return JSONResponse(status_code=401, content=res)
    return res


@app.post("/api/analyze")
async def analyze_frame_http(req: FrameAnalyzeRequest):
    if vision_engine is None:
        return JSONResponse(status_code=503, content={"use_client_mediapipe": True})
    frame = decode_base64_frame(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image frame")
    result = vision_engine.analyze_frame(
        session_id=req.session_id,
        frame_bgr=frame,
        force_calibrate=req.force_calibrate,
    )
    return result


@app.post("/api/calibrate")
async def calibrate_posture(req: FrameAnalyzeRequest):
    if vision_engine is None:
        return {"calibrated": True, "use_client_mediapipe": True}
    frame = decode_base64_frame(req.image_base64)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image frame")
    return vision_engine.calibrate_session(req.session_id, frame)


@app.post("/api/thresholds")
async def update_thresholds(req: ThresholdUpdateRequest):
    if vision_engine is not None:
        vision_engine.update_thresholds(
            session_id=req.session_id,
            ear_threshold=req.ear_threshold,
            mar_threshold=req.mar_threshold,
            posture_sensitivity=req.posture_sensitivity,
        )
    return {"ok": True}


# --- Real-Time WebSocket Endpoint (Local Python Server) ---
@app.websocket("/ws/analyze/{session_id}")
async def websocket_analyze(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if vision_engine is None:
        await websocket.send_json({"use_client_mediapipe": True})
        await websocket.close()
        return
    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)
            msg_type = payload.get("type", "frame")

            if msg_type == "thresholds":
                vision_engine.update_thresholds(
                    session_id=session_id,
                    ear_threshold=payload.get("ear_threshold", 0.22),
                    mar_threshold=payload.get("mar_threshold", 0.58),
                    posture_sensitivity=payload.get("posture_sensitivity", 1.0),
                )
                await websocket.send_json({"type": "thresholds_updated"})
                continue

            img_b64 = payload.get("image")
            force_cal = bool(payload.get("calibrate", False))
            if not img_b64:
                continue

            frame = decode_base64_frame(img_b64)
            if frame is None:
                continue

            telemetry = vision_engine.analyze_frame(
                session_id=session_id,
                frame_bgr=frame,
                force_calibrate=force_cal,
            )
            await websocket.send_json(telemetry)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
