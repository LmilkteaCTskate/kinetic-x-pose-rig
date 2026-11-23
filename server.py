"""
KINETIC-X // Industrial Human Skeleton & Hand Telemetry Server
Server-ready FastAPI + WebSocket backend with:
- Auto hardware detection (CUDA / MPS / Multi-Core CPU)
- External network optimization (0.0.0.0 bind, GZip, latest-frame-only WS queue, RTT ping)
- Optional HTTPS/TLS support for remote browser camera access
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pose_engine import PoseEngine

STATIC_DIR = Path(__file__).parent / "static"
engine: PoseEngine | None = None


def get_engine() -> PoseEngine:
    global engine
    if engine is None:
        engine = PoseEngine(default_engine="yolo11n-pose")
    return engine


def _discover_lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            ips.append(ip)
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


@asynccontextmanager
async def lifespan(app: FastAPI):
    eng = get_engine()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, eng.preload_default)
    yield


app = FastAPI(
    title="KINETIC-X // Industrial Pose & Hand Telemetry System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class EngineSwitchRequest(BaseModel):
    engine: str


@app.get("/")
async def serve_index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health_check():
    eng = get_engine()
    return JSONResponse({
        "status": "ok",
        "device": eng.device,
        "compute_mode": eng.compute_mode,
        "hardware_tier": eng.profile["tier"],
        "hand_ready": eng.hand_ready,
    })


@app.get("/api/system")
async def get_system_info():
    return JSONResponse(get_engine().get_system_info())


@app.post("/api/engine")
async def switch_engine(req: EngineSwitchRequest):
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(None, get_engine().set_engine, req.engine)
        return JSONResponse(info)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    engine_name: str | None = Form(None),
    imgsz: int | None = Form(None),
    enable_pose: bool = Form(True),
    enable_hands: bool = Form(True),
):
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图像文件")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        get_engine().infer,
        frame,
        float(conf),
        engine_name,
        int(imgsz) if imgsz else None,
        bool(enable_pose),
        bool(enable_hands),
    )
    return JSONResponse(result)


@app.websocket("/ws/pose")
async def websocket_pose(ws: WebSocket):
    """
    External-network optimized WebSocket endpoint.
    Uses a latest-frame-only slot so bursty WAN/Wi-Fi frames never build up lag.
    """
    await ws.accept()
    eng = get_engine()
    loop = asyncio.get_running_loop()

    cfg_state: dict[str, Any] = {
        "conf": 0.35,
        "engine": None,
        "imgsz": eng.profile["recommended_imgsz"],
        "enable_pose": True,
        "enable_hands": True,
    }

    latest_frame: list[bytes | None] = [None]
    frame_event = asyncio.Event()
    closed = False

    async def inference_worker():
        nonlocal closed
        while not closed:
            await frame_event.wait()
            frame_event.clear()
            raw_bytes = latest_frame[0]
            latest_frame[0] = None
            if raw_bytes is None:
                continue

            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            try:
                result = await loop.run_in_executor(
                    None,
                    eng.infer,
                    frame,
                    cfg_state["conf"],
                    cfg_state["engine"],
                    cfg_state["imgsz"],
                    cfg_state["enable_pose"],
                    cfg_state["enable_hands"],
                )
                result["type"] = "pose"
                await ws.send_json(result)
            except Exception:
                closed = True
                break

    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    msg: dict[str, Any] = json.loads(message["text"])
                    mtype = msg.get("type")
                    if mtype == "ping":
                        await ws.send_json({"type": "pong", "ts": msg.get("ts", 0)})
                    elif mtype == "config":
                        if "conf" in msg:
                            cfg_state["conf"] = max(0.15, min(0.95, float(msg["conf"])))
                        if "engine" in msg:
                            cfg_state["engine"] = str(msg["engine"])
                        if "imgsz" in msg:
                            cfg_state["imgsz"] = int(msg["imgsz"])
                        if "enable_pose" in msg:
                            cfg_state["enable_pose"] = bool(msg["enable_pose"])
                        if "enable_hands" in msg:
                            cfg_state["enable_hands"] = bool(msg["enable_hands"])
                        await ws.send_json({"type": "config_ack", **cfg_state})
                except Exception:
                    pass
                continue

            if "bytes" in message and message["bytes"] is not None:
                # Overwrite any stale unprocessed frame -> zero backlog latency
                latest_frame[0] = message["bytes"]
                frame_event.set()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        closed = True
        frame_event.set()
        worker_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KINETIC-X Industrial Pose & Hand Server")
    parser.add_argument("--host", default=os.environ.get("KINETIC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("KINETIC_PORT", "8000")))
    parser.add_argument(
        "--device",
        default=os.environ.get("KINETIC_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu"],
        help="Compute device mode (default: auto)",
    )
    parser.add_argument("--ssl-cert", default=os.environ.get("KINETIC_SSL_CERT", ""))
    parser.add_argument("--ssl-key", default=os.environ.get("KINETIC_SSL_KEY", ""))
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    os.environ["KINETIC_DEVICE"] = args.device

    use_ssl = bool(args.ssl_cert and args.ssl_key)
    scheme = "https" if use_ssl else "http"

    print("=" * 64)
    print("  KINETIC-X // INDUSTRIAL POSE & HAND TELEMETRY SERVER v2.0")
    print(f"  Device Mode : {args.device.upper()} (Auto-detects CUDA / CPU)")
    print(f"  Local URL   : {scheme}://127.0.0.1:{args.port}")
    for lan_ip in _discover_lan_ips():
        print(f"  Network URL : {scheme}://{lan_ip}:{args.port}")
    if not use_ssl:
        print("  [NOTE] External devices need HTTPS for live browser camera.")
        print("         Image/Video/Demo modes work over HTTP directly.")
    print("=" * 64)

    uvicorn_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "ws_ping_interval": 20,
        "ws_ping_timeout": 20,
    }
    if use_ssl:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_cert
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_key

    uvicorn.run("server:app", **uvicorn_kwargs)
