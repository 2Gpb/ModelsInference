"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

import os

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_fastapi_instrumentator import Instrumentator

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
TRITON_METRICS_URL = os.getenv("TRITON_METRICS_URL", "http://triton:8002")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
NODE_EXPORTER_URL = os.getenv("NODE_EXPORTER_URL", "http://node-exporter:9100")
CV_MODEL = os.getenv("CV_MODEL", "efficientnet_b0")
JWT_SECRET = os.environ["JWT_SECRET"]
STUDENT_NAME = os.getenv("STUDENT_NAME", "-")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "-")
LLM_MODEL = os.getenv("LLM_MODEL", "-")
GPU_NAME = os.getenv("GPU_NAME", "-")
VERSION = os.getenv("VERSION", "-")

app = FastAPI(title="Real-time Inference Gateway")
client = httpx.AsyncClient(timeout=5.0)
bearer = HTTPBearer(auto_error=False)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


async def proxy_metrics(url: str, name: str):
    try:
        result = await client.get(f"{url}/metrics")
        ready = result.status_code == 200
    except httpx.HTTPError:
        ready = False

    if not ready:
        return JSONResponse(status_code=502, content={"detail": f"{name} is not available"})

    return PlainTextResponse(content=result.text, media_type="text/plain")


def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="missing token")
    try:
        return jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health():
    try:
        result = await client.get(f"{TRITON_URL}/v2/health/ready")
        ready = result.status_code == 200
    except httpx.HTTPError:
        ready = False

    if not ready:
        return JSONResponse(status_code=503, content={"detail": "triton is not ready"})

    return {"backend": "triton", "ready": True, "model": CV_MODEL}


@app.get("/vllm/health")
async def vllm_health():
    try:
        result = await client.get(f"{VLLM_URL}/v1/models")
        ready = result.status_code == 200
        ids = [m["id"] for m in result.json()["data"]]
    except (httpx.HTTPError, ValueError, KeyError):
        ready = False

    if not ready:
        return JSONResponse(status_code=503, content={"detail": "vllm is not ready"})

    return {"backend": "vllm", "ready": True, "models": ids}


@app.get("/metrics/triton")
async def triton_metrics():
    return await proxy_metrics(TRITON_METRICS_URL, "triton")


@app.get("/metrics/vllm")
async def vllm_metrics():
    return await proxy_metrics(VLLM_URL, "vllm")


@app.get("/metrics/node")
async def node_metrics():
    return await proxy_metrics(NODE_EXPORTER_URL, "node-exporter")


@app.get("/info", dependencies=[Depends(verify_token)])
async def info():
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": GPU_NAME,
        "version": VERSION
    }