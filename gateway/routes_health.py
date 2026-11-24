import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from config import (
    CV_MODEL,
    GPU_NAME,
    LLM_MODEL,
    STUDENT_EMAIL,
    STUDENT_NAME,
    TRITON_URL,
    VERSION,
    VLLM_URL,
)
from helpers import client, verify_token

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/triton/health")
async def triton_health():
    try:
        result = await client.get(f"{TRITON_URL}/v2/health/ready")
        ready = result.status_code == 200
    except httpx.HTTPError:
        ready = False

    if not ready:
        return JSONResponse(status_code=503, content={"detail": "triton is not ready"})

    return {"backend": "triton", "ready": True, "model": CV_MODEL}


@router.get("/vllm/health")
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


@router.get("/info", dependencies=[Depends(verify_token)])
async def info():
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": GPU_NAME,
        "version": VERSION,
    }
