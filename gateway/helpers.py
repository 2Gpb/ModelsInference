from __future__ import annotations

import io

import httpx
import jwt
import numpy as np
import tritonclient.http.aio as triton_http
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from config import JWT_SECRET, MEAN, STD, TRITON_URL

Image.MAX_IMAGE_PIXELS = None

client = httpx.AsyncClient(timeout=5.0)
bearer = HTTPBearer(auto_error=False)
triton_client = None


def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="missing token")
    try:
        return jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token")


async def proxy_metrics(url: str, name: str):
    try:
        result = await client.get(f"{url}/metrics")
        ready = result.status_code == 200
    except httpx.HTTPError:
        ready = False

    if not ready:
        return JSONResponse(status_code=502, content={"detail": f"{name} is not available"})

    return PlainTextResponse(content=result.text, media_type="text/plain")


def get_triton():
    global triton_client
    if triton_client is None:
        triton_client = triton_http.InferenceServerClient(url=TRITON_URL.removeprefix("http://"))
    return triton_client


def preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if w <= h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = int(round((new_w - 224) / 2.0))
    top = int(round((new_h - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)
    return x[np.newaxis]


def vllm_error(status: int, text: str) -> HTTPException:
    if 400 <= status < 500 and status != 404:
        return HTTPException(status_code=400, detail=f"vllm rejected request: {text[:500]}")
    return HTTPException(status_code=502, detail=f"vllm error {status}")
