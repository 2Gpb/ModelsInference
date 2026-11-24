import asyncio
import time

import httpx
import numpy as np
import tritonclient.http.aio as triton_http
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config import CHAT_TIMEOUT, CV_MODEL, LABELS, LLM_MODEL, VLLM_URL
from helpers import client, get_triton, preprocess, verify_token, vllm_error

router = APIRouter()


@router.post("/vision", dependencies=[Depends(verify_token)])
async def vision(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        x = await asyncio.to_thread(preprocess, data)
    except Exception:
        raise HTTPException(status_code=400, detail="not an image")

    inp = triton_http.InferInput("INPUT__0", list(x.shape), "FP32")
    inp.set_data_from_numpy(x, binary_data=True)
    out = triton_http.InferRequestedOutput("OUTPUT__0", binary_data=True)

    start = time.perf_counter()
    try:
        result = await get_triton().infer(CV_MODEL, inputs=[inp], outputs=[out])
    except Exception:
        raise HTTPException(status_code=502, detail="triton inference failed")
    latency_ms = (time.perf_counter() - start) * 1000

    logits = result.as_numpy("OUTPUT__0")[0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top5 = np.argsort(probs)[::-1][:5]

    return {
        "model": CV_MODEL,
        "predictions": [
            {"class_id": int(i), "label": LABELS[i], "score": float(probs[i])} for i in top5
        ],
        "latency_ms": latency_ms,
    }


@router.post("/chat", dependencies=[Depends(verify_token)])
async def chat(request: Request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid json")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        raise HTTPException(status_code=400, detail="messages is required")

    stream = body.get("stream", True)
    payload = {
        "model": LLM_MODEL,
        "messages": body["messages"],
        "max_tokens": body.get("max_tokens", 128),
        "temperature": body.get("temperature", 0.0),
        "stream": stream,
    }
    url = f"{VLLM_URL}/v1/chat/completions"

    if not stream:
        try:
            r = await client.post(url, json=payload, timeout=CHAT_TIMEOUT)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="vllm unavailable")
        if r.status_code != 200:
            raise vllm_error(r.status_code, r.text)
        return JSONResponse(content=r.json())

    try:
        req = client.build_request("POST", url, json=payload, timeout=CHAT_TIMEOUT)
        resp = await client.send(req, stream=True)
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="vllm unavailable")
    if resp.status_code != 200:
        text = (await resp.aread()).decode(errors="replace")
        await resp.aclose()
        raise vllm_error(resp.status_code, text)

    async def relay():
        done = False
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                yield line + "\n\n"
                if line.strip() == "data: [DONE]":
                    done = True
        except httpx.HTTPError:
            yield 'data: {"error": {"message": "stream interrupted"}}\n\n'
        finally:
            await resp.aclose()
        if not done:
            yield "data: [DONE]\n\n"

    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
