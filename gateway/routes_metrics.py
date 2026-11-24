from fastapi import APIRouter

from config import NODE_EXPORTER_URL, TRITON_METRICS_URL, VLLM_URL
from helpers import proxy_metrics

router = APIRouter()


@router.get("/metrics/triton")
async def triton_metrics():
    return await proxy_metrics(TRITON_METRICS_URL, "triton")


@router.get("/metrics/vllm")
async def vllm_metrics():
    return await proxy_metrics(VLLM_URL, "vllm")


@router.get("/metrics/node")
async def node_metrics():
    return await proxy_metrics(NODE_EXPORTER_URL, "node-exporter")
