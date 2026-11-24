"""Real-time inference gateway: точка входа.

Ручки разнесены по файлам:
    routes_health.py    — /health, /triton/health, /vllm/health, /info
    routes_metrics.py   — /metrics/triton, /metrics/vllm, /metrics/node
    routes_inference.py — /vision, /chat
Общие функции — helpers.py, настройки — config.py.
"""
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

import routes_health
import routes_inference
import routes_metrics

app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(routes_health.router)
app.include_router(routes_metrics.router)
app.include_router(routes_inference.router)
