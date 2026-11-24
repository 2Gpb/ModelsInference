import os

import httpx
import numpy as np

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
TRITON_METRICS_URL = os.getenv("TRITON_METRICS_URL", "http://triton:8002")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
NODE_EXPORTER_URL = os.getenv("NODE_EXPORTER_URL", "http://node-exporter:9100")

CV_MODEL = os.getenv("CV_MODEL", "efficientnet_b0")
LLM_MODEL = os.getenv("LLM_MODEL", "-")

JWT_SECRET = os.environ["JWT_SECRET"]

STUDENT_NAME = os.getenv("STUDENT_NAME", "-")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "-")
GPU_NAME = os.getenv("GPU_NAME", "-")
VERSION = os.getenv("VERSION", "-")

LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
with open(LABELS_PATH, encoding="utf-8") as f:
    LABELS = [line.strip() for line in f]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CHAT_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
