# proxy.py — Bearer auth + fixed-window rate limit + pass-through to Datasette
import time, asyncio, os
from typing import Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import httpx

DATASETTE = os.getenv("UPSTREAM", "http://127.0.0.1:8001")
REQUIRED_TOKEN = os.getenv("API_TOKEN", "super-secret-token")
# RATE_LIMIT = int(os.getenv("RATE_LIMIT", 500))  # requests per minute per token
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 60))  # requests per minute per token

app = FastAPI()

# simple in-memory counters (sufficient for lab)
_counters = {}
_window_starts = {}

async def check_rate_limit(token: str) -> Optional[int]:
    now = int(time.time())
    window = now // 60
    key = (token, window)
    if _window_starts.get(key) is None:
        _window_starts[key] = window
        _counters[key] = 0
    _counters[key] += 1
    remaining = RATE_LIMIT - _counters[key]
    if remaining < 0:
        reset = (window + 1) * 60 - now
        return reset
    return None

@app.middleware("http")
async def enforce_auth_and_rate_limit(request: Request, call_next):
    # enforce bearer token
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return JSONResponse({"error": "missing bearer token"}, status_code=401)
    token = auth.split(" ", 1)[1]
    if token != REQUIRED_TOKEN:
        return JSONResponse({"error": "invalid token"}, status_code=403)

    # rate limit
    reset = await check_rate_limit(token)
    if reset is not None:
        headers = {"Retry-After": str(reset), "X-RateLimit-Reset": str(reset)}
        return JSONResponse({"error": "rate limit exceeded"}, status_code=429, headers=headers)

    return await call_next(request)

@app.api_route("/{path:path}", methods=["GET"])
async def proxy(path: str, request: Request):
    # Very small pass-through for GET to Datasette
    params = dict(request.query_params)
    upstream_url = f"{DATASETTE}/{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(upstream_url, params=params)
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers), media_type=r.headers.get("content-type"))
    