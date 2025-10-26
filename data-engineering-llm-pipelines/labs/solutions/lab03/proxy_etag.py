# proxy_etag.py — add ETag/If-None-Match on JSON pages
import time, os, httpx, hashlib
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

DATASETTE = os.getenv("UPSTREAM", "http://127.0.0.1:8001")
TOKEN = os.getenv("API_TOKEN", "super-secret-token")
# RATE_LIMIT = int(os.getenv("RATE_LIMIT", 500))  # requests per minute per token
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 60))

app = FastAPI()
_counters = {}

def etag_for(path: str, query: str) -> str:
    h = hashlib.sha256(f"{path}?{query}".encode()).hexdigest()
    return f"W/\"{h[:16]}\""  # weak ETag

@app.middleware("http")
async def auth_rate(request: Request, call_next):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return JSONResponse({"error": "missing bearer token"}, status_code=401)
    if auth.split(" ",1)[1] != TOKEN:
        return JSONResponse({"error": "invalid token"}, status_code=403)
    # very small fixed window limiting
    now = int(time.time()) // 60
    key = (auth, now)
    _counters[key] = _counters.get(key, 0) + 1
    if _counters[key] > RATE_LIMIT:
        return JSONResponse({"error": "rate limit"}, status_code=429, headers={"Retry-After":"10"})
    return await call_next(request)

@app.api_route("/{path:path}", methods=["GET"])
async def passthrough(path: str, request: Request):
    # Compute ETag based on full request target
    q = str(request.query_params)
    et = etag_for(path, q)
    inm = request.headers.get("if-none-match")
    if inm == et:
        return Response(status_code=304, headers={"ETag": et})

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{DATASETTE}/{path}", params=request.query_params)
        hdrs = dict(r.headers)
        hdrs["ETag"] = et
        return Response(content=r.content, status_code=r.status_code, headers=hdrs, media_type=hdrs.get("content-type"))
