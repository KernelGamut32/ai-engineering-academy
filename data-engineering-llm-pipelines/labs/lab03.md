# Lab 03 — Resilient API Harvester + SQL Extractor with Provenance

**Focus Areas:** API Harvester (rate‑limited, paginated, with retries & snapshots) and SQL Extractor (parameterized, chunked, Parquet)

See **Appendices** below for additional information on what *with Provenance* means.

---

## Outcomes

By the end of this lab, you will be able to:

1. Run a **local REST API** backed by SQLite (Datasette) and a lightweight **auth + rate‑limit proxy** that enforces Bearer tokens and emits `429` with `Retry-After`.
2. Implement a production‑style **API harvester** that supports **query params**, **cursor pagination**, **exponential backoff with jitter**, and **idempotent incremental fetch**.
3. Persist **raw JSON snapshots** per page with a **provenance manifest** (source URL, params, status, checksum, timestamp).
4. Normalize harvested JSON into pandas DataFrames and write to **Parquet** (optionally partitioned) for downstream validation/profiling.
5. Use **parameterized SQL** against SQLite via `pd.read_sql_query` with **chunked reads** and append to Parquet in a streaming fashion.

---

## Prerequisites & Setup

- Python 3.13 with `requests`, `pandas`, `numpy`, `pyarrow`, `datasette`, `fastapi`, `uvicorn`, `httpx`, `aiofiles` installed (proxy uses FastAPI).
- JupyterLab or VS Code with Jupyter extension.
- Completion of Lab 02 (reuse of **SQLite** setup).

### 1) Create a clean workspace and env

```bash
mkdir -p lab03 && cd lab03
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install requests pandas numpy pyarrow datasette fastapi uvicorn aiofiles httpx
```

### 2) Reuse / obtain the SQLite DB (Northwind)

```bash
# If you already have northwind.db from Lab 02, copy it here. Otherwise:
curl -L -o northwind.db \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

### 3) Start Datasette on port 8001 (same as Lab 02)

```bash
datasette northwind.db -h 127.0.0.1 -p 8001
```

Leave this running. Open a new terminal for the **proxy**.

### 4) Start an **Auth + Rate‑Limit Proxy** (FastAPI) on port 9000

Create `proxy.py` with the following contents:

```python
# proxy.py — Bearer auth + fixed-window rate limit + pass-through to Datasette
import time, asyncio, os
from typing import Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import httpx

DATASETTE = os.getenv("UPSTREAM", "http://127.0.0.1:8001")
REQUIRED_TOKEN = os.getenv("API_TOKEN", "super-secret-token")
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
```

Run the proxy:

```bash
export API_TOKEN=super-secret-token
uvicorn proxy:app --host 127.0.0.1 --port 9000 --reload
```

> The proxy exposes the same endpoints as Datasette but now requires `Authorization: Bearer super-secret-token` and enforces a per‑minute request cap (default 60). It returns `429` with `Retry-After` when exceeded.

### 5) Start a notebook

- Start a new notebook named: `week02_lab03.ipynb`.

---

## Part A — API Harvester with Retries, Pagination, and Snapshots

### A1. Project scaffolding

```python
import os, json, hashlib, time, datetime as dt
from pathlib import Path
BASE = "http://127.0.0.1:9000"   # go through proxy
DB   = "northwind"
TABLE = "Orders"
PAGE_SIZE = 200
TOKEN = "super-secret-token"

ARTIFACTS = Path("artifacts")
RAW_DIR = ARTIFACTS / "raw"
MANIFEST = ARTIFACTS / "manifest.jsonl"
PARQUET_DIR = ARTIFACTS / "parquet"
for d in [RAW_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

### A2. Resilient GET helper with **exponential backoff + jitter** and `Retry-After`

```python
import requests, random

class HarvestError(Exception):
    pass

def resilient_get(url, params=None, headers=None, max_retries=6):
    backoff = 0.5
    headers = headers or {}
    for attempt in range(1, max_retries+1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                delay = float(ra) if ra else backoff
            elif resp.status_code in (500, 502, 503, 504):
                delay = backoff
            else:
                raise HarvestError(f"Unexpected {resp.status_code}: {resp.text[:200]}")
            # exponential backoff with jitter
            jitter = random.uniform(0, 0.25 * delay)
            time.sleep(delay + jitter)
            backoff = min(backoff * 2, 8.0)
        except (requests.Timeout, requests.ConnectionError) as e:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    raise HarvestError(f"Failed after {max_retries} retries: {url}")
```

### A3. Cursor **pagination** + **raw snapshot** persistence

```python
import pandas as pd

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def save_snapshot(payload_bytes: bytes, meta: dict) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{meta['table']}_{ts}_{meta['page']:05d}.json"
    fpath = RAW_DIR / fname
    fpath.write_bytes(payload_bytes)
    # append to manifest as JSONL
    record = {
        **meta,
        "timestamp": ts,
        "sha256": sha256_bytes(payload_bytes),
        "bytes": len(payload_bytes)
    }
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return fpath

def harvest_table(base, db, table, page_size=100, start_cursor=None, extra_params=None):
    url = f"{base}/{db}/{table}.json"
    params = {"_size": page_size}
    if extra_params:
        params.update(extra_params)
    next_tok = start_cursor
    page = 0
    all_rows = []
    while True:
        if next_tok:
            params["_next"] = next_tok
        resp = resilient_get(url, params=params, headers=HEADERS)
        payload_bytes = resp.content
        payload = resp.json()
        rows = payload.get("rows", [])
        page += 1
        save_snapshot(payload_bytes, {
            "source": url,
            "params": params,
            "table": table,
            "status": resp.status_code,
            "page": page
        })
        if not rows:
            break
        all_rows.extend(rows)
        next_tok = payload.get("next")
        if not next_tok:
            break
    return pd.DataFrame(all_rows)

orders = harvest_table(BASE, DB, "Orders", page_size=PAGE_SIZE)
orders.shape, orders.head()
```

**Checkpoint:** Verify that `artifacts/raw` contains multiple JSON snapshots and `manifest.jsonl` records each page.

### A4. **Incremental harvesting** with a high‑watermark

Pick a date column to watermark—`OrderDate` exists in Northwind. We’ll store `last_watermark.txt`.

```python
WATERMARK_FILE = ARTIFACTS / "last_watermark.txt"

def read_watermark(default="1997-01-01"):
    if WATERMARK_FILE.exists():
        return WATERMARK_FILE.read_text().strip()
    return default

def write_watermark(value: str):
    WATERMARK_FILE.write_text(value)

last = read_watermark()
print("Starting watermark:", last)

# Harvest only rows since last watermark
orders_inc = harvest_table(
    BASE, DB, "Orders", page_size=PAGE_SIZE,
    extra_params={"OrderDate__gte": last}  # Datasette accepts column filters
)

# Advance watermark to the max OrderDate we saw (if any)
if not orders_inc.empty and "OrderDate" in orders_inc.columns:
    new_wm = max(str(d) for d in orders_inc["OrderDate"])  # strings are fine in YYYY-MM-DD
    write_watermark(new_wm)
    print("Advanced watermark to:", new_wm)
else:
    print("No new rows; watermark unchanged:", last)
```

<!-- **Note:** If your Datasette build doesn’t support the `__gte` operator, filter client‑side after a full pull, then write the new watermark. (Instructor may enable canned queries for server‑side filtering.) -->

### A5. Normalize and persist to **Parquet** (partitioned)

```python
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

# Simple normalization: select a subset + parse dates
use = orders[["OrderID","CustomerID","OrderDate","ShipCountry","Freight"]].copy()
use["OrderDate"] = pd.to_datetime(use["OrderDate"], errors="coerce").dt.date

# Partition by ShipCountry for faster downstream filters
out_root = PARQUET_DIR / "orders"
out_root.mkdir(parents=True, exist_ok=True)

for country, g in use.groupby("ShipCountry"):
    table = pa.Table.from_pandas(g, preserve_index=False)
    pq.write_table(table, out_root / f"shipcountry={country}.parquet")

# Quick QA
files = list(out_root.glob("*.parquet"))
len(files), files[:3]
```

### A6. (Optional) Harvest a second table and join

```python
details = harvest_table(BASE, DB, "Order Details", page_size=PAGE_SIZE)
# Normalize column names (Datasette may include spaces)
details.columns = [c.replace(" ", "_") for c in details.columns]

# Join orders to details on OrderID
merged = details.merge(use, on="OrderID", how="inner")
merged.head(), len(merged)
```

Persist `merged` to `PARQUET_DIR/line_items.parquet`.

---

## Part B — SQL Extractor with Parameterization & Chunked Reads (≈45 min)

### B1. Parameterized queries with user inputs

```python
import sqlite3
conn = sqlite3.connect("northwind.db")

country = "USA"
start_date = "1997-01-01"
q = """
SELECT o.OrderID, o.CustomerID, o.OrderDate, o.ShipCountry,
       d.ProductID, d.UnitPrice, d.Quantity, d.Discount
FROM Orders o
JOIN [Order Details] d ON o.OrderID = d.OrderID
WHERE o.ShipCountry = ? AND o.OrderDate >= ?
ORDER BY o.OrderDate
"""
params = (country, start_date)

rows = pd.read_sql_query(q, conn, params=params)
rows.head(), len(rows)
```

### B2. Chunked reads → streaming Parquet append

```python
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

out_path = PARQUET_DIR / "orders_joined.parquet"
writer = None
for chunk in pd.read_sql_query(q, conn, params=params, chunksize=25_000):
    # Optional transforms
    chunk["OrderDate"] = pd.to_datetime(chunk["OrderDate"], errors="coerce")
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
if writer:
    writer.close()

# Verify
ver = pd.read_parquet(out_path)
len(ver), ver.sample(3, random_state=42)
```

### B3. (Optional) Partitioned write by date

```python
# Partition by year of OrderDate
rows2 = pd.read_sql_query(q, conn, params=params)
rows2["OrderDate"] = pd.to_datetime(rows2["OrderDate"], errors="coerce")
rows2["year"] = rows2["OrderDate"].dt.year

for yr, g in rows2.groupby("year"):
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), PARQUET_DIR / f"orders_year={yr}.parquet")

list(PARQUET_DIR.glob("orders_year=*.parquet"))[:5]
```

---

## Part C — Wrap‑Up

- In a final markdown cell, answer:
  1. How does your harvester treat `429` vs `500`? What signal does `Retry-After` convey? Why add **jitter**?
  2. Explain your **high‑watermark** logic. What happens if the API returns out‑of‑order data?
  3. Why are **parameterized queries** critical, even for local labs? Provide a small example of a risky string‑formatted SQL.
  4. When would you choose **chunked** reads? Name one trade‑off.

---

- **Final thoughts:**
  - Adds **auth** + **rate limiting** via a proxy (students must pass Bearer token).
  - Introduces **provenance manifest** with checksums and byte sizes per page.
  - Implements **incremental harvest** with high‑watermark and (optional) server‑side filtering.
  - Writes **partitioned Parquet** and demonstrates a **join** of harvested vs SQL‑selected data.
  - **Common pitfalls:**
    - Forgetting the `Authorization` header → `401/403`.
    - Exhausting rate limits quickly by setting `_size` too small → make effective use of the backoff.
    - Confusing client‑side vs server‑side filtering; Datasette’s filter operators may vary.
    - Using f‑string SQL instead of `params=`.

---

## Stretch Goals

1. **Client‑side caching & ETags:** Extend the proxy to add a fake `ETag` header per `next` cursor; modify harvester to send `If-None-Match` and handle `304 Not Modified`.
2. **Bounded concurrency:** Use `concurrent.futures.ThreadPoolExecutor` to fetch multiple pages concurrently while respecting a **token‑bucket** limiter in the client.
3. **Structured logging:** Emit JSON logs per page with `logger.info(json.dumps({...}))` and include timing, attempts, and sleep durations.
4. **Data contracts:** Define a small `pydantic`/`pandera` schema for Orders; validate types and required columns before Parquet write. On failure, move the page’s snapshot to `artifacts/raw/rejects/` and log.
5. **Idempotent merges:** Write harvested increments to a Delta‑style directory layout (or use a simple upsert by `OrderID` in pandas) so reruns don’t duplicate records.
6. **Retry budget:** Implement a maximum **sleep budget** (e.g., 60s) across all retries; abort if exceeded.
7. **CLI packaging:** Convert harvester into a CLI (`python -m harvester --table Orders --since 1997-01-01`) with `argparse` and a config file (`harvest.yml`).
8. **Unit tests:** Add tests for `resilient_get` and watermark advancement using `pytest` and `responses`/`requests-mock`.

## Stretch Goal Solutions — Guided Steps & Code

These are written to drop directly into your `lab03/` workspace. Where you see `▶` bullets, run them as shell commands (Mac/Linux/Windows PowerShell compatible where possible).

> **Prereqs:**
>
> ```bash
> pip install pydantic pandera[yaml] jsonschema responses requests-mock loguru pyyaml typer rich
> ```

## 1) Client‑side caching & ETags (If‑None‑Match / 304)

### 1.1 Add ETag support to the proxy

Create `proxy_etag.py` (or integrate into `proxy.py`). The proxy emits a stable ETag per `_next` cursor (or per URL when no cursor). On match, it returns **304 Not Modified**.

```python
# proxy_etag.py — add ETag/If-None-Match on JSON pages
import time, os, httpx, hashlib
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

DATASETTE = os.getenv("UPSTREAM", "http://127.0.0.1:8001")
TOKEN = os.getenv("API_TOKEN", "super-secret-token")
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
```

▶ `uvicorn proxy_etag:app --host 127.0.0.1 --port 9000 --reload`

### 1.2 Update the client

Add conditional requests using `If-None-Match` and short‑circuit on **304**.

```python
# harvester_etag_client.py — snippet to replace resilient_get
import requests, time, random
ETAG_CACHE = {}  # key: (url, frozenset(params.items())) -> etag

def resilient_get_conditional(url, params=None, headers=None, max_retries=5):
    params = params or {}
    headers = headers or {}
    key = (url, frozenset(params.items()))
    if key in ETAG_CACHE:
        headers = {**headers, "If-None-Match": ETAG_CACHE[key]}

    backoff = 0.5
    for _ in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 304:
            return None  # signal 'not modified'
        if r.status_code == 200:
            if et := r.headers.get("ETag"):
                ETAG_CACHE[key] = et
            return r
        if r.status_code == 429:
            delay = float(r.headers.get("Retry-After", backoff))
        elif r.status_code in (500,502,503,504):
            delay = backoff
        else:
            r.raise_for_status()
        time.sleep(delay + random.uniform(0, 0.25*delay))
        backoff = min(backoff*2, 8)
    raise RuntimeError("exhausted retries")
```

## 2) Bounded concurrency with a client‑side token bucket

Use threads to prefetch pages, while a **token bucket** enforces request rate.

```python
# concurrent_harvest.py
import time, threading, queue, requests, random
from concurrent.futures import ThreadPoolExecutor, as_completed

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()
    def take(self, n=1):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts)*self.rate)
                self.ts = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
            time.sleep(0.01)

bucket = TokenBucket(rate_per_sec=5, capacity=5)  # <=5 req/s burst 5

def fetch_page(url, params, headers):
    bucket.take()
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code == 200:
        return r.json()
    elif r.status_code == 429:
        time.sleep(float(r.headers.get('Retry-After', '1')))
        return fetch_page(url, params, headers)
    r.raise_for_status()

def parallel_pages(url, cursors, headers):
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_page, url, {"_size":200, "_next":c} if c else {"_size":200}, headers) for c in cursors]
        for f in as_completed(futs):
            results.append(f.result())
    return results
```

---

## 3) Structured logging (JSON) with timing & sleep

Use **loguru** (or stdlib) to emit one JSON line per page.

```python
from loguru import logger
import json, time
logger.add("artifacts/harvest.log", serialize=True, rotation="5 MB")

start = time.perf_counter()
# inside pagination loop
attempts = 1
sleep_total = 0.0
# ... after each request
elapsed = time.perf_counter() - start
logger.info({
    "event":"page_fetched","table":table,"page":page,
    "rows":len(rows),"attempts":attempts,"slept":sleep_total,
    "elapsed":elapsed
})
```

---

## 4) Data contracts with pydantic & pandera

Validate page payloads and final frames.

### 4.1 Pydantic model for JSON rows

```python
from pydantic import BaseModel, Field
from typing import Optional
class OrderRow(BaseModel):
    OrderID: int
    CustomerID: Optional[str]
    OrderDate: Optional[str]
    ShipCountry: Optional[str]
    Freight: Optional[float] = Field(ge=0)  # non‑negative

# validate rows list
validated = [OrderRow(**r) for r in rows]
```

### 4.2 Pandera schema for pandas DataFrame

```python
import pandera as pa
from pandera import Column, Check

OrdersSchema = pa.DataFrameSchema({
    "OrderID": Column(int, Check.ge(1), nullable=False),
    "CustomerID": Column(object, nullable=True),
    "OrderDate": Column(object, nullable=True),
    "ShipCountry": Column(object, nullable=True),
    "Freight": Column(float, Check.ge(0), nullable=True),
})

validated_df = OrdersSchema.validate(use, lazy=True)
```

---

## 5) Idempotent merges (dedupe/upsert by key)

Maintain a single Parquet with unique `OrderID`, replacing duplicates on rerun.

```python
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

TARGET = PARQUET_DIR / "orders_dedup.parquet"

# load existing
if TARGET.exists():
    base = pd.read_parquet(TARGET)
else:
    base = pd.DataFrame(columns=["OrderID","CustomerID","OrderDate","ShipCountry","Freight"]) 

# new increment
inc = use.copy()

# upsert by OrderID
combined = pd.concat([base[~base.OrderID.isin(inc.OrderID)], inc], ignore_index=True)
combined.sort_values(["OrderID"], inplace=True)

pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), TARGET)
print(len(base), ">>>", len(inc), "=>", len(combined))
```

---

## 6) Retry budget (max total sleep)

Abort if cumulative backoff exceeds a threshold.

```python
MAX_SLEEP = 60.0

def resilient_get_with_budget(url, params=None, headers=None, max_retries=6):
    import requests, time, random
    slept = 0.0
    backoff = 0.5
    for _ in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            delay = float(r.headers.get("Retry-After", backoff))
        elif r.status_code in (500,502,503,504):
            delay = backoff
        else:
            r.raise_for_status()
        jitter = random.uniform(0, 0.25*delay)
        time.sleep(delay + jitter)
        slept += delay + jitter
        if slept > MAX_SLEEP:
            raise TimeoutError(f"Retry budget exceeded: {slept:.1f}s")
        backoff = min(backoff*2, 8)
    raise TimeoutError("exhausted retries")
```

---

## 7) CLI packaging (argparse/typer) + YAML config

Create a CLI to run harvests without notebooks.

### 7.1 `harvester_cli.py`

```python
import json, yaml, time
from pathlib import Path
import typer
import requests

app = typer.Typer()

@app.command()
def harvest(config: str = typer.Argument(..., help="Path to harvest.yml")):
    cfg = yaml.safe_load(Path(config).read_text())
    base = cfg["base"]; db = cfg["db"]; table = cfg["table"]
    token = cfg.get("token", "super-secret-token")
    size = int(cfg.get("page_size", 200))
    headers = {"Authorization": f"Bearer {token}"}

    url = f"{base}/{db}/{table}.json"
    params = {"_size": size}
    out_dir = Path(cfg.get("out_dir", "artifacts/raw")); out_dir.mkdir(parents=True, exist_ok=True)

    page = 0
    next_tok = None
    while True:
        if next_tok: params["_next"] = next_tok
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
        page += 1
        (out_dir / f"{table}_{page:05d}.json").write_text(json.dumps(payload))
        next_tok = payload.get("next")
        if not next_tok: break

if __name__ == "__main__":
    app()
```

### 7.2 `harvest.yml`

```yaml
base: "http://127.0.0.1:9000"
db: "northwind"
table: "Orders"
page_size: 200
out_dir: "artifacts/raw"
token: "super-secret-token"
```

▶ `python harvester_cli.py harvest harvest.yml`

---

## 8) Unit tests for retries & watermark

Use **pytest** with **responses** to simulate network conditions.

### 8.1 `tests/test_retries.py`

```python
import responses, requests
from harvester_etag_client import resilient_get_conditional

@responses.activate
def test_retry_then_success():
    url = "http://x/api"
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, json={"ok":1}, status=200, headers={"ETag":"W/\"abc\""})
    r = resilient_get_conditional(url)
    assert r.status_code == 200

@responses.activate
def test_etag_304_shortcircuit():
    url = "http://x/api"
    # first call caches ETag
    responses.add(responses.GET, url, json={"ok":1}, status=200, headers={"ETag":"W/\"abc\""})
    resilient_get_conditional(url)
    # second call returns 304
    responses.add(responses.GET, url, status=304)
    assert resilient_get_conditional(url) is None
```

### 8.2 `tests/test_watermark.py`

```python
from pathlib import Path
from lab1c_harvester import read_watermark, write_watermark  # adjust import path

def test_watermark_roundtrip(tmp_path: Path):
    wm = tmp_path/"last_watermark.txt"
    # monkeypatch the constant or pass path as param in your functions
    write_watermark.__globals__["WATERMARK_FILE"] = wm
    read_watermark.__globals__["WATERMARK_FILE"] = wm

    assert read_watermark("1997-01-01") == "1997-01-01"
    write_watermark("1998-02-03")
    assert read_watermark() == "1998-02-03"
```

▶ `pytest -q`

---

### Finishing notes

- Keep **artifacts** deterministic: set a global `SEED` and include env/version stamps in your manifest.
- Prefer **idempotent** runs: upsert by keys; snapshot but avoid duplicate parquet rows.
- For production, push logs to a central store and emit metrics (success rate, rows/sec, retry count, sleep budget).

---

## **Appendix 01 - "with Provenance" Meaning**

### What "with Provenance" means

"…with Provenance" means the lab doesn’t just pull data—it also records the lineage and context of every page/chunk you harvest and extract, so you can reproduce, audit, debug, and trust downstream results.

### What "provenance" covers

- **Where from:** source URL, database/table, query params, pagination cursor.
- **When/how:** timestamps (UTC), runtime/version info, status codes, retry counts, rate-limit headers (e.g., Retry-After).
- **What exactly:** raw JSON snapshot per page plus a checksum (e.g., SHA-256) and byte size.
- **Which slice:** high-watermark value used (e.g., OrderDate >= 1997-01-01), page number, _next token.
- **Who/which config:** auth mode/token alias, environment variables, config file/hash (for CLI runs).
- **Post-processing:** normalization steps taken, Parquet paths written, partition keys.

### Why it matters (especially for LLM pipelines)

- **Reproducibility:** Rehydrate any dataset state used to train/evaluate an LLM component.
- **Auditability & compliance:** Show exactly what inputs fed a result—key for regulated data or FOIA-style reviews.
- **Debuggability:** Track bad pages, retry storms, schema drifts, or partial harvests.
- **Data quality:** Tie validation failures back to the specific snapshot(s) that caused them.
- **Idempotency:** Checksums + manifests help prevent duplicates and enable safe re-runs.

### How the lab implements it

- **Raw snapshots:** Each harvested page is saved unchanged (artifacts/raw/...json).
- **Manifest:** A newline-delimited manifest.jsonl records metadata per page (URL, params, status, timestamp, checksum, bytes).
- **Watermarks:** last_watermark.txt captures incremental boundaries to make runs repeatable and explainable.
- **Partitioned outputs:** Parquet written with explicit partitioning (e.g., by ShipCountry) is referenced in the manifest for traceability.

### Minimal provenance schema (example)

```json
{
  "timestamp": "2025-10-25T18:02:11Z",
  "source": "http://127.0.0.1:9000/northwind/Orders.json",
  "params": {"_size": 200, "_next": "eyJjdXJzb3I..."},
  "status": 200,
  "attempts": 1,
  "retry_after": null,
  "sha256": "b6f2…",
  "bytes": 124567,
  "table": "Orders",
  "page": 7,
  "watermark_in": "1997-01-01",
  "output_parquet": "artifacts/parquet/orders/shipcountry=USA.parquet",
  "env": {"python": "3.11.9", "pandas": "2.2.2", "numpy": "1.26.4"}
}
```

**In short:** "with Provenance" = harvests you can defend—every result is traceable back to the exact inputs and code/config that produced it.

## **Appendix 02 - "high watermark" Meaning**

A high watermark is the saved "last-seen" value from a monotonic field (e.g., a timestamp or numeric ID) that you persist after each run so the next run only fetches newer records.

### In this lab’s context

- You harvest pages from the local API and track the maximum `OrderDate` (or `OrderID`) you observed.
- You store that max in `last_watermark.txt`.
- On the next run, you query **incrementally:** "give me rows where `OrderDate >= <last_watermark>`" (or `>`, depending on your overlap policy).
- After harvesting, you advance the file to the new max, so subsequent runs don’t re-download old data.

### Why use it

- **Efficiency:** avoid re-pulling everything every time.
- **Idempotency:** with a small overlap window you can safely handle pagination edges.
- **Reproducibility/auditability:** the exact boundary you used is recorded.

### Choosing a watermark column

Pick a field that is:

- **Monotonic (grows over time):** e.g., `created_at`, `updated_at`, `OrderDate`, or an ever-increasing `OrderID`.
- **Comparable & stable** (consistent type/timezone).
- **High cardinality** (to avoid lots of ties).

### Edge cases to handle

- **Ties/ordering at the boundary:** Prefer `>=` with a dedupe on write, or use `>` and keep an overlap page to be safe.
- **Out-of-order events:** If data can arrive late, consider using `updated_at` as the watermark (captures corrections) or run periodic backfills.
- **Timezone/format drift:** Normalize to UTC ISO-8601 before comparing.
- **Deletes/updates:** A `created_at` watermark won’t see updates; an `updated_at` watermark will.

### Watermark vs. checkpoints (quick contrast)

- **Watermark:** a single boundary value you compare against to fetch only new/changed rows.
- **Checkpoint/snapshot:** a fuller record of progress (e.g., page cursor, file offsets, manifest entries). You’re already keeping snapshots + a manifest for provenance; the watermark is the tiny, fast index that powers incremental harvesting.
