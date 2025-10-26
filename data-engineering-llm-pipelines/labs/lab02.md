# Lab 02 — Local API (`requests`) + SQL Extraction to pandas

**Focus Areas:** HTTP APIs with `requests`, SQL → pandas (SQLite)

---

## Outcomes

By the end of this lab, you will be able to:

1. Call a **local REST API** with query parameters, parse JSON, and implement **robust error handling** for status codes.
2. Implement **pagination** and an **exponential backoff** retry policy that respects `429 Too Many Requests` and `5xx` errors.
3. Extract data from **SQLite** into pandas using **parameterized queries** and `pd.read_sql_query`, including **chunked reads**.
4. Persist results to **Parquet** for downstream LLM preprocessing.

---

## Prerequisites & Setup

- Python 3.13 with `requests`, `pandas`, `numpy`, `matplotlib`, `pyarrow` installed.
- JupyterLab or VS Code with Jupyter extension.
- **SQLite** available (via Python’s built‑in `sqlite3`; optional CLI install if you want the shell - see **Appendix** below for more info on installation).
- **Local API** served by **Datasette** exposing data from a SQLite database.

### 1) Create a project folder and environment

```bash
mkdir lab02 && cd lab02
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install requests pandas numpy matplotlib pyarrow datasette
```

### 2) Get a sample SQLite DB (Northwind)

```bash
curl -L -o northwind.db \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

### 3) Run a **local REST API** with Datasette (read‑only JSON over SQLite)

```bash
# Serves a browsable site *and* JSON endpoints
# Terminal will print a local http://127.0.0.1:8001 URL
# macOS/Linux:
datasette northwind.db -h 127.0.0.1 -p 8001
# Windows PowerShell (same command works)
```

Keep this server running. Open a second terminal for the notebook.

### 4) Start a notebook

- Start a new notebook named: `week02_lab02.ipynb`.

---

## Part A — HTTP API with `requests`

> You will call the Datasette JSON API. Endpoint pattern:  
> `http://127.0.0.1:8001/<db>/<table>.json?_size=PAGE_SIZE&_next=...`  
> We’ll use `Orders` and `OrderDetails` tables.

### A1. Warm‑up: GET with params & `.json()`

```python
import requests
BASE = "http://127.0.0.1:8001"
DB = "northwind"
TABLE = "Orders"
PAGE_SIZE = 50

params = {"_size": PAGE_SIZE}
r = requests.get(f"{BASE}/{DB}/{TABLE}.json", params=params, timeout=10)
print(r.status_code)
data = r.json()  # raises if not JSON
list(data.keys()), data.get("rows", [])[:2]
```

**Checkpoint:** Identify where rows live (Datasette returns `rows`).

### A2. Robust request helper with **status handling**

```python
from typing import Dict, Any
import time

class APIError(Exception):
    pass

def get_json(url: str, params: Dict[str, Any] | None = None, *, max_retries: int = 5) -> dict:
    backoff = 0.5
    for attempt in range(1, max_retries+1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            status = resp.status_code
            if status == 200:
                return resp.json()
            elif status in (429, 500, 502, 503, 504):
                # exponential backoff
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise APIError(f"Unexpected status {status}: {resp.text[:200]}")
        except (requests.Timeout, requests.ConnectionError) as e:
            time.sleep(backoff)
            backoff *= 2
    raise APIError(f"Failed after {max_retries} attempts: {url}")
```

**Simulating 429/5xx:** Stop the server briefly or change the port in `BASE` to provoke errors and observe retries.

### A3. **Pagination** (`_next` cursor)

```python
import pandas as pd

def fetch_all(base: str, db: str, table: str, page_size: int = 100) -> pd.DataFrame:
    url = f"{base}/{db}/{table}.json"
    params = {"_size": page_size}
    out = []
    next_tok = None
    while True:
        if next_tok:
            params["_next"] = next_tok
        payload = get_json(url, params)
        rows = payload.get("rows", [])
        if not rows:
            break
        out.extend(rows)
        next_tok = payload.get("next")  # Datasette provides a cursor token
        if not next_tok:
            break
    return pd.DataFrame(out)

orders = fetch_all(BASE, DB, "Orders", page_size=200)
orders.head(), len(orders)
```

### A4. Query filters via params

Datasette supports simple filter syntax. Example: find orders with `ShipCountry = 'USA'`.

```python
usa = get_json(f"{BASE}/{DB}/Orders.json", params={"ShipCountry": "USA", "_size": 50})
len(usa["rows"]) , usa["rows"][:2]
```

**Checkpoint:** Note how query parameters map to column filters.

---

## Part B — SQL → pandas with SQLite

### B1. Parameterized queries

```python
import sqlite3, pandas as pd
# Update file path as needed
conn = sqlite3.connect("lab02/northwind.db")

country = "USA"  # from user input in real apps
q = """
SELECT OrderID, CustomerID, OrderDate, ShipCountry
FROM Orders
WHERE ShipCountry = ? AND OrderDate >= ?
ORDER BY OrderDate DESC
"""
params = (country, "1997-01-01")

safe_df = pd.read_sql_query(q, conn, params=params)
safe_df.head()
```

> **Why `?` placeholders?** Prevents SQL injection—SQLite driver will safely bind values.

### B2. Chunked reads for large tables

```python
big_q = "SELECT * FROM [Order Details]"  # space requires brackets in SQLite
chunks = pd.read_sql_query(big_q, conn, chunksize=10_000)

import pyarrow.parquet as pq
import pyarrow as pa

# Stream to Parquet in chunks
writer = None
for i, chunk in enumerate(chunks, start=1):
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        # Update file path as needed
        writer = pq.ParquetWriter("lab02/order_details.parquet", table.schema)
    writer.write_table(table)

if writer is not None:
    writer.close()
```

**Checkpoint:** Verify output file size and row count by re‑reading with pandas.

### B3. Quick validation snapshot

```python
import pandas as pd
# Update file path as needed
p = pd.read_parquet("lab02/order_details.parquet")
print(len(p))
print(p.select_dtypes(include='number').describe().T.head())
```

---

## Part C — Wrap‑Up

- In a final markdown cell, answer:
  1. How does your retry/backoff behave for 429 vs 500?
  2. Why are **parameterized** queries the default choice? Provide a one‑sentence example of a potential injection if not parameterized.
  3. When would you choose chunked reads? What trade‑off do you incur?

---

- **Final thoughts:**
  - **Datasette tips:** JSON is at `.../table.json` (use `_size`, `_next`, and column filters)
  - **Simulating errors:** Stop server to trigger connection errors; add a bogus param to force 4xx; explore retry logs.
  - **Common pitfalls:**
    - Missing `timeout` in `requests` → hanging cells.
    - Forgetting `params` vs string concatenation in URLs.
    - Using f‑strings to inject SQL instead of `params=...`.

---

## Appendix — SQLite setup quick guide

- **Use Python’s built‑in driver:** No extra install needed for the lab (`import sqlite3`).
- **Optional CLI:**
  - **macOS (Homebrew):** `brew install sqlite` → verify with `sqlite3 --version`.
  - **Windows:** install from <https://www.sqlite.org/download.html> (precompiled binaries). Add folder to `PATH`, then `sqlite3 --version`.
  - **Linux:** `sudo apt-get install sqlite3` (Debian/Ubuntu) or distro equivalent.
- **Sanity check:**

  ```bash
  sqlite3 northwind.db ".tables"
  sqlite3 northwind.db
  .schema Orders
  .quit
  ```

---

## Solution Snippets (reference)

**Backoff conditions you might retry:** `429, 500, 502, 503, 504` (idempotent GETs). Use jitter in production to avoid thundering herds.

**Cursor pagination recap (Datasette):** Examine `payload["next"]` and pass it back as `_next` with the same `_size` to retrieve subsequent pages.

**`read_sql_query` params:** Use `?` for SQLite, `%s` for Postgres, and pass values via `params=...` so the DB‑API binds them safely.

**Artifacts to retain:** `orders.parquet`, `order_details.parquet`.
