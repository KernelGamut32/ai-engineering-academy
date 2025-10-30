# Lab 13 — From Tables/Text to JSONL for LLM Workloads

**Focus Area:** JSON Lines (JSONL) for large **LLM** corpora — streaming, memory efficiency, one-record-per-line, provenance, and validation

> In this lab, you’ll work with an **LLM-native dataset** (help articles, policies, release notes, FAQs), not transactional tables. You’ll (1) create a **1,000-row CSV** corpus, (2) expose it via a **local API** (no external network), (3) **ingest + clean** (missing values, filtering, deduping), and (4) write **RAG-ready** and **SFT/eval-ready** JSONL with proper metadata. The flow mirrors real pre-training/RAG prep.

---

## Outcomes

By the end of this lab, you will be able to:

1. Explain why **JSONL** is preferred for large LLM corpora (streaming, append-friendly, line-recoverable, map-reduce-friendly).
2. Generate an **LLM-focused CSV** corpus (1,000 entries) and **serve it via API** with pagination + simple query.
3. Implement a cleaning pipeline: **governance filtering** (e.g., `confidentiality`), **language filtering**, **missing handling**, and **deduplication**.
4. Produce **RAG-chunk JSONL** (`doc_id`→`chunk_id`) and **SFT/eval JSONL** (`{"input","output","metadata"}`) with provenance.
5. Validate JSONL line-by-line and write reviewer **samples**.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `orjson`, `regex`, `fastapi`, `uvicorn`, `pydantic`, `tqdm` (optional)
- JupyterLab or VS Code with Jupyter extension.

- **No external network required.**

**Start a notebook:** `week02_lab13.ipynb`

Create folders:

```python
from pathlib import Path
for p in ['artifacts/jsonl','artifacts/samples','tools','data']:
    Path(p).mkdir(parents=True, exist_ok=True)
```

---

## Part A — Build an LLM-Native Source CSV

We’ll synthesize a realistic **1,000-row** CSV of docs/policies/FAQs.

### A1. Generate `data/corpus_llm.csv` (1,000 rows)

```python
import csv, random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
N = 1000
TYPES = ["help_article","policy","release_note","faq"]
SECTIONS = ["Overview","Setup","Troubleshooting","FAQ"]
TAGS = ["billing","security","compliance","sso","api","governance","export","retention","privacy","rate_limits"]
LANGS = ["en","en","en","de","fr"]  # mostly English
CONF = ["public","internal"]  # governance labels
now = datetime(2025, 2, 10)

boiler = (
    "This article explains how to configure single sign-on with step-by-step instructions. "
    "Use the admin console to enable SAML and verify claim mappings. "
    "Common pitfalls include clock skew and incorrect audience URIs. "
)

rows = []
for i in range(1, N+1):
    kind = random.choice(TYPES)
    doc_id = f"DOC-{i:04d}"
    title = {
        "help_article": f"How to configure SSO (v{random.randint(1,5)}).",
        "policy": f"Data Retention Policy — Region {random.choice(['US','EU','APAC'])}",
        "release_note": f"Release 2025{random.randint(1,12):02d} — Key fixes",
        "faq": f"FAQ: {random.choice(['Exports','Rate Limits','Privacy','Billing'])}"
    }[kind]
    section = random.choice(SECTIONS)
    # Simulate occasional missing/short text and duplicates later
    body = (boiler * random.randint(1,3)) + f"Additional details about {random.choice(TAGS)} and {random.choice(TAGS)}.\n"
    tags = ",".join(random.sample(TAGS, k=random.randint(2,4)))
    created = (now - timedelta(days=random.randint(0, 240))).strftime('%Y-%m-%d')
    updated = (now - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
    language = random.choice(LANGS)
    confidentiality = random.choice(CONF)
    source_url = f"https://example.local/{kind}/{doc_id.lower()}"
    rows.append({
        "doc_id": doc_id,
        "type": kind,
        "title": title,
        "section": section,
        "body_text": body.strip(),
        "tags": tags,
        "source_url": source_url,
        "created_at": created,
        "updated_at": updated,
        "language": language,
        "confidentiality": confidentiality,
    })

# Add deliberate duplicates and a few missing values
for j in range(10):
    rows.append({**rows[j], "doc_id": f"DOC-DUP-{j:02d}"})
for k in range(5):
    rows[k]["body_text"] = ""  # missing text

out_csv = Path('data/corpus_llm.csv')
with out_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print('Wrote', out_csv, 'rows=', len(rows))
```

### A2. Quick peek

```python
import pandas as pd
pd.read_csv('data/corpus_llm.csv').head(5)
```

> **Why this schema?** Mirrors real LLM corpora: metadata for **RAG filtering** (tags/lang/confidentiality), **textual body**, and timestamps.

---

## Part B — Expose the CSV via a Local API

Students will ingest via an API boundary.

### B1. Create `tools/corpus_api.py` (FastAPI)

```python
api_code = '''# tools/corpus_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

app = FastAPI(title="LLM Corpus API", version="1.0.0")
DF = None

class Doc(BaseModel):
    doc_id: str
    type: str
    title: str
    section: str
    body_text: str
    tags: str
    source_url: str
    created_at: str
    updated_at: str
    language: str
    confidentiality: str

@app.on_event("startup")
async def load_data():
    global DF
    DF = pd.read_csv('../data/corpus_llm.csv')
    # Convert NaN to empty strings for all string columns
    string_cols = ['body_text', 'title', 'section', 'tags', 'source_url']
    for col in string_cols:
        DF[col] = DF[col].fillna('')

@app.get('/health')
async def health():
    return {"ok": True}

@app.get('/v1/corpus', response_model=List[Doc])
async def list_docs(page: int = 1, page_size: int = 50, q: str | None = None,
                    language: str | None = None, conf: str | None = None):
    if page < 1 or page_size < 1 or page_size > 200:
        raise HTTPException(400, 'bad paging params')
    df = DF
    if q:
        mask = df['body_text'].astype(str).str.contains(q, case=False, na=False) \\
             | df['title'].astype(str).str.contains(q, case=False, na=False)
        df = df[mask]
    if language:
        df = df[df['language'] == language]
    if conf:
        df = df[df['confidentiality'] == conf]
    start = (page - 1) * page_size
    end = start + page_size
    recs = df.iloc[start:end].to_dict(orient='records')
    return recs

@app.get('/v1/corpus/{doc_id}', response_model=Doc)
async def get_doc(doc_id: str):
    row = DF.loc[DF['doc_id'] == doc_id]
    if row.empty:
        raise HTTPException(404, 'not found')
    return row.iloc[0].to_dict()
'''

with open('tools/corpus_api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)
print("API file created: tools/corpus_api.py")
print("\nTo run the API, execute in a terminal from the tools folder (optionally in a virtual environment):")
print("  uvicorn corpus_api:app --reload --port 8000")
print("\nNote: The API now handles NaN values by converting them to empty strings")
```

### B2. Run the API

Open a new terminal and run:

```bash
uvicorn tools.corpus_api:app --reload --port 8000
```

Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/v1/corpus?page=1&page_size=3`.

---

## Part C — Ingest, Clean, Filter, De-duplicate

We’ll ingest from **CSV** and from the **API**, then enforce governance, language, missing handling, and dedupe.

### C1. Load CSV and define cleaners

```python
import pandas as pd, re, numpy as np
from datetime import datetime

raw = pd.read_csv('data/corpus_llm.csv')

# Basic cleaners
def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s

def clean_row(row):
    row['title'] = normalize_ws(row.get('title',''))
    row['section'] = normalize_ws(row.get('section','')) or 'Overview'
    row['body_text'] = normalize_ws(row.get('body_text',''))
    return row

clean = raw.apply(clean_row, axis=1)
```

### C2. Governance & language filters + missing handling

```python
# Keep only public + English for a typical RAG public index
flt = (clean['confidentiality'] == 'public') & (clean['language'] == 'en')
clean = clean.loc[flt].copy()

# Drop rows with empty body; log counts
before = len(clean)
clean = clean[clean['body_text'].str.len() >= 30].copy()  # min length
print('Dropped for empty/short body:', before - len(clean))
```

### C3. De-duplicate by content signature (keep latest update)

```python
import hashlib

def content_key(row):
    sig = normalize_ws(row['title'] + ' ' + row['body_text']).lower()
    return hashlib.sha1(sig.encode('utf-8')).hexdigest()

clean['content_key'] = clean.apply(content_key, axis=1)
# Keep most recent updated_at per content_key
clean['updated_at'] = pd.to_datetime(clean['updated_at'])
clean.sort_values(['content_key','updated_at'], ascending=[True, False], inplace=True)
clean = clean.drop_duplicates(subset=['content_key'], keep='first')
clean.drop(columns=['content_key'], inplace=True)
print('Rows after dedupe:', len(clean))
```

**Checkpoint:** Report counts: raw rows → after governance/lang → after empty-body drop → after dedupe.

### C4. Optional API ingestion (simulate service boundary)

```python
%pip install requests orjson

# This code would work if the API is running at http://127.0.0.1:8000
# Uncomment and run if you have the API server started

import requests, time, orjson

API = 'http://127.0.0.1:8000'
page = 1; PAGE_SIZE = 100
api_rows = []
while True:
    try:
        r = requests.get(f"{API}/v1/corpus", params={'page': page, 'page_size': PAGE_SIZE, 'language':'en', 'conf':'public'}, timeout=5)
        if r.status_code != 200:
            time.sleep(1); continue
        batch = r.json()
        if not batch:
            break
        api_rows.extend(batch)
        page += 1
    except Exception as e:
        print(f"API not available: {e}")
        break
        
if api_rows:
    api_df = pd.DataFrame(api_rows)
    print('API fetched rows:', len(api_df))
else:
    print("API ingestion skipped - using CSV-based clean dataframe")

print("API ingestion code ready")
print("For this solution, we'll continue with the CSV-based clean dataframe")```

> You can run either CSV path (`clean`) or API path (`api_df`) through the next steps—the cleaning logic is the same.

---

## Part D — Write JSONL (RAG Chunks + SFT/Eval)

### D1. Chunking helper (overlap to preserve context)

```python
import re

def split_chunks(text: str, max_chars=900, overlap=150):
    text = re.sub(r"\s+", " ", str(text)).strip()
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunk = text[i:end]
        chunks.append(chunk)
        if end == len(text): break
        i = max(0, end - overlap)
    return chunks
```

### D2. CSV→RAG JSONL with provenance

```python
import orjson
from datetime import timezone
from pathlib import Path

rag_path = Path('artifacts/jsonl/rag_chunks_from_csv.jsonl')
with rag_path.open('w', encoding='utf-8') as f:
    for _, r in clean.iterrows():
        chunks = split_chunks(r['body_text'])
        for j, ch in enumerate(chunks):
            rec = {
                'doc_id': r['doc_id'],
                'chunk_id': f"{r['doc_id']}-{j:04d}",
                'text': ch,
                'metadata': {
                    'title': r['title'], 'section': r['section'], 'tags': r['tags'],
                    'source_url': r['source_url'], 'language': r['language'],
                    'confidentiality': r['confidentiality'], 'schema_version': 'rag-chunk-v1'
                }
            }
            f.write(orjson.dumps(rec).decode() + '\n')
rag_path
```

### D3. Build an SFT/Eval JSONL view (instruction → output)

```python
import numpy as np
np.random.seed(0)

sample = clean.sample(min(120, len(clean)), random_state=7)
sft_path = Path('artifacts/jsonl/corpus_sft.jsonl')
with sft_path.open('w', encoding='utf-8') as f:
    for _, r in sample.iterrows():
        prompt = f"Summarize the key steps from: {r['title']} ({r['section']})."
        # Simple templated target; in real life, use human-authored or heuristic extraction
        target = "Key steps: enable SAML; map claims; verify time sync; check audience URI; review settings."
        obj = {'input': prompt, 'output': target,
               'metadata': {'doc_id': r['doc_id'], 'type': r['type'], 'lang': r['language']}}
        f.write(orjson.dumps(obj).decode() + '\n')
sft_path
```

### D4. Validate JSONL & write reviewer samples

```python
import json, itertools

def validate_jsonl(path):
    bad = 0; total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)
                assert isinstance(obj, dict)
            except Exception:
                bad += 1
    return total, bad

for p in ['artifacts/jsonl/rag_chunks_from_csv.jsonl','artifacts/jsonl/corpus_sft.jsonl']:
    t,b = validate_jsonl(p)
    print(p, 'total=', t, 'bad=', b)

# small reviewer sample
sample_path = Path('artifacts/samples/jsonl_samples.jsonl')
with sample_path.open('w', encoding='utf-8') as out:
    for p in ['artifacts/jsonl/rag_chunks_from_csv.jsonl','artifacts/jsonl/corpus_sft.jsonl']:
        with open(p, 'r', encoding='utf-8') as f:
            for line in itertools.islice(f, 3):
                out.write(line)
sample_path
```

**Checkpoint:** Show counts and 3 sample lines. Confirm each line is standalone JSON and includes provenance.

---

## Part E — Wrap-Up

Add a markdown cell and answer:

1. Three reasons **JSONL** is preferred vs a single JSON array for LLM corpora.  
2. What **governance** + **language** filters did you apply, and at what stage?  
3. How did you deduplicate? What key did you use and why?  
4. Which metadata fields are essential for **RAG** vs **SFT** in your outputs?

Export the notebook to HTML. Confirm outputs:

- CSV: `data/corpus_llm.csv` (≈1,000 rows)
- JSONL: `artifacts/jsonl/rag_chunks_from_csv.jsonl`, `artifacts/jsonl/corpus_sft.jsonl`
- Samples: `artifacts/samples/jsonl_samples.jsonl`
- Optional: API running at `http://127.0.0.1:8000` and `api_df` parity with `clean`

---

- **Distinctiveness:** Fully LLM-focused sources + governance filtering + dedupe before JSONL.  
- **Common pitfalls:** Forgetting one-line-per-record; skipping governance filters; missing provenance; deduping by `doc_id` instead of **content signature**; writing giant JSON arrays (OOM).

---

## Solution Snippets (reference)

**Governance + language filter in one pass:**

```python
clean = raw[(raw['confidentiality']=='public') & (raw['language']=='en')].copy()
```

**Content signature for dedupe:**

```python
key = hashlib.sha1((normalize_ws(title+' '+body).lower()).encode()).hexdigest()
```

**Gzip the final JSONL:**

```python
import gzip
with open('artifacts/jsonl/rag_chunks_from_csv.jsonl','r',encoding='utf-8') as src, \
     gzip.open('artifacts/jsonl/rag_chunks_from_csv.jsonl.gz','wt',encoding='utf-8') as dst:
    for line in src: dst.write(line)
```
