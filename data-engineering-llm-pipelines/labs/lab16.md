# Lab 16 — RAG Data Prep: Chunking, Overlap, and Metadata

**Focus Area:** Chunking strategies (sentences, headings, tokens), overlap tuning, metadata (source, timestamp, section), and careful text cleaning (retain meaning‑bearing punctuation)

> This lab **builds on Labs 13-15**. You will transform your cleaned LLM corpus into a high‑quality RAG index input by experimenting with **three chunkers** (sentence‑window, heading‑aware, token‑bounded), adding provenance‑rich metadata, and validating that punctuation needed for semantics (e.g., dates, decimals, clause boundaries) is preserved.

---

## Outcomes

By the end of this lab, you will be able to:

1. Implement **sentence‑window**, **heading‑aware**, and **token‑bounded** chunkers with configurable **overlap**.  
2. Attach consistent **metadata** (source, filename/URI, section, timestamps, schema version) to each chunk.  
3. Apply **cleaners** that normalize whitespace without removing **meaning‑bearing punctuation**.  
4. Generate **JSONL shards** for embeddings/retrieval and validate them line‑by‑line.  
5. Compare chunk distributions and select defaults for your project.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `regex`, `orjson`, `numpy`, `tqdm`, `tokenizers` (re‑use tiny BPE from **Lab 15**)  
- **Artifacts from prior labs:**  
  - `artifacts/jsonl/rag_chunks_from_csv.jsonl` (Lab 13) — optional baseline  
  - `artifacts/tokenizer/bytebpe.json` (Lab 15) — used for token counting  
  - (If starting from raw text) `data/text/` directory as in Lab 13

**Start a notebook:** `week02_lab16.ipynb`

Create directories:

```python
from pathlib import Path
for p in ['artifacts/rag','artifacts/samples','artifacts/stats']:
    Path(p).mkdir(parents=True, exist_ok=True)
```

---

## Part A — Cleaners that Preserve Meaning

### A1. Normalize whitespace but **keep** punctuation that carries semantics

```python
import regex as re

def normalize_text(s: str) -> str:
    # Collapse whitespace but keep punctuation (.,:;?!%$-@/&) and digits
    # Preserve newlines as soft boundaries for heading detection
    s = s.replace('\r', '')
    # Normalize multiple spaces but keep single spaces and newlines
    s = re.sub(r"[\t ]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    # Strip leading/trailing whitespace lines
    s = s.strip('\n ')
    return s
```

### A2. Quick before/after sanity

```python
raw = "Price was $1,234.50 on 2025-01-03\n\nSee Section 2.1: Rate Limits? Yes!"
print(normalize_text(raw))
```

> **Why:** In RAG, punctuation like decimals, hyphens, colons, and question marks often **change meaning**. Avoid `re.sub(r"\W", " ", text)`‑style cleaners that strip them.

---

## Part B — Chunkers

We’ll implement three chunking strategies with **overlap**. Choose one as your default and compare stats.

### B1. Sentence‑window chunker (regex‑based)

```python
import regex as re
from typing import List

# Lightweight sentence splitter (no external models)
sent_split = re.compile(r"(?<=\S[\.!?])\s+(?=[A-Z0-9])")

def to_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    sents = sent_split.split(text)
    # Merge very short trailing sentences into previous
    merged = []
    for s in sents:
        s = s.strip()
        if not s: continue
        if merged and len(s) < 40:  # chars
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged

# Sliding window over sentences
from itertools import islice

def chunk_by_sentences(text: str, window: int = 5, overlap: int = 2):
    sents = to_sentences(text)
    i = 0
    while i < len(sents):
        end = min(i + window, len(sents))
        yield " ".join(sents[i:end])
        if end == len(sents):
            break
        i = max(0, end - overlap)
```

### B2. Heading‑aware chunker (Markdown/plaintext)

```python
import regex as re

heading_re = re.compile(r"^(?:\s*#+\s+.+|\s*[A-Z][A-Z0-9 .:/-]{3,}$)", re.M)

def split_by_headings(text: str):
    text = normalize_text(text)
    # Find heading spans
    spans = [(m.start(), m.end()) for m in heading_re.finditer(text)]
    if not spans:
        return [text]
    chunks = []
    pos = 0
    for start, end in spans:
        if start > pos:
            chunks.append(text[pos:start].strip('\n '))
        pos = start
    chunks.append(text[pos:].strip('\n '))
    # Post‑process: attach small sections to neighbors
    merged = []
    for ch in chunks:
        if not ch: continue
        if merged and len(ch) < 300:
            merged[-1] += "\n" + ch
        else:
            merged.append(ch)
    return merged

# Add overlap at paragraph granularity
def chunk_by_headings(text: str, overlap_chars: int = 1500, max_chars: int = 3000):
    for section in split_by_headings(text):
        start = 0
        while start < len(section):
            end = min(start + max_chars, len(section))
            yield section[start:end]
            if end == len(section):
                break
            start = max(0, end - overlap_chars)
```

### B3. Token‑bounded chunker (uses tokenizer from Lab 15)

```python
from tokenizers import Tokenizer
from pathlib import Path

tok = Tokenizer.from_file('artifacts/tokenizer/bytebpe.json')

def token_len(s: str) -> int:
    return len(tok.encode(s).ids)

# Greedy grow with token overlap

def chunk_by_tokens(text: str, max_tokens: int = 300, overlap_tokens: int = 60):
    text = normalize_text(text)
    words = text.split(' ')
    i = 0
    while i < len(words):
        cur = []
        cur_len = 0
        j = i
        while j < len(words):
            candidate = (" ".join(cur + [words[j]])).strip()
            if token_len(candidate) > max_tokens:
                break
            cur.append(words[j]); j += 1
        chunk = " ".join(cur)
        if not chunk:
            # fallback: force‑add a single long word
            chunk = words[j]
            j += 1
        yield chunk
        if j >= len(words):
            break
        # step back by overlap tokens
        # approximate by words: walk backward until token budget ~ overlap
        back = 0
        while back < len(cur) and token_len(" ".join(cur[-(back+1):])) < overlap_tokens:
            back += 1
        i = max(i + len(cur) - back, i + 1)
```

---

## Part C — Build RAG Chunks + Metadata + Shards

We’ll read from **either** a directory of text files (`data/text/`) or prior **Lab 13** JSONL (use its `text` and `metadata.source`).

### C1. Source loader (text dir or JSONL)

```python
from pathlib import Path
import json, orjson

USE_JSONL = Path('artifacts/jsonl/rag_chunks_from_csv.jsonl').exists()

def iter_documents():
    if USE_JSONL:
        # Each line: {doc_id, chunk_id?, text?, metadata{source?...}}
        with open('artifacts/jsonl/rag_chunks_from_csv.jsonl','r',encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                yield obj.get('metadata',{}).get('source', 'jsonl://rag_chunks_from_csv'), obj.get('text','')
    else:
        for path in Path('data/text').glob('*'):
            yield str(path), Path(path).read_text(encoding='utf-8', errors='ignore')
```

### C2. Choose a chunker & parameters

```python
CHUNKER = 'tokens'   # choose: 'sentences' | 'headings' | 'tokens'
PARAMS  = {
    'sentences': {'window': 6, 'overlap': 2},
    'headings':  {'max_chars': 2600, 'overlap_chars': 500},
    'tokens':    {'max_tokens': 320, 'overlap_tokens': 64}
}
```

### C3. Emit JSONL shards with provenance

```python
from datetime import datetime, timezone
import hashlib
from tqdm import tqdm

schema = 'rag-chunk-v2'
max_shard_bytes = 50_000_000
cur = 0; idx = 0

out = open(f'artifacts/rag/rag_{CHUNKER}_{idx:03d}.jsonl','w',encoding='utf-8')

def write_line(obj):
    global cur, idx, out
    line = orjson.dumps(obj).decode() + '\n'
    if cur + len(line.encode('utf-8')) > max_shard_bytes:
        out.close(); idx += 1; cur = 0
        out = open(f'artifacts/rag/rag_{CHUNKER}_{idx:03d}.jsonl','w',encoding='utf-8')
    out.write(line); cur += len(line.encode('utf-8'))

for source, text in tqdm(iter_documents()):
    text = normalize_text(text)
    # choose chunker
    if CHUNKER == 'sentences':
        chunks = list(chunk_by_sentences(text, **PARAMS['sentences']))
    elif CHUNKER == 'headings':
        chunks = list(chunk_by_headings(text, **PARAMS['headings']))
    else:
        chunks = list(chunk_by_tokens(text, **PARAMS['tokens']))

    # stable doc id from source path
    doc_id = hashlib.sha1(source.encode('utf-8')).hexdigest()[:16]
    for j, ch in enumerate(chunks):
        meta = {
            'source': source,
            'schema_version': schema,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'chunk_index': j,
            'n_chars': len(ch),
            'chunker': CHUNKER
        }
        rec = {
            'doc_id': doc_id,
            'chunk_id': f'{doc_id}-{j:04d}',
            'text': ch,
            'metadata': meta
        }
        write_line(rec)

out.close()
```

**Checkpoint:** Count shards/lines and show 3 example chunks with metadata.

```python
from glob import glob
import itertools

files = sorted(glob(f'artifacts/rag/rag_{CHUNKER}_*.jsonl'))
print('SHARDS:', files)
lines = sum(1 for p in files for _ in open(p,'r',encoding='utf-8'))
print('TOTAL LINES:', lines)

with open(files[0],'r',encoding='utf-8') as f:
    print('\n'.join(list(itertools.islice(f, 3))))
```

---

## Part D — Validate, Compare, and Choose Defaults

### D1. Line validator (presence + minimal length)

```python
import json

def validate_jsonl(path):
    total = bad = 0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
                ok = isinstance(obj, dict) and 'text' in obj and 'metadata' in obj and len(obj['text']) >= 40
                if not ok: bad += 1
            except Exception:
                bad += 1
    return total, bad

stats = []
for p in files:
    t,b = validate_jsonl(p)
    stats.append({'file': p, 'total': t, 'bad': b})
stats
```

### D2. Distribution stats by chunker

```python
import json, numpy as np
from collections import Counter

lens = []
for p in files:
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            lens.append(len(obj['text']))

import math
summary = {
    'count': len(lens),
    'mean_chars': float(np.mean(lens)) if lens else 0,
    'p50': float(np.percentile(lens, 50)) if lens else 0,
    'p90': float(np.percentile(lens, 90)) if lens else 0,
    'p99': float(np.percentile(lens, 99)) if lens else 0,
}
summary
```

### D3. Token budget check (optional)

```python
from tokenizers import Tokenizer
import numpy as np

tok = Tokenizer.from_file('artifacts/tokenizer/bytebpe.json')

sample_tokens = []
for p in files[:1]:
    with open(p,'r',encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200: break
            text = json.loads(line)['text']
            sample_tokens.append(len(tok.encode(text).ids))

{'mean_tokens': float(np.mean(sample_tokens)), 'p95_tokens': float(np.percentile(sample_tokens,95))}
```

**Checkpoint:** Using your model’s max context, decide: Do you need shorter chunks or less overlap? Note the trade‑off: retrieval recall vs embedding cost.

---

## Wrap‑Up

Add a markdown cell and answer:

1. Which chunker and parameters did you choose (window/overlap or max_tokens/overlap_tokens) and **why**?  
2. Which metadata fields are essential for your retriever (e.g., `source`, `section`, `created_at`)?  
3. Show validator results (bad/total). What rules would you tighten before production?  
4. If a downstream embedder requires sentence boundaries, how would you modify your chunker?

Confirm outputs:

- `artifacts/rag/rag_<CHUNKER>_*.jsonl`  
- `artifacts/samples/` (optional samples you created)  
- `artifacts/stats/` (optional summary you saved)

---

- **Common pitfalls:** Over‑aggressive cleaning that kills punctuation; no overlap leading to context loss; metadata gaps; shards without validation.

---

## Solution Snippets (reference)

**Switch chunkers quickly:**

```python
for CHUNKER in ['sentences','headings','tokens']:
    PARAMS = {
        'sentences': {'window': 6, 'overlap': 2},
        'headings':  {'max_chars': 2600, 'overlap_chars': 500},
        'tokens':    {'max_tokens': 320, 'overlap_tokens': 64}
    }
    # re‑run Part C with chosen CHUNKER
```

**Sample a few lines from every shard:**

```python
import itertools, glob
for p in sorted(glob.glob('artifacts/rag/rag_*_*.jsonl')):
    with open(p,'r',encoding='utf-8') as f:
        print(p); print(''.join(list(itertools.islice(f,2))))
```

**Minimal dedupe by content hash (optional):**

```python
import hashlib, json
seen = set()

def dedupe_jsonl(inp, out):
    with open(inp,'r',encoding='utf-8') as f, open(out,'w',encoding='utf-8') as g:
        for line in f:
            obj = json.loads(line)
            key = hashlib.sha1(obj['text'].strip().lower().encode('utf-8')).hexdigest()
            if key in seen: continue
            seen.add(key)
            g.write(line)
# dedupe_jsonl('artifacts/rag/rag_tokens_000.jsonl','artifacts/rag/rag_tokens_000_dedup.jsonl')
```
