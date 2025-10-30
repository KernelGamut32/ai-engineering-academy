# Lab 17 — Few‑Shot Prompting Patterns

**Focus Area:** **Few‑shot** prompt construction; consistent exemplar formatting; exemplar selection from prior JSONL artifacts (Labs 13–16)

> This lab **builds on Labs 13–16**. You will create a reusable **exemplar bank** from your instruction datasets, learn to format **few‑shot** prompts consistently, and implement simple **similarity‑based selection** to choose the best K shots per query — all **offline**. We’ll use TF‑IDF as a light‑weight stand‑in for embedding similarity.

---

## Outcomes

By the end of this lab, you will be able to:

1. Define **few‑shot prompting** and list cases where it helps vs hurts.  
2. Build a curated **exemplar bank** from `{instruction,input,output}` or `{prompt,completion}` JSONL.  
3. Implement **consistent templates** (header style / chat style) and enforce schema checks.  
4. Select K exemplars using **TF‑IDF cosine similarity** and assemble a final prompt with a deterministic layout.  
5. Evaluate prompt length (token proxy) and produce a compact **prompt pack** JSONL for downstream use.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `orjson`, `regex`, `scikit-learn`, `tokenizers` (reuse tiny tokenizer from **Lab 15**)  
- JupyterLab or VS Code with Jupyter extension.
- **Artifacts from earlier labs:**  
  - `artifacts/jsonl/instruct_trio.jsonl` **and/or** `artifacts/jsonl/instruct_prompt_completion_cleansed.jsonl` (Lab 14)  
  - Optional: `artifacts/rag/rag_*_*.jsonl` (Lab 16) for building domain‑matched shots

**Start a notebook:** `week02_lab17.ipynb`

Create folders:

```python
from pathlib import Path
for p in ['artifacts/prompts','artifacts/samples','artifacts/stats']:
    Path(p).mkdir(parents=True, exist_ok=True)
```

---

## Part A — What is Few‑Shot Prompting?

Add a markdown cell and answer:

- Definition: conditioning the model with **formatted exemplars** (input→output pairs) that resemble the target task.  
- Helps when: task is under‑specified; model needs **style**/format cues; domain jargon.  
- Hurts when: exemplars are **off‑domain**, inconsistent formatting, or too many tokens crowd out problem context.

---

## Part B — Build an Exemplar Bank

### B1. Load from Lab 14 (Trio and/or Prompt‑Completion)

```python
import json, orjson, pandas as pd
from pathlib import Path

# Prefer Trio; fall back to prompt-completion
path_trio = Path('artifacts/jsonl/instruct_trio.jsonl')
path_pc   = Path('artifacts/jsonl/instruct_prompt_completion_cleansed.jsonl')

rows = []
if path_trio.exists():
    for line in open(path_trio, 'r', encoding='utf-8'):
        o = json.loads(line)
        rows.append({
            'instruction': o.get('instruction','').strip(),
            'input': o.get('input','').strip(),
            'output': o.get('output','').strip(),
            'meta': o.get('metadata',{})
        })
elif path_pc.exists():
    for line in open(path_pc, 'r', encoding='utf-8'):
        o = json.loads(line)
        # Heuristic: split PC into instruction/empty input
        rows.append({
            'instruction': o.get('prompt','').strip(),
            'input': '',
            'output': o.get('completion','').strip(),
            'meta': o.get('metadata',{})
        })
else:
    raise FileNotFoundError('No instruction JSONL found from Lab 14')

bank = pd.DataFrame(rows)
len(bank), bank.head(2)
```

### B2. Light cleaning + filtering

```python
import regex as re

MIN_OUT_WORDS = 5

def norm(s: str) -> str:
    # Keep meaning-bearing punctuation; just collapse whitespace
    return re.sub(r"\s+", " ", s or '').strip()

bank['instruction'] = bank['instruction'].map(norm)
bank['input'] = bank['input'].map(norm)
bank['output'] = bank['output'].map(norm)

bank = bank[(bank['instruction']!='') & (bank['output'].str.split().str.len()>=MIN_OUT_WORDS)].copy()
len(bank)
```

### B3. Persist as an **exemplar bank** (JSONL)

```python
from pathlib import Path
ex_path = Path('artifacts/prompts/exemplar_bank.jsonl')
with ex_path.open('w', encoding='utf-8') as f:
    for rec in bank[['instruction','input','output']].to_dict(orient='records'):
        f.write(orjson.dumps(rec).decode() + '\n')
ex_path
```

**Checkpoint:** Why separate a curated bank? → Enables auditing, dedupe, and stable few‑shot packs across experiments.

---

## Part C — Consistent Templates

### C1. Header template (Trio style)

```python
HDR_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

HDR_NO_INPUT_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)
```

### C2. Chat template (system/user/assistant)

```python
CHAT_TMPL = (
    "<system>You are a concise technical assistant.</system>\n"
    "<user>Instruction: {instruction}\n{input_block}</user>\n"
    "<assistant>"
)

def render_chat(instruction: str, input_text: str) -> str:
    input_block = ("\nContext: " + input_text) if input_text else ''
    return CHAT_TMPL.format(instruction=instruction, input_block=input_block)
```

### C3. Schema check utility

```python
from typing import Dict

def trio(rec: Dict):
    assert 'instruction' in rec and 'output' in rec
    return rec['instruction'], rec.get('input',''), rec['output']

# Example
rec0 = bank.iloc[0].to_dict()
trio(rec0)
```

---

## Part D — Select K Exemplars via TF‑IDF Similarity

### D1. Fit a TF‑IDF model on exemplar instructions (+inputs)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = (bank['instruction'] + ' ' + bank['input']).tolist()
vec = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
X = vec.fit_transform(corpus)
X.shape
```

### D2. Exemplar selector

```python
import numpy as np

def topk_exemplars(query_instruction: str, k: int = 3):
    qv = vec.transform([query_instruction])
    sims = cosine_similarity(qv, X)[0]
    idxs = np.argsort(-sims)[:k]
    return bank.iloc[idxs][['instruction','input','output']].to_dict(orient='records')

# Smoke test
topk_exemplars('Summarize SSO setup steps', k=3)
```

### D3. Assemble a **few‑shot** prompt (header style)

```python
from tokenizers import Tokenizer

# token length proxy to keep under budget
_tok = Tokenizer.from_file('artifacts/tokenizer/bytebpe.json')

def tok_len(s: str) -> int:
    return len(_tok.encode(s).ids)

def build_fewshot_prompt(task_instruction: str, task_input: str = '', k: int = 3, mode: str = 'header'):
    shots = topk_exemplars(task_instruction, k=k)
    parts = []
    for ex in shots:
        if mode == 'header':
            parts.append(HDR_TMPL.format(instruction=ex['instruction'], input=ex.get('input','')) + ex['output'] + "\n\n")
        else:
            parts.append(render_chat(ex['instruction'], ex.get('input','')) + ex['output'] + "\n\n")
    # final task
    if mode == 'header':
        parts.append(HDR_TMPL.format(instruction=task_instruction, input=task_input))
    else:
        parts.append(render_chat(task_instruction, task_input))
    prompt = ''.join(parts)
    return prompt, shots

p, shots = build_fewshot_prompt('Summarize the key steps for enabling SAML SSO', task_input='AcmeCorp IdP')
print(p[:600] + '...')
print('approx tokens:', tok_len(p))
```

### D4. Persist as a **prompt pack** JSONL

```python
import orjson
from pathlib import Path

pack_path = Path('artifacts/prompts/fewshot_prompt_pack.jsonl')
with pack_path.open('w', encoding='utf-8') as f:
    # Create a few trial tasks
    tasks = [
        {'instruction':'Summarize SSO setup steps','input':'', 'k':3},
        {'instruction':'Explain rate limits policy','input':'', 'k':4},
        {'instruction':'Create a troubleshooting checklist for SAML claims','input':'', 'k':3},
    ]
    for t in tasks:
        prompt, used = build_fewshot_prompt(t['instruction'], t.get('input',''), k=t['k'])
        rec = {'prompt': prompt, 'metadata': {'k': t['k'], 'approx_tokens': tok_len(prompt), 'used': used}}
        f.write(orjson.dumps(rec).decode() + '\n')
pack_path
```

**Checkpoint:** Inspect the pack; verify consistent formatting across all exemplars and the final instruction block.

---

## Part E — Wrap‑Up

Add a markdown cell and answer:

1. In your domain, how many shots (K) strike the best balance between **structure signaling** and **context budget**?  
2. When would you prefer **chat** templating vs **header** templating?  
3. What filters would you add (e.g., domain tags, language) to keep exemplars **on‑topic**?  
4. How will you monitor **prompt drift** (shots degrading over time) as your corpus evolves?

Confirm outputs:

- `artifacts/prompts/exemplar_bank.jsonl`  
- `artifacts/prompts/fewshot_prompt_pack.jsonl`  
- Any notes on token budgets and template choice

---

- **Common pitfalls:** Inconsistent templates; off‑domain or contradictory shots; exceeding token limits; repeating nearly identical examples.

---

## Solution Snippets (reference)

**Switch to chat style few‑shot:**

```python
p, _ = build_fewshot_prompt(
    'Draft a privacy FAQ outline for exports',
    mode='chat',
    k=3
)
```

**Constrain by domain tag (if present in metadata):**

```python
# Filter bank by a domain key first
mask = bank['instruction'].str.contains('SSO|SAML|claims', case=False, na=False)
bank_dom = bank[mask].reset_index(drop=True)
```

**Guardrail: drop shots with long outputs (>200 tokens):**

```python
bank = bank[bank['output'].map(lambda s: tok_len(s) <= 200)].copy()
```
