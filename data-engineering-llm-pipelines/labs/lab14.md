# Lab 14 — Instruction Tuning Schemas

**Focus Area:** Common instruction‑tuning schemas; templating; line‑wise JSONL generation; validation; dataset hygiene (dedupe, filtering, length control)

> You will transform LLM‑native corpora into instruction‑tuning JSONL using two common layouts—`{instruction, input, output}` and `{prompt, completion}`—plus a templated chatty style. You will implement schema validation, de‑duplication, basic decontamination against eval prompts, and token‑length governance.

---

## Outcomes

By the end of this lab, you will be able to:

1. Explain differences between `{instruction, input, output}` and `{prompt, completion}` schemas and when to choose each.  
2. Convert **RAG chunks** and **SFT views** from Lab 13 into instruction‑tuning JSONL with **stable metadata** and **templated prompts**.  
3. Apply **dataset hygiene**: dedupe, filter by language/governance, and enforce min/max token lengths.  
4. Validate and **sample** final JSONL; compute basic dataset statistics.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `orjson`, `regex`, `numpy`, `pydantic` (for optional validation)  
- JupyterLab or VS Code with Jupyter extension.
- **Artifacts from Lab 13 (required):**
  - RAG chunks: `artifacts/jsonl/rag_chunks_from_csv.jsonl` (or API variant)
  - SFT view: `artifacts/jsonl/corpus_sft.jsonl`
  - (Optional) reviewer samples in `artifacts/samples/`

**Start a notebook:** `week02_lab14.ipynb`

Make sure directories exist:

```python
from pathlib import Path
for p in ['artifacts/jsonl','artifacts/samples','artifacts/stats']:
    Path(p).mkdir(parents=True, exist_ok=True)
```

---

## Part A — Schemas & Templates

### A1. Two common record layouts

- **Schema 1: Trio** *(popular in Flan / Alpaca style)*

  ```json
  {"instruction": "...", "input": "...", "output": "...", "metadata": {"source": "..."}}
  ```

  *Pros:* separates task from context; easy to mix **with** or **without** `input`.  
  *Cons:* needs templating at train time; slightly larger records.

- **Schema 2: Prompt‑Completion** *(OpenAI fine‑tuning style)*

  ```json
  {"prompt": "...", "completion": "...", "metadata": {"source": "..."}}
  ```

  *Pros:* simple; direct fit to APIs expecting prompt/completion.  
  *Cons:* prompt formatting is **baked in**; harder to change templates later.

### A2. Human‑readable templating variants (examples)

- **Header style**

  ```text
  ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n```

- **Chat style** (system/user/assistant)

  ```xml
  <system>You are a helpful assistant.</system>\n<user>{instruction}\n{input}</user>\n<assistant>
  ```

Pick **one** canonical style per dataset for consistency.

---

## Part B — Build a Trio Dataset from Lab 13 SFT View

We’ll transform `corpus_sft.jsonl` into Trio records. (In Lab 13, `input`/`output` pairs are already present—here we standardize and augment metadata.)

### B1. Loader + light cleaner

```python
import json, orjson, re
from pathlib import Path

SRC = Path('artifacts/jsonl/corpus_sft.jsonl')
TRIO = Path('artifacts/jsonl/instruct_trio.jsonl')

lang_allow = {"en"}
min_out_chars = 20

with SRC.open('r', encoding='utf-8') as fin, TRIO.open('w', encoding='utf-8') as fout:
    kept = 0; seen = set()
    for line in fin:
        obj = json.loads(line)
        # Expected keys in Lab 13: input, output, metadata{doc_id,type,lang}
        lang = (obj.get('metadata') or {}).get('lang', 'en')
        if lang not in lang_allow:
            continue
        instruction = obj['input']
        inp = ""  # Lab 13 'input' already contains the task; keep Trio input empty here
        out = re.sub(r"\s+", " ", obj['output']).strip()
        if len(out) < min_out_chars:
            continue
        # Deduplicate by (instruction, out)
        key = (instruction, out)
        if key in seen:
            continue
        seen.add(key)
        rec = {
            'instruction': instruction,
            'input': inp,
            'output': out,
            'metadata': {
                'doc_id': (obj.get('metadata') or {}).get('doc_id'),
                'schema_version': 'trio-v1'
            }
        }
        fout.write(orjson.dumps(rec).decode() + '\n'); kept += 1

kept
```

### B2. Add a templated prompt‑completion view

```python
PROMPT = Path('artifacts/jsonl/instruct_prompt_completion.jsonl')
TEMPLATE = """### Instruction:\n{instruction}\n\n### Response:\n"""

with TRIO.open('r', encoding='utf-8') as fin, PROMPT.open('w', encoding='utf-8') as fout:
    for line in fin:
        t = json.loads(line)
        prompt = TEMPLATE.format(instruction=t['instruction'])
        completion = t['output']
        fout.write(orjson.dumps({'prompt': prompt, 'completion': completion, 'metadata': t.get('metadata')}).decode()+'\n')

PROMPT
```

**Checkpoint:** Print 3 records from both files; explain when you’d pick Trio vs Prompt‑Completion.

---

## Part C — Synthesize Trio from RAG Chunks

You’ll turn **knowledge chunks** (`rag_chunks_from_csv.jsonl`) into simple task pairs like *“Summarize this section”*. This technique is useful for building weakly‑supervised instruction corpora.

### C1. Build summarization tasks from chunks

```python
import json, orjson, re, itertools
from pathlib import Path

RAG = Path('artifacts/jsonl/rag_chunks_from_csv.jsonl')
TRIO_RAG = Path('artifacts/jsonl/instruct_trio_from_rag.jsonl')

min_text_chars = 200   # avoid trivially short chunks
max_text_chars = 1200  # avoid overly long chunks

with RAG.open('r', encoding='utf-8') as fin, TRIO_RAG.open('w', encoding='utf-8') as fout:
    kept = 0; seen = set()
    for line in fin:
        obj = json.loads(line)
        text = re.sub(r"\s+", " ", obj['text']).strip()
        if not (min_text_chars <= len(text) <= max_text_chars):
            continue
        instruction = "Summarize the following section in 2–3 sentences focusing on the key steps and caveats."
        inp = text
        # weak target: headerized bullet summary; in production use a labeler or heuristics
        target = "Key steps: ensure SSO claims are mapped; verify time sync; review settings as noted."
        key = (hash(text) % (10**12))  # approximate dedupe by text hash
        if key in seen:
            continue
        seen.add(key)
        rec = {
            'instruction': instruction,
            'input': inp,
            'output': target,
            'metadata': {
                'doc_id': obj.get('doc_id'),
                'chunk_id': obj.get('chunk_id'),
                'schema_version': 'trio-from-rag-v1'
            }
        }
        fout.write(orjson.dumps(rec).decode() + '\n'); kept += 1

kept
```

### C2. Convert to a chat‑style prompt‑completion

```python
CHAT = Path('artifacts/jsonl/instruct_chat_from_rag.jsonl')
CHAT_TEMPLATE = (
    "<system>You are a concise technical assistant.</system>\n"
    "<user>Instruction: {instruction}\n\nContext: {context}</user>\n"
    "<assistant>"
)

with TRIO_RAG.open('r', encoding='utf-8') as fin, CHAT.open('w', encoding='utf-8') as fout:
    for line in fin:
        t = json.loads(line)
        prompt = CHAT_TEMPLATE.format(instruction=t['instruction'], context=t['input'])
        completion = t['output']
        fout.write(orjson.dumps({'prompt': prompt, 'completion': completion, 'metadata': t.get('metadata')}).decode()+'\n')
CHAT
```

**Checkpoint:** Compare token length distributions of `prompt` and `completion` to ensure they stay within your model’s budget.

---

## Part D — Validation, Length Governance, and Hygiene

### D1. Pydantic validators (optional but recommended)

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict

class Trio(BaseModel):
    instruction: str
    input: str
    output: str
    metadata: Optional[Dict] = Field(default_factory=dict)

class PC(BaseModel):
    prompt: str
    completion: str
    metadata: Optional[Dict] = Field(default_factory=dict)

# Validate a sample
try:
    _ = Trio(instruction='Do X', input='', output='Done', metadata={'doc_id':'DOC-0001'})
    _ = PC(prompt='### Instruction:\nDo X\n\n### Response:\n', completion='Done')
    print('Validation OK')
except ValidationError as e:
    print(e)
```

### D2. Length filters and simple decontamination

```python
import json, numpy as np
from pathlib import Path

# Token proxy: count words as a rough stand‑in
def token_proxy(s):
    return max(1, len(s.split()))

MAX_PROMPT_TOK = 700
MAX_COMP_TOK = 350

INP = Path('artifacts/jsonl/instruct_prompt_completion.jsonl')
OUT = Path('artifacts/jsonl/instruct_prompt_completion_cleansed.jsonl')

# Suppose we have eval prompts to avoid leakage
EVAL = {"How many orders does customer", "Summarize the key steps from:"}

kept = 0
with INP.open('r', encoding='utf-8') as fin, OUT.open('w', encoding='utf-8') as fout:
    for line in fin:
        obj = json.loads(line)
        p, c = obj['prompt'], obj['completion']
        if any(trigger in p for trigger in EVAL):
            continue  # decontaminate against eval set
        if token_proxy(p) > MAX_PROMPT_TOK or token_proxy(c) > MAX_COMP_TOK:
            continue
        fout.write(line); kept += 1
kept
```

### D3. Dataset stats

```python
import json, numpy as np
from pathlib import Path

STATS = Path('artifacts/stats/instruct_stats.json')
counts = {'trio':0,'pc':0,'chat_pc':0}
lengths = {'prompt':[], 'completion':[]}

for p in [
    'artifacts/jsonl/instruct_trio.jsonl',
    'artifacts/jsonl/instruct_prompt_completion.jsonl',
    'artifacts/jsonl/instruct_chat_from_rag.jsonl']:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if 'instruction' in obj:
                    counts['trio'] += 1
                if 'prompt' in obj and 'completion' in obj:
                    counts['pc'] += 1
                    lengths['prompt'].append(len(obj['prompt'].split()))
                    lengths['completion'].append(len(obj['completion'].split()))
    except FileNotFoundError:
        pass

import json
STATS.write_text(json.dumps({
    'counts': counts,
    'prompt_tokens_mean': float(np.mean(lengths['prompt'])) if lengths['prompt'] else 0.0,
    'completion_tokens_mean': float(np.mean(lengths['completion'])) if lengths['completion'] else 0.0
}, indent=2))

STATS
```

---

## Part E — Wrap‑Up

Add a markdown cell and answer:

1. For your project, which schema will you choose and **why** (Trio vs Prompt‑Completion vs Chat)?  
2. Show an example template you’ll use at training time, and explain how you would change it without rewriting your dataset.  
3. What thresholds did you choose for length filtering? How would these change for a smaller/larger context model?  
4. Where in your pipeline will you run **decontamination** against eval/holdout tasks?

Confirm outputs:

- `artifacts/jsonl/instruct_trio.jsonl`  
- `artifacts/jsonl/instruct_prompt_completion.jsonl`  
- `artifacts/jsonl/instruct_trio_from_rag.jsonl`  
- `artifacts/jsonl/instruct_chat_from_rag.jsonl`  
- `artifacts/jsonl/instruct_prompt_completion_cleansed.jsonl`  
- `artifacts/stats/instruct_stats.json`

---

- **Continuity:** Uses Lab 13 outputs directly; reinforces JSONL discipline and adds instruction‑tuning specifics.  
- **Common pitfalls:** Mixing templates in one dataset; not separating `instruction` vs `input`; forgetting decontamination; ignoring length budgets; leaking governance‑restricted content.

---

## Solution Snippets (reference)

**Switch Trio→Prompt‑Completion on the fly:**

```python
prompt = f"### Instruction:\n{t['instruction']}\n\n### Input:\n{t['input']}\n\n### Response:\n"
completion = t['output']
```

**Chat template variant:**

```python
prompt = (
  "<system>You are a helpful assistant.</system>\n"
  f"<user>{t['instruction']}\n{t['input']}</user>\n"
  "<assistant>"
)
```

**Guard against empty/very short texts:**

```python
if not t['output'] or len(t['output'].split()) < 5: continue
```
