# Lab 15 — Loading JSONL with 🤗 Datasets

**Focus Area:** Load **JSONL** from Lab 13/14; **map/tokenize**; **shuffle & batch**; **train/test splits**; save reusable Arrow caches

> This lab **builds on Lab 13 & 14 artifacts**. You will use 🤗 **datasets** to load JSONL files, create splits, and (optionally) tokenize **without internet** using a tiny local tokenizer. You’ll finish with batched, shuffled, memory‑efficient datasets ready for model training or evaluation.

---

## Outcomes

By the end of this lab, you will be able to:

1. Load line‑delimited **JSONL** into `datasets.Dataset` and `DatasetDict`.  
2. Create **deterministic** train/validation/test splits and **shuffle** with a fixed seed.  
3. Apply **batched transforms** via `map` (cleaning, schema alignment, tokenization).  
4. Batch with **dynamic padding** and persist datasets to disk for reuse.

---

## Prerequisites & Setup

- Python 3.13 with `datasets`, `pandas`, `orjson`, `numpy`, `tokenizers` (for local tokenizer), `torch` (or `numpy` if not using PyTorch)  
- JupyterLab or VS Code with Jupyter extension.
- **Artifacts from previous labs (required):**
  - `artifacts/jsonl/instruct_prompt_completion.jsonl` **or** `instruct_prompt_completion_cleansed.jsonl` (Lab 14)
  - `artifacts/jsonl/instruct_trio.jsonl` (Lab 14) — optional
  - `artifacts/jsonl/rag_chunks_from_csv.jsonl` (Lab 13) — optional

**Start a notebook:** `week02_lab15.ipynb`

Create working dirs:

```python
from pathlib import Path
for p in ['artifacts/hf_cache','artifacts/tokenizer','artifacts/datasets','artifacts/samples']:
    Path(p).mkdir(parents=True, exist_ok=True)
```

> **No internet assumption:** We’ll **train a tiny tokenizer locally** using `tokenizers` to avoid downloads. If you already have a local tokenizer (e.g., a copied `gpt2` tokenizer folder), you can use that instead.

---

## Part A — Load JSONL with Datasets

### A1. Pick your source JSONL

Choose **one** main file (prefer cleansed prompt‑completion):

```python
# Create a larger test dataset with at least 10 examples
test_data = [
    {"prompt": "What is AI?", "completion": "AI stands for Artificial Intelligence."},
    {"prompt": "Explain machine learning", "completion": "Machine learning is a subset of AI that learns from data."},
    {"prompt": "What is deep learning?", "completion": "Deep learning uses neural networks with multiple layers."},
    {"prompt": "Define NLP", "completion": "Natural Language Processing helps computers understand human language."},
    {"prompt": "What is computer vision?", "completion": "Computer vision enables machines to interpret visual information."},
    {"prompt": "Explain reinforcement learning", "completion": "Reinforcement learning trains agents through rewards and penalties."},
    {"prompt": "What is supervised learning?", "completion": "Supervised learning uses labeled data to train models."},
    {"prompt": "Define unsupervised learning", "completion": "Unsupervised learning finds patterns in unlabeled data."},
    {"prompt": "What is a neural network?", "completion": "A neural network is inspired by biological neurons and processes information."},
    {"prompt": "Explain gradient descent", "completion": "Gradient descent optimizes model parameters by minimizing loss."},
]

import json
with open('artifacts/jsonl/instruct_prompt_completion.jsonl', 'w', encoding='utf-8') as f:
    for record in test_data:
        f.write(json.dumps(record) + '\n')

print(f"Created test JSONL file with {len(test_data)} examples")
```

### A2. Load as Dataset

```python
%pip install datasets

from pathlib import Path
from datasets import load_dataset

# Define the source file (created in the previous cell)
SRC = Path('artifacts/jsonl/instruct_prompt_completion.jsonl')

# Verify file exists before loading
if not SRC.exists():
    raise FileNotFoundError(f"JSONL file not found at {SRC}. Run the previous cell to create it.")

ds = load_dataset('json', data_files=str(SRC), split='train')
ds
```

### A3. Quick sanity checks

```python
# Peek at schema & first rows
print(ds.features)
ds.select(range(2))

# Basic counts
print('num_rows =', ds.num_rows)
print('columns  =', ds.column_names)
```

> Tip: `datasets` streams from Arrow cache on disk; transforms are lazy until materialized. Fast, memory‑efficient.

---

## Part B — Deterministic Splits & Shuffling

### B1. Canonical column alignment

Ensure required columns exist and are strings; add metadata fallback.

```python
import numpy as np

def ensure_schema(batch):
    # Expect {prompt, completion, metadata?}
    prompts = batch.get('prompt')
    completions = batch.get('completion')
    inputs = batch.get('input')
    outputs = batch.get('output')
    # Align Trio → prompt/completion if needed
    if (prompts is None or completions is None) and inputs is not None and outputs is not None:
        prompts = [f"### Instruction:\n{ins}\n\n### Response:\n" for ins in inputs]
        completions = outputs
    return {
        'prompt': list(map(lambda x: '' if x is None else str(x), prompts)),
        'completion': list(map(lambda x: '' if x is None else str(x), completions)),
        'metadata': batch.get('metadata') or [{}]*len(prompts)
    }

ds_aligned = ds.map(ensure_schema, batched=True, remove_columns=ds.column_names)
ds_aligned = ds_aligned.filter(lambda ex: len(ex['prompt'])>0 and len(ex['completion'])>0)
ds_aligned
```

### B2. Train/validation/test split (deterministic)

```python
seed = 13

# Check if dataset is large enough for splits
if len(ds_aligned) < 10:
    print(f"⚠️  Warning: Only {len(ds_aligned)} examples. Creating minimal splits.")
    # For very small datasets, just use different percentages
    train_test = ds_aligned.train_test_split(test_size=0.3, seed=seed)
    
    # If we have at least 3 examples in test, split it further
    if len(train_test['test']) >= 2:
        val_test = train_test['test'].train_test_split(test_size=0.5, seed=seed)
        dsd = {
            'train': train_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        }
    else:
        # Too small for 3-way split, just use train/test
        print("⚠️  Dataset too small for validation split. Using train/test only.")
        dsd = {
            'train': train_test['train'],
            'test': train_test['test']
        }
else:
    # Normal splits for larger datasets
    train_test = ds_aligned.train_test_split(test_size=0.2, seed=seed)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    
    dsd = {
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }

from datasets import DatasetDict
dds = DatasetDict(dsd)

# Shuffle each split with same seed for determinism
dds = DatasetDict({k: v.shuffle(seed=seed) for k,v in dds.items()})
dds
```

### B3. Persist splits to disk (Arrow)

```python
dds.save_to_disk('artifacts/datasets/instruct_pc_splits')
# Load back later with:  datasets.load_from_disk(...)
```

---

## Part C — Tokenization (Offline) with `tokenizers`

We’ll **train a tiny Byte‑Level BPE tokenizer** on the prompts to avoid network downloads.

### C1. Export prompt text for training the tokenizer

```python
from pathlib import Path
train_txt = Path('artifacts/tokenizer/train_prompts.txt')
with train_txt.open('w', encoding='utf-8') as f:
    for ex in dds['train']:
        f.write(ex['prompt'].replace('\n','\n') + '\n')
train_txt, train_txt.stat().st_size
```

### C2. Train a small tokenizer

```python
%pip install tokenizers

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

vocab_size = 8_000  # small for classroom speed
unk_token = "<unk>"

_tok = Tokenizer(BPE(unk_token=unk_token))
_tok.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[unk_token, "<pad>", "<bos>", "<eos>"])
_tok.train([str(train_txt)], trainer)
_tok.post_processor = ByteLevelProcessor()
_tok.decoder = ByteLevelDecoder()
_tok.save('artifacts/tokenizer/bytebpe.json')
```

### C3. Wrap as a simple encode function for `datasets.map`

```python
import numpy as np
from functools import partial

from tokenizers import Tokenizer
tok = Tokenizer.from_file('artifacts/tokenizer/bytebpe.json')
pad_id = tok.token_to_id('<pad>')
bos_id = tok.token_to_id('<bos>')
eos_id = tok.token_to_id('<eos>')

max_len = 512

def encode_batch(batch):
    ids = []
    attn = []
    for p, c in zip(batch['prompt'], batch['completion']):
        text = p + c  # simple concat; for chat models you may add BOS/EOS markers between
        enc = tok.encode(text)
        input_ids = enc.ids[:max_len-1] + [eos_id if eos_id is not None else 0]
        # pad
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id]*pad_len
        attention_mask = [1]*min(len(enc.ids)+1, max_len) + [0]*max(0, max_len - (min(len(enc.ids)+1, max_len)))
        ids.append(input_ids)
        attn.append(attention_mask)
    return {'input_ids': ids, 'attention_mask': attn}
```

### C4. Apply batched tokenization with `map`

```python
batched_enc = dds.map(encode_batch, batched=True, batch_size=256, remove_columns=dds['train'].column_names)
batched_enc
```

### C5. Set format for PyTorch and create DataLoaders (optional)

```python
%pip install torch

import torch
from torch.utils.data import DataLoader

for split in batched_enc:
    batched_enc[split].set_format(type='torch', columns=['input_ids','attention_mask'])

train_loader = DataLoader(batched_enc['train'], batch_size=8, shuffle=True)
val_loader   = DataLoader(batched_enc['validation'], batch_size=8)

# Sanity: iterate one batch
xb = next(iter(train_loader))
xb['input_ids'].shape, xb['attention_mask'].shape
```

> If you prefer **numpy** only, call `set_format(type='numpy', columns=...)` and iterate normally.

---

## Part D — Advanced: Map Pipelines, Filtering & Batching

### D1. Length filtering before/after tokenization

```python
def too_short(ex):
    return len(ex['completion'].split()) >= 5

filtered = dds.filter(too_short)
filtered = DatasetDict({k: v.shuffle(seed=7) for k,v in filtered.items()})
```

### D2. Compose multiple maps

```python
from datasets import DatasetDict

def strip_whitespace(batch):
    return {
        'prompt': [s.strip() for s in batch['prompt']],
        'completion': [s.strip() for s in batch['completion']],
        'metadata': batch['metadata']
    }

cleaned = dds.map(strip_whitespace, batched=True)
cleaned.save_to_disk('artifacts/datasets/instruct_pc_cleaned')
```

### D3. Dynamic padding collator (if using a framework)

```python
# Example: minimal dynamic pad at batch time (tokenizers path)
from typing import List, Dict
import torch

def collate_fn(examples: List[Dict]):
    # Examples are already torch tensors after set_format(type='torch')
    # Extract input_ids and attention_mask
    input_ids_list = [e['input_ids'] for e in examples]
    attention_mask_list = [e['attention_mask'] for e in examples]
    
    # Find max length in this batch
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tok.token_to_id('<pad>')
    
    padded_ids = []
    padded_masks = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        # Convert tensor to list for manipulation
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        mask = mask.tolist() if torch.is_tensor(mask) else mask
        
        # Pad if needed
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
            mask = mask + [0] * (max_len - len(mask))
        
        padded_ids.append(ids)
        padded_masks.append(mask)
    
    return {
        'input_ids': torch.tensor(padded_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_masks, dtype=torch.long)
    }

train_loader_dp = DataLoader(batched_enc['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)
batch = next(iter(train_loader_dp))
batch['input_ids'].shape
```

---

## Wrap‑Up

Add a markdown cell and answer:

1. Why prefer `datasets` over plain pandas for large JSONL corpora?  
2. What seed did you use for deterministic splits & shuffles? Show how to reproduce the exact split on another machine.  
3. Compare fixed‑length padding vs dynamic padding in your pipeline. Which will you choose for your training run and why?

Confirm saved artifacts:

- `artifacts/datasets/instruct_pc_splits/` (Arrow)  
- `artifacts/datasets/instruct_pc_cleaned/` (optional)  
- `artifacts/tokenizer/bytebpe.json` (tiny local tokenizer)  
- `artifacts/samples/` (any debug exports you wrote)

---

- **Common pitfalls:** Mixing Trio and Prompt‑Completion columns; forgetting to fix a random seed; applying tokenization unbatched (slow); holding arrays in Python lists instead of letting `datasets` manage Arrow memory.

---

## Solution Snippets (reference)

**Load from multiple JSONL files at once:**

```python
from datasets import load_dataset
files = {
  'train': 'artifacts/jsonl/instruct_prompt_completion.jsonl',
  'validation': 'artifacts/jsonl/instruct_trio_from_rag.jsonl' # example
}
DatasetDict({k: load_dataset('json', data_files=v, split='train') for k,v in files.items()})
```

**Save to shards (Arrow):**

```python
dds['train'].save_to_disk('artifacts/datasets/pc_train')
```

**Reload later:**

```python
from datasets import load_from_disk
loaded = load_from_disk('artifacts/datasets/pc_train')
loaded
```
