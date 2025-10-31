# Mini‑Capstone — From CSV to Instruction Datasets & RAG Chunks (180–210 min)

**Scenario:** You are building an “Ask the Assistant” model. You’ve been given a CSV file with 10,000 Q&A pairs (one row per record). Your goal is to **ingest, clean, validate, and transform** this corpus into artifacts suitable for **retrieval‑augmented generation (RAG)** and **instruction tuning (SFT)**, following the same practices you learned earlier.

**You are provided:** `ask_assistant_corpus.csv` (≈10k rows). Do **not** generate the CSV yourself.

> This capstone mirrors the prior labs on CSV→JSONL conversion, instruction‑tuning schemas, loading with 🤗 Datasets, and chunking/overlap strategies for RAG. Use the same guardrails and hygiene steps you practiced earlier. (See Labs 13–17.)

---

## Learning Outcomes

By the end of the capstone, you will be able to:

1. **Ingest & sanity‑check** a large Q&A CSV; profile counts, missing values, and duplicates.
2. **Apply governance & hygiene**: normalize whitespace, enforce language/length constraints, and deduplicate by content signature.
3. **Produce RAG chunks** with metadata and overlap that preserves context.
4. **Create instruction‑tuning JSONL** in two views: **Trio** (`instruction,input,output`) and **Prompt‑Completion** (templated `prompt,completion`).
5. **Load with 🤗 Datasets**, create deterministic splits, and (optionally) train a **tiny local tokenizer** for offline tokenization.
6. **Assemble few‑shot prompt packs** from your instruction dataset using TF‑IDF similarity, with a consistent template and token budget.

---

## Provided Files

- `ask_assistant_corpus.csv` — 10,000-row Q&A corpus (columns: `id,category,topic,difficulty,question,answer,source,created_at`).

---

## Timebox & Flow (approx.)

- **Part A (30–40 min):** Load, profile, and clean the CSV.
- **Part B (40–50 min):** Build **RAG chunk** JSONL with provenance + overlap.
- **Part C (40–50 min):** Build **instruction‑tuning** JSONL (Trio + Prompt‑Completion) with validation and length governance.
- **Part D (40–50 min):** Load with **🤗 Datasets**; create reproducible splits; (optional) offline tokenization.
- **Part E (20–30 min):** Build a **few‑shot exemplar bank** and assemble prompt packs (header or chat template).

---

## Part A — Ingest & Hygiene (CSV)

1. Read the CSV with a robust dtype strategy (avoid silent NaNs).
2. Compute basic profile: row count, per‑column non‑null counts, category/topic distribution, and duplicate detection on `(question, answer)` after normalization.
3. Implement **normalization**:
   - Collapse internal whitespace; **preserve meaning‑bearing punctuation** (e.g., decimals, hyphens, question marks).
   - Strip leading/trailing spaces; keep ASCII plus UTF‑8 punctuation.
4. **Governance gates** (example): require `answer` length ≥ 20 chars; drop rows with empty/short `question` or `answer`.
5. **Deduplicate** by content signature (normalized question + answer). Keep the most recent `created_at` when duplicates exist.
6. Save a small reviewer sample (`artifacts/samples/capstone_csv_sample.jsonl`) showing 5 normalized Q&A records with minimal metadata.

---

## Part B — RAG Chunks with Provenance

1. Write a chunker with overlap (choose tokens or characters) that preserves punctuation and returns chunks ≈ 300–500 tokens (or ≤ 900–1200 chars) with ~10–20% overlap.
2. For each **Q&A pair**, treat the **answer** as the retrievable text. Emit **RAG JSONL** lines:

   ```json
   {"doc_id": "...", "chunk_id": "...", "text": "...",
     "metadata": { "category": "...", "topic": "...", "difficulty": "...",
                    "source": "...", "schema_version": "rag-chunk-v1" } }
   ```

3. Validate one‑record‑per‑line, required keys present, and min text length ≥ 40 chars.
4. Shard JSONL to keep each file under ~50 MB. Write 3 sample lines to `artifacts/samples/`.

---

## Part C — Instruction‑Tuning JSONL (Two Views)

1. **Trio view:** Transform each row into:
   - `instruction` ← the **question**
   - `input` ← empty string
   - `output` ← the **answer**
   - Add stable `metadata` (id, category, topic, difficulty, created_at, schema_version).
2. **Prompt‑Completion view:** Apply a header template, e.g.:

   ```text
   ### Instruction:
   {question}

   ### Response:
   ```

   - `prompt` ← formatted header with the question
   - `completion` ← the answer
3. Implement validators (length thresholds, forbidden phrases if any, and optional eval‑set decontamination). Write counts and summary stats to `artifacts/stats/`.

---

## Part D — 🤗 Datasets + (Optional) Offline Tokenization

1. Use `datasets.load_dataset('json', ...)` to load your **Prompt‑Completion** JSONL.
2. Create deterministic **train/validation/test** splits with a fixed seed; shuffle splits reproducibly.
3. Export splits to disk (Arrow).  
4. *(Optional)* Train a tiny **Byte‑Level BPE** tokenizer on prompts for offline tokenization; map to `input_ids`/`attention_mask` via a batched `map` function and dynamic padding if using a framework.

---

## Part E — Few‑Shot Exemplar Bank & Prompt Packs

1. Build an **exemplar bank** from your Trio data (or Prompt‑Completion), filtering out very short answers.
2. Fit a **TF‑IDF** vectorizer on exemplar instructions (+ optional inputs). Implement a selector to pick top‑K exemplars for a new task.
3. Choose **header or chat** template and assemble a final **few‑shot prompt** under a token budget. Persist a **prompt pack** JSONL with metadata (`k`, approx tokens, exemplar ids).

---

## Deliverables (commit to `artifacts/`)

- `artifacts/jsonl/rag_chunks_from_csv.jsonl` (sharded if needed)  
- `artifacts/jsonl/instruct_trio.jsonl`  
- `artifacts/jsonl/instruct_prompt_completion.jsonl` (+ optional cleansed variant)  
- `artifacts/datasets/...` (Arrow splits)  
- `artifacts/prompts/exemplar_bank.jsonl`, `artifacts/prompts/fewshot_prompt_pack.jsonl`  
- `artifacts/stats/...` (validation + length stats)  
- `artifacts/samples/...` (reviewer samples)

---

## Checkpoints (show in notebook)

- Row counts before/after cleaning and dedupe; sample normalized lines.
- Validator results for RAG JSONL and instruction JSONL.
- Split sizes and reproducibility seed.
- Prompt pack examples with computed approximate token counts.

> **Notes:** Keep templates consistent across the dataset. Avoid mixing trio and prompt‑completion formats within one training run. Respect token budgets and store provenance metadata for auditability.
