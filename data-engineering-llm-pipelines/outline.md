# Data Engineering for LLM Pipelines

## 5-Day Course Plan (≈7 hours per day)

**Daily cadence (suggested):**

- 45–60 min blocks; 10–15 min breaks AM/PM; 60 min lunch.
- Target **35–40% instructor-led demo** / **60–65% guided labs**

---

## Prereqs & Environment

- Python 3.13; VS Code or JupyterLab; venv.  
- Packages: `numpy`, `pandas`, `requests`, `pandera` (or `pydantic`), `ydata-profiling`, `pyarrow`, `datasets`, `tqdm`.  
- One SQL engine (SQLite bundled; Postgres optional).  
- Small sample datasets (orders/customers), plus 2–3 public APIs for practice.

---

## Day 1 — Foundations & Data Extraction (SQL + APIs)

**Theme:** Build reliable, secure data extractors for downstream LLM work.

### Day 1 - Learning objectives

- Compare Python lists vs. NumPy arrays and vectorization benefits.
- Use `requests` for API access (params, JSON parsing, backoff).
- Query SQL safely (parameterized queries) and land results into pandas.
- Run and document work in Jupyter (magics, markdown, EDA setup).

### Day 1 - Topics & demos

1. **Course kickoff & LLM pipeline map (30 min)**  
   - Where extraction, transformation, validation, profiling, and LLM-ready formats fit.

2. **Python + NumPy quick refresh (45 min)**  
   - Array creation, shape/reshape (`arr.reshape`), vectorized ops vs. loops; memory footprint.
   <!-- Assessment tie-ins: NumPy vs list/efficiency; reshape. -->

3. **Jupyter for EDA (Exploratory Data Analysis) & pipelines (40 min)**  
   - Markdown narratives, `%matplotlib inline`/other magics, reproducibility.
   <!-- Assessment tie-ins: Why notebooks for EDA; magic commands. -->

4. **HTTP APIs with `requests` (70 min)**  
   - GET with `params`, auth, handling status codes, `response.json()`, pagination, **rate-limit backoff/exponential retry**.  
   <!-- Assessment tie-ins: passing URL params; `.json()`; backoff strategy. -->

5. **SQL extraction to pandas (75 min)**  
   - `sqlite3` or Postgres, parameterized queries (`?` / `%s`), `pd.read_sql_query`, chunked reads.
   <!-- Assessment tie-ins: parameterized queries & SQL injection; `pd.read_sql_query`. -->

### Day 1 - Guided labs

- **Lab 1.1: API harvester**  
  Implement a client that queries a rate-limited REST API with `params`, retries, and pagination; persist raw JSON snapshots.
- **Lab 1.2: SQL extractor**  
  Use parameterized queries to pull tables and views into pandas; save to Parquet.

---

## Day 2 — Cleaning, Transforming, and Joining for LLM Use

**Theme:** Turn messy raw data into consistent, joined datasets.

### Day 2 - Learning objectives

- Master pandas selection, boolean logic, deduping, missing-data handling.
- Clean tricky text/numeric fields (currency, dates).
- Perform correct joins for downstream tasks.

### Day 2 - Topics & demos

1. **Pandas selection & filtering (50 min)**  
   - `df[(cond1) & (cond2)]`, `query`, chained masks, `loc`.  
   <!-- Assessment tie-ins: correct boolean filtering syntax. -->

2. **Missing data, duplicates, and type normalization (75 min)**  
   - `dropna`, `fillna`, `astype`, `to_datetime`, `drop_duplicates`.  
   <!-- Assessment tie-ins: `dropna`; dedupe by key; `Series.apply`. -->

3. **Cleaning real-world fields (60 min)**  
   - Currency strings → numerics (strip `$` and commas), locale issues; categorical normalization ("USA" vs "U.S.A." vs "United States").  
   <!-- Assessment tie-ins: chain string cleans → `astype(float)`; consistency vs validity. -->

4. **GroupBy and joins (70 min)**  
   - `groupby().agg({'col': 'mean'})`, `merge` patterns; choosing **inner vs left vs outer** joins.  
   <!-- Assessment tie-ins: groupby/mean; when to use inner join orders↔customers. -->

### Day 2 - Guided labs

- **Lab 2.1: Clean & standardize**  
  Fix prices, dates, and categorical country values; build `is_adult` with a vectorized boolean.
- **Lab 2.2: Join & aggregate**  
  Join orders↔customers (inner); compute per-segment summary with `groupby().agg`.

---

## Day 3 — Schema Validation & Data Quality (Pydantic/Pandera) + Profiling

**Theme:** Make pipelines trustworthy via contracts and profiles.

### Day 3 - Learning objectives

- Define schemas with **Pydantic** or **Pandera** (types, ranges, custom checks).
- Add validation stages to ETL/ELT runs.
- Profile datasets to surface anomalies early.

### Day 3 - Topics & demos

1. **Why schema validation in ML/LLM pipelines (40 min)**  
   - Catching upstream drift; protecting downstream training/eval.  
   <!-- Assessment tie-ins: purpose of validation; "unexpected changes in structure/types." -->

2. **Pandera/Pydantic walkthrough (90 min)**  
   - Column types, constraints (`Check.in_range`, regex), row/DF checks, error handling, CI hooks.  
   <!-- Assessment tie-ins: libraries for schemas & custom rules. -->

3. **Data profiling at scale (60 min)**  
   - Summary stats, cardinality, value distributions, outlier flags with **ydata-profiling**; integrating reports in review.  
   <!-- Assessment tie-ins: profiling tasks; tool purpose. -->

4. **Quality dimensions (50 min)**  
   - Completeness, validity, **consistency** (country naming), timeliness; thresholds & alerts.

### Day 3 - Guided labs

- **Lab 3.1: Author a schema**  
  Write a Pandera schema for your cleaned, joined table (types, ranges, enums).
- **Lab 3.2: Profiling report**  
  Generate a ydata-profiling HTML report; list top 5 data risks and mitigation steps.

---

## Day 4 — Preparing Data for LLMs: Formats, Tokenization & RAG-Ready Corpora

**Theme:** Make data usable for fine-tuning and retrieval-augmented generation.

### Day 4 - Learning objectives

- Produce **LLM-ready file formats** (JSONL) and instruction-tuning structures.
- Understand tokenization, batching, and dataset loading with `datasets`.
- Prepare documents for RAG: chunking, metadata, embeddings.

### Day 4 - Topics & demos

1. **From tables/text to **JSONL** (60 min)**  
   - Why JSON Lines for large corpora, streaming, memory efficiency; writing one record per line.  
   <!-- Assessment tie-ins: JSONL advantages. -->

2. **Instruction tuning schemas (70 min)**  
   - Common structures: `{instruction, input, output}` vs `{prompt, completion}`; templating ("### Instruction: …").  
   <!-- Assessment tie-ins: prompt-completion pairs; structured templates. -->

3. **Loading with `datasets` (60 min)**
   - Load JSONL; map/tokenize; shuffle & batch; train/test splits.  
   <!-- Assessment tie-ins: prepares for efficient tokenization/shuffling/batching. -->

4. **RAG data prep (75 min)**  
   - **Chunking** strategies (by sentences/headings/tokens), overlap, metadata (source, timestamp, section), basic cleaners (keep punctuation when meaningful).  
   <!-- Assessment tie-ins: keep punctuation for meaning; chunking for embeddings/retrieval. -->

5. **Prompting patterns (35 min)**  
   - **Few-shot** prompts; formatting consistent exemplars.  
   <!-- Assessment tie-ins: definition of few-shot; text-centric "feature engineering". -->

### Day 4 - Guided labs

- **Lab 4.1: Build an instruction-tuning set**  
  Transform a cleaned table + notes into `{instruction,input,output}` JSONL with a clear template.
- **Lab 4.2: RAG chunk pack**  
  Split a small document set into chunks with metadata; export to Parquet/JSONL ready for embedding.

---

## Day 5 — End-to-End Pipeline Orchestration, Checks, and Handoff

**Theme:** Make it production-friendly: modular code, tests, validation gates, and artifacts.

### Day 5 - Learning objectives

- Compose an **extract → clean → validate → profile → LLM-format** pipeline.
- Add data-quality gates and join correctness tests.
- Produce teach-back artifacts (reports, data dictionary, runbook).

### Day 5 - Topics & demos

1. **Pipeline assembly in notebooks/scripts (60 min)**  
   - Layering: extraction modules; transformation functions; validation step; writers for JSONL/Parquet; logging/metrics.

2. **Join correctness & invariants (60 min)**  
   - Unit-style checks (row counts before/after, null keys, duplicate keys), sample-based assertions.

3. **Performance & cost awareness (45 min)**  
   - Efficient I/O, chunking large files, memory tips; understanding token counts and dataset sizes for LLM training.

4. **Deliverables & review (60 min)**  
   - Data dictionaries, schema files, profiling exports, JSONL samples, RAG chunk manifest.

### Mini-Capstone lab (≈3 hours)

- **Build the full pipeline** from your chosen source (SQL + API) through:  
  cleaning → joins → Pandera validation → ydata-profiling snapshot → LLM JSONL (instruction-tuning) **and/or** RAG chunk pack.  
- **Exit criteria:**  
  - Passing schema checks; profiling report saved; correct join semantics; JSONL valid line-by-line; README with instructions to load via `datasets`.

---

## Deliverables produced by the end

- Reusable **API & SQL extractors** with backoff and parameterized queries.  
- A **cleaned & joined** dataset with documented transforms.  
- **Schema file(s)** (Pandera) and an automated **validation step**.  
- A **profiling report** with a short "data quality risks & mitigations" note.  
- **LLM-ready JSONL** (instruction tuning) and/or **RAG chunk pack** with metadata.  
- A concise **README/runbook** explaining how to load with `datasets`.
