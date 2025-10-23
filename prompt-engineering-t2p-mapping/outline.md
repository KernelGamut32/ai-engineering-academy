# Prompt Engineering & Task-to-Prompt Mapping

## 5-Day Course Plan (≈7 hours per day)

**Daily cadence (suggested):**

- 45–60 min blocks; 10–15 min breaks AM/PM; 60 min lunch.
- Target **30–35% instructor-led demo** / **65–70% guided labs**

---

## Prereqs & Environment

- Python 3.13; VS Code or JupyterLab; venv.
- Packages (suggested): `langchain`, `langchain-community`, an LLM provider SDK (e.g., `openai`/`anthropic`), `tiktoken` or tokenizer equivalent, `jsonschema`, `pydantic`, `pandas`.  
- Accounts/keys: **LangSmith** and/or **PromptLayer** for experiment tracking.  
- Small sample corpora (support tickets, policy docs, product specs) for classification, extraction, and summarization tasks.

---

## Day 1 — Foundations & Task Framing

**Theme:** Build reliable prompts by mapping real tasks to precise frames and output schemas.

### Day 1 - Learning objectives

- Distinguish **zero-shot**, one-shot, and **few-shot** prompting and when to use each.  
- Frame tasks as **classification**, **extraction**, **summarization**, or **generation** with explicit constraints.  
- Specify output formats (JSON/Markdown tables) and write validator prompts.

### Day 1 - Topics & demos

1. **Prompt anatomy & taxonomy (45–60 min)**  
   - Instruction, context, constraints, output contract; anti-patterns (vagueness, leakage).
   <!-- Assessment tie-ins: identify zero/one/few-shot; map task-to-frame. -->

2. **Task-to-prompt mapping (60–70 min)**  
   - **Classification**: label policy, tie-break rules, abstain policy.  
   - **Extraction**: schema-first design; strict JSON with `jsonschema`.  
   - **Summarization**: audience, scope, style/length controls.
   <!-- Assessment tie-ins: extraction framing and schema compliance. -->

3. **Reliability boosters (45–60 min)**  
   - Delimiters, refusal guidance, "answer then verify", rubric-based scoring prompts.  
   - Determinism levers (temperature, top_p) vs creativity.
   <!-- Assessment tie-ins: verification prompts & constraints. -->

### Day 1 - Guided labs

- **Lab 1.1: Frame it three ways**  
  Convert one messy request into **classification**, **extraction (JSON schema)**, and **summarization** prompts; add validator prompts.  
- **Lab 1.2: Schema guardrail**  
  Use `jsonschema` to validate model outputs; on failure, auto-generate a "repair" prompt and re-try.

---

## Day 2 — Instruction-Based, Few-Shot, & Chain-of-Thought (CoT)

**Theme:** Strengthen instruction-following with examples and reasoning traces.

### Day 2 - Learning objectives

- Write crisp, testable **instruction-based** prompts with acceptance criteria.  
- Construct effective **few-shot** exemplars (typical, edge, adversarial).  
- Apply **CoT** to improve reasoning; manage verbosity and tokens.

### Day 2 - Topics & demos

1. **Instruction-based prompting deep dive (50–60 min)**  
   - Role prompting, constraints, step lists, acceptance tests; style/tone controls.
   <!-- Assessment tie-ins: instruction quality and acceptance criteria. -->

2. **Few-shot prompting (70–80 min)**  
   - Curating examples; ordering strategies; counter-examples; what makes few-shot distinct from zero-shot.
   <!-- Assessment tie-ins: "few-shot includes several examples before the query." -->

3. **Chain-of-Thought (CoT) (60–70 min)**  
   - "Let’s think step by step." Self-consistency (sample-and-select). Private vs exposed rationales.
   <!-- Assessment tie-ins: CoT trigger phrase and benefits. -->

### Day 2 - Guided labs

- **Lab 2.1: Zero vs few-shot A/B**  
  Build zero-shot and few-shot classifiers on a small labeled set; compute simple precision/recall.  
- **Lab 2.2: CoT uplift**  
  Add CoT to a calculation/policy reasoning task; compare accuracy and token cost; finalize a "reason→answer" template (reasoning hidden, final JSON answer concise).

---

## Day 3 — Prompt Chaining & **LangChain** Basics (with Memory)

**Theme:** Decompose complex goals into multi-step pipelines; use memory where it truly helps.

### Day 3 - Learning objectives

- Break complex tasks into **chains** (plan → extract → normalize → decide → write).  
- Implement chains with **LangChain** (PromptTemplate, Runnables, parsers).  
- Add **memory** (buffer/summary/entity) and understand its trade-offs.

### Day 3 - Topics & demos

1. **Why chain? (45–60 min)**  
   - Simpler steps, clearer evaluation; reduce prompt sprawl; enforce structure with intermediate schemas.
   <!-- Assessment tie-ins: primary purpose of chaining. -->

2. **LangChain fundamentals (70–80 min)**  
   - PromptTemplate, ChatModels, `RunnableSequence`, output parsers, retries/timeouts; tool-calling overview (if applicable).
   <!-- Assessment tie-ins: chaining mechanics. -->

3. **Memory patterns (60–70 min)**  
   - Conversation buffer vs summary memory vs entity memory; privacy/latency considerations; when **not** to use memory.
   <!-- Assessment tie-ins: definition and function of memory. -->

### Day 3 - Guided labs

- **Lab 3.1: Three-step chain**  
  (1) Extract structured data → (2) validate/repair → (3) summarize for executives (Markdown).  
- **Lab 3.2: Conversational chain with memory**  
  Add buffer or summary memory; demonstrate how answers depend on prior turns; add a "memory reset" control.

---

## Day 4 — Experimentation, Logging, and Monitoring with **PromptLayer** & **LangSmith**

**Theme:** Make prompt work observable, comparable, and reproducible.

### Day 4 - Learning objectives

- Log runs and traces with **PromptLayer**/**LangSmith**; attach metadata (prompt version, dataset hash).  
- Run A/B experiments: **zero vs few-shot**, **with vs without CoT**.  
- Create small offline eval sets and regression gates.

### Day 4 - Topics & demos

1. **Observability & experiment design (50–60 min)**  
   - Why logging matters; defining task metrics (schema validity, accuracy, latency, cost).
   <!-- Assessment tie-ins: purpose of PromptLayer/LangSmith. -->

2. **PromptLayer integration (60–70 min)**  
   - Setup, logging calls, filtering by tags/versions; export comparisons.
   <!-- Assessment tie-ins: instrumentation steps. -->

3. **LangSmith projects & datasets (60–70 min)**  
   - Traces, dataset runs, feedback functions, side-by-side diffs; CI-like checks on merges.
   <!-- Assessment tie-ins: experiment workflows. -->

### Day 4 - Guided labs

- **Lab 4.1: Instrument a chain**  
  Add tracing and metadata; produce a run list filtered by prompt version; chart latency and token usage.  
- **Lab 4.2: A/B experiments & evals**  
  Build two prompt variants (zero vs **few-shot**, ± **CoT**); run against a labeled mini-set; record wins/losses and schema validity; produce a "prompt release note."

---

## Day 5 — Capstone: Design, Build, and Evaluate a Production-Ready Chain

**Theme:** Ship something real: a robust chained workflow with memory, validation, and tracked experiments.

### Day 5 - Learning objectives

- Implement an end-to-end chain with memory and verification.  
- Compare prompt versions; present trade-offs and results.

### Day 5 - Topics & demos

1. **Capstone kickoff & scenarios (45–60 min)**  
   - Teams pick one:  
     - **Support Triage**: extract → classify → summarize.  
     - **Policy Q&A**: retrieval-lite context + memory + attribution-style summary.  
     - **Data Cleaning Assistant**: normalize fields, explain decisions, produce final JSON & executive abstract.
     <!-- Assessment tie-ins: task framing + chaining + memory. -->

2. **Build & instrument (120–140 min)**  
   - Implement with LangChain; add memory (buffer/summary/entity); enforce `jsonschema`; retries/repairs; tag versions; log with PromptLayer/LangSmith.

3. **Experiments & reporting (60–75 min)**  
   - Run A/Bs (zero vs **few-shot**; ± **CoT**); compute accuracy/schema validity/latency; write a one-pager with results and a go/no-go.

### Mini-Capstone lab (≈3 hours)

- **Deliver** an end-to-end chain from prompt design → chain construction → memory → validation → instrumentation → A/B results.  
- **Exit criteria:**  
  - Passing schema checks; comparison table with accuracy & schema validity; latency & token summaries; clearly versioned prompts; short README on how to reproduce runs.

---

## Deliverables produced by the end

- **Prompt templates** for classification, extraction (with JSON schema), summarization, and generation.  
- **Few-shot exemplar sets** and a **CoT** reasoning template (with hidden rationale, concise final answer).  
- **LangChain chains** with validation/repair and optional **memory**.  
- **PromptLayer/LangSmith** logs, traces, and an A/B experiment report (accuracy, schema validity, latency, token/cost).  
- A capstone **README/runbook** and "prompt release notes" documenting versions and results.
