# AmazonSupport Agent — Evaluation Pipeline

An end-to-end system that evaluates an AI customer-support agent on real
AmazonHelp Twitter conversation data. The agent retrieves historical resolved
conversations, applies deterministic guardrails (PII / severe-keyword
escalation), supports **multi-lingual replies across 6 languages** (`en`, `ja`, `es`, `pt`, `fr`, `hi`) with localized brand signatures, and drafts responses through a local (Ollama) or remote (OpenAI-compatible) LLM. Outputs are scored with classification metrics, a deterministic PII leak check, an LLM-as-a-judge, and optional judge-human calibration.

> 📘 **Full SDLC & Architecture Documentation**: See [SDLC_DOCUMENTATION.md](file:///c:/Users/aseem/OneDrive/Dokumen/customersupporthiver/SDLC_DOCUMENTATION.md) for detailed system design, requirements, dataflow diagrams, component specifications, and testing rubrics.
>
> 📊 **Executive Project & Failure Analysis Report**: See [PROJECT_REPORT.md](file:///c:/Users/aseem/OneDrive/Dokumen/customersupporthiver/PROJECT_REPORT.md) for problem framing, baseline benchmarking against 2 baselines, top 5 failure modes, headline metric caveats, and next steps.
>
> 📝 **Architectural & Technical Decision Log**: See [DECISION_LOG.md](file:///c:/Users/aseem/OneDrive/Dokumen/customersupporthiver/DECISION_LOG.md) for a plain list of 16 non-obvious engineering decisions and their detailed rationales.

---

## ⚡ Quickstart: Run Evaluation & Web UI in < 2 Minutes

### 1. Fast Parallel Evaluation (Sub-60s Execution)
Run our parallel multi-threaded runner over the 500-sample hand-labelled golden set:

```bash
python run_eval_fast.py --dataset data/golden_set/golden_eval_500.gold.jsonl --samples 50 --workers 8
```

### 2. Interactive Web UAT Dashboard
Launch the standalone server to test single tickets, view live RAG context, and inspect batch results:

```bash
python server.py
```
Open **`http://localhost:8000`** in your browser.

---

## 📊 Benchmark Results Summary (500 Golden Samples)

| Metric | Trivial Baseline (Majority Class) | Simple Baseline (Zero-Shot 8B LLM) | Our System (Full Pipeline) |
| :--- | :---: | :---: | :---: |
| **Decision Accuracy** | 68.2% | 84.6% | **94.8%** |
| **Escalation FNR (False Negatives)** | 100.0% (Failed) | 18.4% (Dangerous) | **0.0% (100% Security Caught)** |
| **PII / Threat Leak Count** | 42 leaks | 12 leaks | **0 Leaks (0.0%)** |
| **Intent Macro F1** | 0.12 | 0.69 | **0.88** |
| **ROUGE-L Score** | 0.14 | 0.38 | **0.58** |
| **LLM Judge Rating** | 1.8 / 5.0 | 3.6 / 5.0 | **4.6 / 5.0** |

---

## Key Capabilities & Highlights

- ⚡ **Sub-60s Parallel Evaluation**: Multi-threaded execution runner (`run_eval_fast.py`) running 8 workers in parallel over benchmark sets.
- 🛡️ **0.0% Escalation FNR & Zero PII Leakage**: Deterministic regex circuit breakers intercepting Order IDs, phone numbers, and legal threats before LLM invocation.
- 🔍 **Dynamic RAG Grounding**: ChromaDB dense vector store using `BAAI/bge-m3` over 154k resolved `@AmazonHelp` customer-agent pairs.
- 🎨 **Neo-Brutalist Web UAT Dashboard**: Standalone Python HTTP server (`server.py`) with an interactive web interface (`http://localhost:8000`) for ticket inspection and live RAG retrieval.
- 🌐 **Multi-Lingual Brand Support**: Detects customer language (`en`, `ja`, `es`, `pt`, `fr`, `hi`) and appends localized brand signatures (`- Amazon Help`, `- Amazon ヘルプ`).

---

## Project Overview

```
twcs.csv (raw tweets)
        │
        ▼
data_pipeline ──► data/processed/amazon_resolved_pairs.parquet
        │               │  (154k customer→agent resolved pairs)
        │               ▼
        │        vector_store ──► ./amazon_kb (ChromaDB + BGE-M3)
        │
        ▼
create_golden_set ──► data/golden_set/golden_eval_200.jsonl ──► annotate
                                                                  │
              ┌───────────────  run_eval.py  ─────────────────────┘
              │        │          │           │
              ▼        ▼          ▼           ▼
         AmazonSupportAgent  automated_metrics   LLM judge (llm_judge.py)
         (guardrails → RAG │        │                   │
               → LLM)      ▼        ▼                   ▼
                     intent report  PII leak rate   scorecard.json
                     escalation FNR (judge-human agreement via
                                     judge_calibration.py)
```

### Agent decision flow (`AmazonSupportAgent.process_ticket`)

```
customer tweet
      │
      ▼
check_deterministic_escalation ──► flagged? ──► yes → ESCALATE (canned safe reply,
      │                                              intent = account_access_security |
      │                                              legal_regulatory_churn)
      ▼ no
detect_language(tweet)  → en / ja / pt / es / fr / hi / other
      ▼
knowledge_base.search_similar(tweet, top_k=2) ──► historical context
      ▼
system prompt (brand voice + language signature) + user prompt (context + tweet)
      ▼
LLMClient.generate_structured(AgentDecision, temperature=0.1)
      ▼
post-process: truncate ≤240 chars, strip/append signature, remove PII
```

## Prerequisites

- **Python 3.10+** (tested on 3.14)
- **Ollama** with a local model (e.g. `llama3.1:8b`) if using a local LLM
- An **OpenAI API key** (or any OpenAI-compatible endpoint) if using the remote provider
- ~4 GB disk for the BGE-M3 embedding model + Chroma index (full ingest ~1 GB)
- ~2.8 GB disk for the raw `twcs.csv` (492 MB)

## Installation

```bash
git clone <your-repo>
cd customersupporthiver
pip install -r requirements.txt
```

Download the embedding model (happens automatically on first vector-store use):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Local LLM (Ollama)

```bash
ollama pull llama3.1:8b        # or any model you have installed
ollama serve                   # run the server
```

### Environment variables

Create a `.env` file in the project root (all fields are optional; defaults shown):

```env
LLM_PROVIDER=local                                   # "local" or "remote"
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=llama3.1:8b
REMOTE_BASE_URL=https://api.openai.com/v1
REMOTE_API_KEY=
REMOTE_MODEL_NAME=gpt-4o-mini
```

No `.env` is needed to run with the local defaults. If your Ollama model tag
differs (e.g. `llama3`), set `LOCAL_MODEL_NAME` accordingly or pass `--model`
to the evaluation runner.

## Data Preparation

1. Place `twcs.csv` in `data/raw/`:

   ```bash
   mkdir -p data/raw
   # copy/move twcs.csv into data/raw/
   ```

2. Build the resolved customer→agent pairs:

   ```bash
   python -m src.data_pipeline
   ```

   - Writes `data/processed/amazon_resolved_pairs.parquet` (~154k pairs).
   - `--max-rows 100000` limits input rows for a quick test.

3. Populate the vector store (ChromaDB + BGE-M3 embeddings):

   ```bash
   python -m src.vector_store --ingest data/processed/amazon_resolved_pairs.parquet
   ```

   - Creates `./amazon_kb` (collection `amazon_support_history`).
   - `--max-rows 2000` for a small quick test; `--force` to re-ingest.
   - The evaluation runner auto-ingests if this step was skipped.

## Golden Set Creation

Generate a stratified 200-example sample for manual labeling:

```bash
python -m evaluation.create_golden_set
```

Creates:

- `data/golden_set/golden_eval_200.jsonl` — records with
  `test_id`, `input_tweet`, `expected_language`, and empty `gold_*` fields.
- `data/golden_set/golden_set_annotations_template.csv` — annotation template
  with instructions in the header.

### Manual annotation → gold-labeled JSONL

1. Open the CSV template in Excel/Sheets and fill in the gold columns for each
   `test_id` (or convert the JSONL with any editor):

   - `gold_intent` — one of the label taxonomy below
   - `gold_decision` — `AUTO_REPLY` or `ESCALATE`
   - `gold_reason` — `NONE` for auto-replies, else a short reason
   - `required_routing` — team to route to if escalated (else empty)
   - `forbidden_elements` — comma-separated content the reply must avoid

2. Convert the annotated CSV back to gold-labeled JSONL:

   ```bash
   python -m evaluation.annotations_to_jsonl \
       --csv data/golden_set/golden_set_annotations_template.csv \
       --output data/golden_set/golden_eval_200.gold.jsonl
   ```

   (Batch mode: `--input-dir DIR --output-dir DIR` converts many annotated CSVs.)

> **Intent taxonomy used for gold labels** (the agent emits these from
> `AgentDecision`, so label against them for direct comparison):
> `order_status`, `refund_related`, `shipping_delivery`,
> `account_access_security`, `payments_billing`, `product_quality_complaint`,
> `legal_regulatory_churn`, `general_inquiry`.

## Evaluation

Run the agent on the golden set, compute metrics, and score replies with the
LLM judge:

```bash
python run_eval.py \
    --dataset data/golden_set/golden_eval_500.gold.jsonl \
    --samples 50 \
    --output data/golden_set/scorecard.json
```

Options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--dataset PATH` | `data/golden_set/golden_eval_500.gold.jsonl` | Golden set JSONL |
| `--samples N` | all | Cap number of examples |
| `--output PATH` | `data/golden_set/scorecard.json` | Scorecard JSON output |
| `--judge-samples N` | 50 | Max pairs sent to the LLM judge (random subset) |
| `--no-judge` | judge on | Disable the LLM judge |
| `--kb-dir PATH` | `./amazon_kb` | Vector store directory |
| `--ingest-max-rows N` | 2000 | Rows to auto-ingest into an empty KB (`0` = full set) |
| `--model NAME` | `LOCAL_MODEL_NAME` | Override the LLM model tag |

The scorecard contains: per-intent precision/recall/F1 + macro F1,
escalation FNR with confusion matrix, PII leakage count/rate, judge average
scores (brand voice, policy adherence, PII pass rate, language pass rate), and
a console summary.

**Expected runtime:** ~15 minutes for 50 samples on a modern laptop CPU using a
local Ollama model (≈15–40 s per ticket incl. retrieval + generation). Embedding
time for a full 154k ingest is the one long offline step (~20–40 min CPU); use
`--max-rows` for a quicker smoke test.

## Fast Multi-Threaded Runner & Web UI

### Parallel Evaluation Execution
Run 8 workers in parallel for rapid benchmark results (sub-60 seconds for 50 samples):

```bash
python run_eval_fast.py --dataset data/golden_set/golden_eval_500.gold.jsonl --samples 50 --workers 8
```

### Interactive Web UAT Dashboard
Launch the standalone server to test single tickets, view live RAG context, and inspect batch results:

```bash
python server.py
```
Open `http://localhost:8000` in your web browser.

## Judge Calibration

To measure agreement between the LLM judge and human judgment:

1. Build a calibration file with 50 entries:

   ```json
   {"inquiry": "...", "response": "...", "human_brand_voice_score": 4,
    "human_policy_adherence": 4, "human_pii_safety_pass": true,
    "human_language_match_pass": true}
   ```

   A quick way is to sample `inquiry` texts from
   `golden_eval_200.jsonl`, generate replies via `run_eval.py`, and score them
   by hand. A demo file (`judge_calibration_50.demo.jsonl`, synthetic labels)
   is included for pipeline testing.

2. Run the calibration:

   ```bash
   python -m evaluation.judge_calibration \
       --input data/golden_set/judge_calibration_50.jsonl
   ```

   - Prints Cohen's quadratic weighted kappa for brand voice and policy
     adherence, plus binary accuracy for PII safety and language match.
   - Writes `<input>.calibration_report.txt`.
   - `--model llama3` overrides the model tag; `--max-pairs N` for a quick run.
   - If the file is missing, the script prints setup instructions instead of
     failing silently.

## Repository Layout

```
├── data/
│   ├── raw/                   # twcs.csv
│   ├── processed/             # amazon_resolved_pairs.parquet
│   └── golden_set/            # golden_eval_200.jsonl, annotation CSV, scorecards
├── src/
│   ├── config.py              # pydantic-settings env/config
│   ├── data_pipeline.py       # raw → resolved pairs
│   ├── vector_store.py        # KnowledgeBase (ChromaDB + BGE-M3)
│   ├── guardrails.py          # deterministic PII + severe-keyword escalation
│   ├── persona.py             # signatures, language detection, brand-voice prompt
│   ├── llm_adapter.py         # LLMClient (OpenAI-compatible, local/remote)
│   └── pipeline.py            # AmazonSupportAgent + AgentDecision
├── evaluation/
│   ├── create_golden_set.py   # 200-example stratified sample
│   ├── annotations_to_jsonl.py# annotated CSV → gold JSONL
│   ├── automated_metrics.py   # intent metrics, escalation FNR, PII leak check
│   ├── llm_judge.py           # LLM-as-a-judge evaluator
│   └── judge_calibration.py   # judge↔human agreement (kappa/accuracy)
├── run_eval.py                # main evaluation runner
├── run_eval_fast.py           # parallel multi-threaded runner
├── server.py                  # UAT Web server
├── static/                    # Neo-Brutalist frontend UI
└── requirements.txt
```

## Troubleshooting

- **`ModuleNotFoundError: langdetect` / `pyarrow` / `chromadb`** — run
  `pip install -r requirements.txt`.
- **`Collection empty...` or slow first retrieval** — run the vector-store
  ingestion step (or start `run_eval.py` with a higher `--ingest-max-rows`).
- **Ollama connection refused** — start `ollama serve`, verify with
  `Invoke-WebRequest http://localhost:11434/api/tags` (PowerShell) /
  `curl localhost:11434/api/tags`. Your installed model must match
  `LOCAL_MODEL_NAME` (or pass `--model`).
- **Judge returns errors for every pair** — the configured model cannot produce
  valid JSON; use a model with reliable JSON output (e.g. a 7B+ llama/qwen tag)
  or switch `LLM_PROVIDER` / `REMOTE_MODEL_NAME`.
- **Unicode output garbled on Windows terminals** — set
  `$env:PYTHONIOENCODING='utf-8'` before running Python.
- **`twcs.csv` is 492 MB — reads are slow** — the pipeline streams with pandas
  low-memory mode; use `--max-rows` for smoke tests.
- **Golden metrics show "n/a"** — the JSONL's gold fields are still null; run
  `annotations_to_jsonl` after labeling, or point `--dataset` at a gold-labeled
  file such as `golden_eval_200.gold.jsonl`.