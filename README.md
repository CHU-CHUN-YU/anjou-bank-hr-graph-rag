# 安久銀行 HR AI — Graph RAG Assistant

A proof-of-concept Traditional-Chinese HR question-answering assistant for a fictional
bank ("安久銀行"). It grounds answers in Taiwan's Labor Standards Act (勞動基準法) plus the
bank's internal staff regulations, using **hybrid retrieval + a knowledge graph + a
LangGraph workflow with deterministic guardrails**. The generative model is a local
HuggingFace LLM (no OpenAI API key required) intended for a Colab T4 GPU.

> **Design principle:** the LLM is a *signal provider*, never the final decision-maker.
> Routing (answer / disclaimer / clarify / escalate) is decided by deterministic
> guardrails; risk can only be raised by the LLM, never lowered.

## Repository layout

```
.
├── src/
│   └── hr_ai_graph_rag/            # the pipeline, split into focused modules
│       ├── config.py               #   env / models / global constants
│       ├── utils.py                #   text · category · route helpers (no deps)
│       ├── artifacts.py            #   offline LLM-assisted knowledge loader
│       ├── ingestion.py            #   DOCX parse · chunking · HRKnowledgeBuilder
│       ├── graph.py                #   HRKnowledgeGraph (生成圖)
│       ├── retrieval.py            #   HybridRetriever — hybrid + rerank (排序)
│       ├── llm.py                  #   LocalHFLLM + call_llm_* (本地生成模型)
│       ├── workflow.py             #   HRAssistantGraph — LangGraph (對話)
│       ├── evaluation.py           #   golden-set metrics
│       ├── runner.py               #   I/O + main() entry point
│       └── __main__.py             #   python -m hr_ai_graph_rag
├── data/
│   ├── policies/                   # internal bank policy DOCX (simulated)
│   ├── golden/                     # 50-question evaluation set
│   └── hr_offline_artifacts/       # curated offline knowledge (9 JSON files)
├── tests/
│   ├── test_pipeline_no_llm.py     # runs the non-LLM layers on the real data
│   └── test_retrieval_rerank.py    # embeddings + cross-encoder rerank (needs ML stack)
├── docs/
│   ├── WORKFLOW.md                     # detailed system & workflow walkthrough (中文)
│   └── README_RUNTIME_PATTERN_REWRITE.md   # original design notes (中文)
├── scripts/run_local.sh
├── requirements.txt                # full stack (embeddings + local LLM, GPU)
└── requirements-core.txt           # no-LLM subset (parsing/graph/eval)
```

## Pipeline

```
DOCX (law + internal policy) ─┐
Golden Dataset JSON ──────────┤   data/ is bundled & auto-discovered
Offline artifacts (9 JSON) ───┘
        │
        ▼
 Hierarchical chunking → document / article / semantic chunks
   (+ faq chunks only when USE_GOLDEN_AS_FAQ_CHUNKS=true)
        │
        ▼
 HRKnowledgeGraph   concept ↔ law/policy articles, typed relations
 HybridRetriever    FAISS dense + BM25 + metadata, then cross-encoder rerank
        │
        ▼
 LangGraph workflow — one pass per question (HRAssistantGraph):
   1. query_understanding    heuristic (+ optional local-LLM) → category / intent / risk / rewrite / slots
   2. retrieval_orchestrator hybrid retrieval + 1-hop knowledge-graph expansion
   3. guardrails (deterministic) → route ∈ { answer · disclaimer · clarify · escalate }
        ├─ answer / disclaimer → 4. generate_answer (local LLM, [S#] citations) → 5. faithfulness_check
        ├─ clarify  → return follow-up questions
        └─ escalate → hand off to HR
        │
        ▼
 Evaluation vs. golden set:
   category · route · retrieval-hit · source-type-hit · citation · faithfulness · latency
```

> Every stage prints a `[STAGE] <name> — <facts>` line (with a short content preview)
> so you can see what each step produced; silence it with `STAGE_LOG=false`.

## Quick start

### A. No-LLM verification (CPU, lightweight — recommended first run)

Runs DOCX parsing → chunking → artifact loading → knowledge-graph build → matchers →
golden-dataset loading on the bundled real data. No GPU or LLM needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
python tests/test_pipeline_no_llm.py
```

Expected: `ALL PASSED — 24 checks`.

### B. Full pipeline (GPU / Colab)

```bash
pip install -r requirements.txt
bash scripts/run_local.sh        # or: PYTHONPATH=src python -m hr_ai_graph_rag
```

To verify retrieval + the cross-encoder rerank stage in isolation (downloads models):

```bash
python tests/test_retrieval_rerank.py
# fast check with small stand-in models:
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2 \
python tests/test_retrieval_rerank.py
```

In **Google Colab**: choose a T4 GPU runtime, then `git clone` this repo and run
`PYTHONPATH=src python -m hr_ai_graph_rag` (bundled `data/` is auto-discovered; set the env
vars below to override inputs). The default LLM (Qwen2.5-1.5B-Instruct) is open and
**not gated**, so no HuggingFace token is required out of the box.

The bundled `data/` files are auto-discovered when running from the repo. A 勞動基準法
DOCX placed in `data/policies/` (filename containing 勞動基準法 / 勞基法 / labor) is
auto-discovered too; if none is present, a built-in sample of key articles is used so
the pipeline still runs end-to-end. Override with `LABOR_LAW_DOCX_PATH`.

### Surviving Colab's runtime limit (resumable evaluation)

The evaluation step runs **every** Golden-Dataset question through the LLM and **checkpoints
after each one** (`OUTPUT_DIR/eval_checkpoint.jsonl`, one record per line, fsync'd). If the
session is cut off (Colab's ~50-min free limit), at most the single in-flight question is
lost. On the next run, completed questions are loaded from the checkpoint and skipped, so
only the remaining ones are generated — and the **final detail/summary are always computed
over all questions** (resumed + new). Finished questions also skip the showcase demo so the
time budget goes to real evaluation.

To make the checkpoint (and all outputs) survive a disconnect on Colab, mount Google Drive
and point `OUTPUT_DIR` at it — `/content/` is wiped on disconnect, Drive is not:

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['OUTPUT_DIR'] = '/content/drive/MyDrive/hr_ai_graph_rag_outputs'
# then run the pipeline; re-run the SAME cell/notebook after a disconnect to resume:
!PYTHONPATH=src python -m hr_ai_graph_rag
```

Re-running with the same `OUTPUT_DIR` auto-resumes. To start fresh, delete
`eval_checkpoint.jsonl` (or set `EVAL_CHECKPOINT_PATH` elsewhere).

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LABOR_LAW_DOCX_PATH` | auto / sample | 勞動基準法 DOCX (auto-discovered in `data/policies/`, else built-in sample) |
| `INTERNAL_POLICY_DOCX_PATH` | bundled | Internal policy DOCX |
| `GOLDEN_DATASET_JSON_PATH` | bundled | Evaluation set |
| `OFFLINE_ARTIFACT_DIR` | bundled | Offline artifact folder |
| `OFFLINE_ARTIFACT_ZIP_PATH` | *(unset)* | Offline artifacts as a ZIP (alternative to the folder) |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Dense-retrieval embedding model |
| `USE_RERANKER` | `true` | Enable cross-encoder rerank stage |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RERANK_CANDIDATES` / `RERANK_WEIGHT` | `20` / `0.7` | Rerank pool size / score blend |
| `HF_LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Local generative model (response) |
| `HF_LLM_USE_4BIT` | `true` | 4-bit quantize the LLM (GPU only; auto-off on CPU) |
| `HF_MAX_NEW_TOKENS` / `HF_TEMPERATURE` | `768` / `0.1` | LLM generation length / sampling temperature |
| `USE_LLM` | `true` | Use the generative LLM (false = deterministic template answer) |
| `USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING` | `true` | Local-LLM query classification |
| `LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS` | `5` | Max optional LLM query-expansion terms |
| `USE_GOLDEN_AS_FAQ_CHUNKS` | `false` | Keep eval data out of the KB (leave false) |
| `LOAD_PENDING_GRAPH_EDGES` | `false` | Load only HR-approved graph edges |
| `STAGE_LOG` | `true` | Print a `[STAGE]` line per pipeline step (what each stage output) |
| `STAGE_LOG_PREVIEW` | `160` | Max chars of the per-stage content preview |
| `RUN_LOG` | `true` | Tee all console output (prints + `[STAGE]`) to a log file |
| `RUN_LOG_FILE` | `OUTPUT_DIR/run_log.txt` | Path of the saved run log |
| `OUTPUT_DIR` | `./hr_ai_graph_rag_outputs` (`/content/...` in Colab) | All outputs + eval checkpoint; point at Google Drive to survive a Colab disconnect |
| `EVAL_CHECKPOINT_PATH` | `OUTPUT_DIR/eval_checkpoint.jsonl` | Resumable per-question eval checkpoint (delete to start fresh) |
| `DEMO_QUESTIONS` | `9` | Showcase demo question count (`0` = skip; auto-skipped when resuming) |

> The default `Qwen/Qwen2.5-1.5B-Instruct` is **not gated** — no HuggingFace token or
> license acceptance is needed. If you switch `HF_LLM_MODEL_NAME` to a gated model (e.g.
> `google/gemma-2-2b-it`), accept its license on the model page and authenticate
> (`huggingface-cli login` or set `HF_TOKEN`) before the first run.
> `BAAI/bge-m3` (~2GB) and `bge-reranker-v2-m3` (~2GB) are best on a GPU.

## Notes

- `import hr_ai_graph_rag` is side-effect free (no heavy deps, no pipeline, no prints);
  the config summary is printed by the runner at the start of a run (so it is captured
  in the run log). The full pipeline runs via `PYTHONPATH=src python -m hr_ai_graph_rag`,
  `runner.main()`, or in Colab.
- The package re-exports every public symbol at the top level, so `import hr_ai_graph_rag
  as hr` keeps the flat `hr.HRAssistantGraph`, `hr.main`, … interface — submodules
  (`config`/`utils`/`graph`/`retrieval`/`workflow`/…) are an internal detail.
- Heavy deps (torch, faiss, sentence-transformers, transformers, langgraph) are imported
  lazily, so the data/parse/graph/eval layers work with just `requirements-core.txt`.
- A full run writes to `OUTPUT_DIR` (`./hr_ai_graph_rag_outputs`, or `/content/...` in
  Colab, gitignored): knowledge-graph `kg_nodes.csv` / `kg_edges.csv` /
  `hr_knowledge_graph.gexf`, chunks, evaluation detail + summary,
  `evaluation_records.json` and `evaluation_records/<id>.json` (every question's
  **full answer + full reference sources**, aggregate + one file per question),
  `feedback_log.csv`, the run log, and a bundled ZIP (auto-downloaded in Colab).
- See `docs/WORKFLOW.md` for a very detailed (Chinese) walkthrough of the whole
  pipeline and the per-question LangGraph workflow.
- See `docs/README_RUNTIME_PATTERN_REWRITE.md` for the original (Chinese) design notes.
