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
│   └── hr_ai_graph_rag.py          # the full pipeline (single module)
├── notebooks/
│   └── 安久銀行_HR_AI_Graph_RAG_..._Colab.ipynb
├── data/
│   ├── policies/                   # internal bank policy DOCX (simulated)
│   ├── golden/                     # 50-question evaluation set
│   └── hr_offline_artifacts/       # curated offline knowledge (9 JSON files)
├── tests/
│   ├── test_pipeline_no_llm.py     # runs the non-LLM layers on the real data
│   └── test_retrieval_rerank.py    # embeddings + cross-encoder rerank (needs ML stack)
├── docs/
│   └── README_RUNTIME_PATTERN_REWRITE.md   # original design notes (中文)
├── scripts/run_local.sh
├── requirements.txt                # full stack (embeddings + local LLM, GPU)
└── requirements-core.txt           # no-LLM subset (parsing/graph/eval)
```

## Pipeline

```
DOCX (law + internal policy) ─┐
Golden Dataset JSON ──────────┤
Offline artifacts (9 JSON) ───┘
        │
        ▼
 Hierarchical chunking (document / article / semantic)
        │
        ▼
 LangGraph workflow:
   query understanding  → (heuristic + optional local-LLM classification)
   retrieval orchestrator → hybrid (FAISS vector + BM25 + metadata) + graph expansion
   deterministic guardrails → answer | disclaimer | clarify | escalate
   grounded answer generation (local LLM) → faithfulness check
        │
        ▼
 Evaluation vs. golden set (category / route / retrieval / citation / faithfulness / latency)
```

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
bash scripts/run_local.sh        # or: python src/hr_ai_graph_rag.py
```

To verify retrieval + the cross-encoder rerank stage in isolation (downloads models):

```bash
python tests/test_retrieval_rerank.py
# fast check with small stand-in models:
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2 \
python tests/test_retrieval_rerank.py
```

In **Google Colab**: open the notebook under `notebooks/`, choose a T4 GPU runtime,
and upload the three input files when prompted (or set the env vars below).

The bundled `data/` files are auto-discovered when running from the repo. The official
勞動基準法 DOCX is **not** bundled (it is an external legal source); if you don't supply
one, a built-in sample of key articles is used so the pipeline still runs end-to-end.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LABOR_LAW_DOCX_PATH` | *(sample)* | Official 勞動基準法 DOCX |
| `INTERNAL_POLICY_DOCX_PATH` | bundled | Internal policy DOCX |
| `GOLDEN_DATASET_JSON_PATH` | bundled | Evaluation set |
| `OFFLINE_ARTIFACT_DIR` | bundled | Offline artifact folder |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Dense-retrieval embedding model |
| `USE_RERANKER` | `true` | Enable cross-encoder rerank stage |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `RERANK_CANDIDATES` / `RERANK_WEIGHT` | `20` / `0.7` | Rerank pool size / score blend |
| `HF_LLM_MODEL_NAME` | `google/gemma-2-2b-it` | Local generative model (response) |
| `USE_LLM` | `true` | Use the generative LLM (false = template answer) |
| `USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING` | `true` | Local-LLM query classification |
| `USE_GOLDEN_AS_FAQ_CHUNKS` | `false` | Keep eval data out of the KB (leave false) |
| `LOAD_PENDING_GRAPH_EDGES` | `false` | Load only HR-approved graph edges |

> **Gemma is a gated model on the HuggingFace Hub.** Accept its license on the model
> page and authenticate (`huggingface-cli login` or set `HF_TOKEN`) before the first run.
> `BAAI/bge-m3` (~2GB) and `bge-reranker-v2-m3` (~2GB) are best on a GPU.

## Notes

- `import hr_ai_graph_rag` is side-effect free; the full pipeline runs only via
  `python src/hr_ai_graph_rag.py`, `%run`, or in Colab.
- Heavy deps (torch, faiss, sentence-transformers, transformers, langgraph) are imported
  lazily, so the data/parse/graph/eval layers work with just `requirements-core.txt`.
- See `docs/README_RUNTIME_PATTERN_REWRITE.md` for the original (Chinese) design notes.
