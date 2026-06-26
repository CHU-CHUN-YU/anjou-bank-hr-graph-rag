# ============================================================
# config — 環境設定、依賴啟動、全域常數
#
# 集中所有環境變數、模型名稱與路徑設定;heavy/LLM 依賴採延後載入。
# 本模組為最底層,不依賴 package 內其他模組。import 時會印出設定摘要。
# ============================================================

# ============================================================
# Colab-ready HR AI Assistant: external DOCX/JSON ingestion + Offline Artifacts + Runtime Local LLM classifier + RAG + Graph-enhanced RAG + LangGraph
# 本版為「成熟 workflow 調整版」：
# - 勞動基準法由使用者上傳 DOCX
# - 模擬銀行員工內規由使用者上傳 DOCX
# - Golden Dataset 由使用者上傳 JSON
# - Offline LLM-assisted artifacts 由 JSON/ZIP 接入：concept_nodes, risk_policy, query_patterns, rewrite_rules, relation_schema, graph_relation_candidates
# - Runtime local LLM 負責 category / intent / concept / ambiguity / candidate risk structured classification
# - Query patterns：local LLM runtime 判斷 pattern / missing slots，offline query_patterns 作 schema + fallback
# - Rewrite rules：offline mandatory terms + local LLM optional semantic terms 混合 query expansion
# - Final route 仍由 deterministic guardrails + risk_policy override 決定
# - Colab GPU local LLM via HuggingFace Transformers, no OpenAI API key required
# ============================================================

# -----------------------------
# 0. Install packages in Colab
# -----------------------------
import sys, subprocess, pkgutil, os

REQUIRED_PACKAGES = {
    "python-docx": "docx",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "rank-bm25": "rank_bm25",
    "networkx": "networkx",
    "langgraph": "langgraph",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "torch": "torch",
    "pandas": "pandas",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "rich": "rich",
}

def pip_install_if_needed():
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES.items():
        if pkgutil.find_loader(import_name) is None:
            missing.append(pip_name)
    if missing:
        print("Installing missing packages:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U"] + missing)
    else:
        print("All required packages are installed.")

# Auto-install on import only inside Colab, or when AUTO_PIP_INSTALL=true.
# When running from a normal checkout you should install via requirements.txt instead,
# which keeps `import hr_ai_graph_rag` side-effect free.
def _module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

_IN_COLAB_EARLY = _module_available("google.colab")
if _IN_COLAB_EARLY or os.getenv("AUTO_PIP_INSTALL", "false").lower() == "true":
    pip_install_if_needed()

# -----------------------------
# 1. Imports & Global Settings
# -----------------------------
import re
import json
import time
import zipfile
import warnings
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional, Literal, Tuple
from getpass import getpass
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from docx import Document
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

# Heavy / GPU / LLM dependencies (torch, faiss, sentence-transformers, transformers,
# langgraph) are imported lazily inside the components that need them
# (HybridRetriever, LocalHFLLM, HRAssistantGraph). This keeps the data / parsing /
# chunking / graph / evaluation layers importable and runnable without a GPU or the
# generative-LLM stack installed.
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional for the no-LLM pipeline
    torch = None
    _TORCH_AVAILABLE = False

try:
    from IPython.display import display, Markdown, Image
except Exception:  # pragma: no cover - only present in notebooks/Colab
    def display(*args, **kwargs):
        for a in args:
            print(a)

    def Markdown(x=""):
        return x

    def Image(*args, **kwargs):
        return None


def _cuda_available() -> bool:
    return bool(_TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


warnings.filterwarnings("ignore")

# Colab file upload support
try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    files = None
    IN_COLAB = False

# Embedding model for dense retrieval. Default is BAAI/bge-m3 — a strong multilingual
# (incl. Traditional Chinese) embedding model. Override with EMBEDDING_MODEL_NAME.
# bge-m3 is ~2GB and is best on GPU; on CPU it works but is slow.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# Cross-encoder reranker applied to the hybrid-retrieval candidates before the final cut.
# Default is BAAI/bge-reranker-v2-m3 (multilingual). Set USE_RERANKER=false to disable.
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
# How many hybrid candidates to feed into the reranker (before trimming to top_k).
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# Weight blending reranker score with the hybrid score (1.0 = reranker only).
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.7"))

# Local HuggingFace LLM settings (response generation).
# Default is Qwen/Qwen2.5-1.5B-Instruct — open-source, multilingual (incl. Traditional
# Chinese), Colab T4 friendly with 4-bit, and NOT gated (no HF license/token needed).
# For more quality on a larger GPU try Qwen/Qwen2.5-7B-Instruct.
# You may also point this at a gated model such as google/gemma-2-2b-it, but then you
# must accept its license and provide an HF token (`huggingface-cli login` or HF_TOKEN).
HF_LLM_MODEL_NAME = os.getenv("HF_LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_LLM_USE_4BIT = os.getenv("HF_LLM_USE_4BIT", "true").lower() == "true"
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "768"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))

# Runtime local LLM classification is enabled by default in this version.
# It classifies category / intent / matched concepts / ambiguity / candidate risk.
# Final routing still uses deterministic guardrails and risk_policy override.
USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING = os.getenv("USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING", "true").lower() == "true"
# Query patterns and optional rewrite terms are handled by the same runtime classifier.
# Offline artifacts still provide the controlled schema / mandatory terms.
LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS = int(os.getenv("LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS", "5"))

# Local LLM is used for final answer generation; if loading fails, the workflow falls back to template answer.
# Set USE_LLM=false to skip the generative LLM entirely and use the deterministic
# template answer (useful for running the pipeline without the GPU/transformers stack).
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

OUTPUT_DIR = Path("/content/hr_ai_graph_rag_outputs") if IN_COLAB else Path("./hr_ai_graph_rag_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Repo root and bundled data locations (used for zero-config local runs).
# Layout:  <repo>/src/hr_ai_graph_rag/config.py  +  <repo>/data/{policies,golden,hr_offline_artifacts}
# config.py is nested at src/hr_ai_graph_rag/, so the repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd()
REPO_DATA_DIR = REPO_ROOT / "data"

# Optional: you may set these paths manually before running the script/notebook.
# If left blank in Colab, the script will ask you to upload three files:
# 1) 勞動基準法 DOCX, 2) 模擬銀行內規 DOCX, 3) Golden Dataset JSON.
# Outside Colab, paths default to the bundled files under <repo>/data when present.
LABOR_LAW_DOCX_PATH = os.getenv("LABOR_LAW_DOCX_PATH", "").strip()
INTERNAL_POLICY_DOCX_PATH = os.getenv("INTERNAL_POLICY_DOCX_PATH", "").strip()
GOLDEN_DATASET_JSON_PATH = os.getenv("GOLDEN_DATASET_JSON_PATH", "").strip()

if not IN_COLAB:
    _bundled_policy = REPO_DATA_DIR / "policies" / "安久銀行員工工作與福利規章辦法_模擬版.docx"
    _bundled_golden = REPO_DATA_DIR / "golden" / "anjou_bank_hr_ai_golden_dataset_50.json"
    if not INTERNAL_POLICY_DOCX_PATH and _bundled_policy.exists():
        INTERNAL_POLICY_DOCX_PATH = str(_bundled_policy)
    if not GOLDEN_DATASET_JSON_PATH and _bundled_golden.exists():
        GOLDEN_DATASET_JSON_PATH = str(_bundled_golden)

# Offline artifacts can be provided as a folder or a ZIP file.
# Recommended files inside the folder/ZIP:
# workflow_role_mapping.json, concept_nodes.json, risk_policy.json, query_patterns.json,
# rewrite_rules.json, relation_schema.json, graph_relation_candidates.json, local_llm_usage_policy.json
OFFLINE_ARTIFACT_DIR = os.getenv("OFFLINE_ARTIFACT_DIR", "").strip()
OFFLINE_ARTIFACT_ZIP_PATH = os.getenv("OFFLINE_ARTIFACT_ZIP_PATH", "").strip()

if not IN_COLAB and not OFFLINE_ARTIFACT_DIR:
    _bundled_artifacts = REPO_DATA_DIR / "hr_offline_artifacts"
    if (_bundled_artifacts / "concept_nodes.json").exists():
        OFFLINE_ARTIFACT_DIR = str(_bundled_artifacts)

# For safety, only approved graph edges are loaded by default.
LOAD_PENDING_GRAPH_EDGES = os.getenv("LOAD_PENDING_GRAPH_EDGES", "false").lower() == "true"

# Important: keep evaluation data out of the knowledge base by default.
# Set to true only for retrieval experiments. For honest evaluation, leave it false.
USE_GOLDEN_AS_FAQ_CHUNKS = os.getenv("USE_GOLDEN_AS_FAQ_CHUNKS", "false").lower() == "true"

print("IN_COLAB =", IN_COLAB)
print("torch available =", _TORCH_AVAILABLE)
print("CUDA available =", _cuda_available())
if _cuda_available():
    print("GPU =", torch.cuda.get_device_name(0))
print("USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING =", USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING)
print("LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS =", LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS)
print("HF_LLM_MODEL_NAME =", HF_LLM_MODEL_NAME)
print("HF_LLM_USE_4BIT =", HF_LLM_USE_4BIT)
print("EMBEDDING_MODEL =", EMBEDDING_MODEL_NAME)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("LABOR_LAW_DOCX_PATH =", LABOR_LAW_DOCX_PATH or "<upload required>")
print("INTERNAL_POLICY_DOCX_PATH =", INTERNAL_POLICY_DOCX_PATH or "<upload required>")
print("GOLDEN_DATASET_JSON_PATH =", GOLDEN_DATASET_JSON_PATH or "<upload required>")
print("USE_GOLDEN_AS_FAQ_CHUNKS =", USE_GOLDEN_AS_FAQ_CHUNKS)
print("OFFLINE_ARTIFACT_DIR =", OFFLINE_ARTIFACT_DIR or "<auto-detect or upload optional>")
print("OFFLINE_ARTIFACT_ZIP_PATH =", OFFLINE_ARTIFACT_ZIP_PATH or "<auto-detect or upload optional>")
print("LOAD_PENDING_GRAPH_EDGES =", LOAD_PENDING_GRAPH_EDGES)
