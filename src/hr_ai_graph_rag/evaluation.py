# ============================================================
# evaluation — golden 載入與評估指標
#
# 載入 50 題 golden set,計算 category/route/retrieval/citation/
# faithfulness/latency 等指標,並提供使用者回饋紀錄。
# 依賴:config、utils、workflow。
# ============================================================

from .config import *
from .utils import *
from .workflow import *


def load_golden_dataset(json_path: str) -> pd.DataFrame:
    """Load user-provided Golden Dataset JSON.

    Supported structures:
    1) {"items": [{...}, {...}]}
    2) [{...}, {...}]

    Required field: question
    Recommended fields: id, question_type, expected_category, expected_route,
    expected_citations, expected_key_points, should_escalate, should_clarify.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("items", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Golden Dataset JSON must be a list or an object with an 'items' list.")
    df = pd.DataFrame(items)
    if "question" not in df.columns:
        raise ValueError("Golden Dataset must contain a 'question' field.")
    if "expected_route" not in df.columns:
        df["expected_route"] = np.where(df.get("should_clarify", False), "clarify", np.where(df.get("should_escalate", False), "escalate", "answer"))
    if "expected_category" not in df.columns:
        df["expected_category"] = df["question"].apply(detect_category)
    if "expected_citations" not in df.columns:
        df["expected_citations"] = [[] for _ in range(len(df))]
    if "expected_key_points" not in df.columns:
        df["expected_key_points"] = [[] for _ in range(len(df))]
    if "id" not in df.columns:
        df["id"] = [f"G{i+1:03d}" for i in range(len(df))]
    return df


def citation_retrieval_hit(expected_citations: Any, citations: List[Dict[str, Any]]) -> Optional[bool]:
    if expected_citations is None or (isinstance(expected_citations, float) and np.isnan(expected_citations)):
        return None
    if isinstance(expected_citations, str):
        try:
            parsed = json.loads(expected_citations)
            expected_citations = parsed
        except Exception:
            expected_citations = [expected_citations]
    if not isinstance(expected_citations, list):
        expected_citations = [expected_citations]
    expected_citations = [str(x).strip() for x in expected_citations if str(x).strip()]
    if not expected_citations:
        return None

    retrieved_text = normalize_for_match(" ".join([safe_json_dumps(c) for c in citations]))
    for exp in expected_citations:
        exp_norm = normalize_for_match(exp)
        if exp_norm and exp_norm in retrieved_text:
            return True
        # Match by article number, e.g. expected citation「安久銀行...第11條」 vs retrieved article_no「第 11 條」.
        for ref in extract_article_refs(exp):
            if ref and ref in retrieved_text:
                return True
    return False


def source_type_hit(expected_source_type: Any, citations: List[Dict[str, Any]]) -> Optional[bool]:
    if expected_source_type is None or (isinstance(expected_source_type, float) and np.isnan(expected_source_type)):
        return None
    if isinstance(expected_source_type, str):
        try:
            parsed = json.loads(expected_source_type)
            expected_source_type = parsed
        except Exception:
            expected_source_type = [expected_source_type]
    if not isinstance(expected_source_type, list):
        expected_source_type = [expected_source_type]
    expected = {str(x).strip() for x in expected_source_type if str(x).strip()}
    if not expected:
        return None
    actual = {str(c.get("source_type", "")).strip() for c in citations}
    return bool(expected & actual)


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars so json round-trips to native types.

    Critical for the checkpoint: a resumed np.float64 written via json would otherwise
    come back as a Python float fine, but np.bool_/np.int* need .item() and NaN must
    become null so summary means stay numeric after a resume.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


RAGAS_FIELDS = ("ragas_faithfulness", "ragas_answer_relevancy",
                "ragas_context_precision", "ragas_context_recall")


def _coerce_str_list(val: Any) -> List[str]:
    """Normalize a golden field (list / JSON string / scalar / NaN) into a string list."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return [val.strip()] if val.strip() else []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in list(val) if str(x).strip()]
    s = str(val).strip()
    return [s] if s else []


# Golden columns that may hold a full reference answer, in priority order. The dataset
# ships only expected_key_points, but if you add a free-text answer column (any of these),
# the RAGAS context metrics use it instead — see _resolve_ground_truths.
_REFERENCE_ANSWER_COLS = ("expected_answer", "reference_answer", "ground_truth", "golden_answer")


def _resolve_ground_truths(row: pd.Series) -> List[str]:
    """Ground truth (a list of statements) for RAGAS context recall / precision.

    Prefers a full reference answer (expected_answer / reference_answer / ground_truth /
    golden_answer) when the golden row has one, decomposing free text into sentence-level
    statements so recall/precision keep per-statement granularity (this matches RAGAS,
    which breaks the ground-truth answer into statements). Falls back to the pre-decomposed
    expected_key_points. Returns [] when neither is present (→ those two metrics become None).
    """
    for col in _REFERENCE_ANSWER_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, (list, tuple, np.ndarray)):
            items = _coerce_str_list(val)  # already a list of statements → use as-is
            if items:
                return items
            continue
        s = str(val).strip()
        if not s or s.lower() == "nan":
            continue
        sents = split_sentences_zh(s)  # free-text answer → decompose into statements
        return sents if sents else [s]
    return _coerce_str_list(row.get("expected_key_points"))


class RagasScorer:
    """Embedding-based RAGAS-style metrics — no external LLM/API call.

    Reuses the retriever's bge-m3 embedder so similarities use the same multilingual
    model as retrieval. Cosine sims use normalized embeddings (dot product). Any metric
    whose inputs are missing returns None (e.g. no ground-truth key points → no context
    precision/recall), so summary means stay honest. Definitions:

    - faithfulness:        mean over answer sentences of their max similarity to any
                           retrieved context (is the answer grounded in the context?).
    - answer_relevancy:    similarity of the whole answer to the question.
    - context_precision:   average precision of the ranked contexts, a context counting
                           as relevant when it is ≥ threshold-similar to a ground-truth
                           key point (relevant contexts ranked higher → higher score).
    - context_recall:      fraction of ground-truth key points each ≥ threshold-similar to
                           at least one retrieved context (was the truth retrieved?).
    """

    def __init__(self, embedder: Any = None, sim_threshold: Optional[float] = None):
        self.embedder = embedder
        self.threshold = float(os.getenv("RAGAS_SIM_THRESHOLD", "0.5")) if sim_threshold is None else sim_threshold
        self._cache: Dict[Tuple[str, ...], Any] = {}

    def available(self) -> bool:
        return self.embedder is not None

    def _emb(self, texts: List[str]):
        key = tuple(texts)
        if key in self._cache:
            return self._cache[key]
        arr = np.asarray(
            self.embedder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True),
            dtype="float32",
        )
        self._cache[key] = arr
        return arr

    def _sim(self, a_texts: List[str], b_texts: List[str]):
        if not a_texts or not b_texts:
            return None
        return self._emb(a_texts) @ self._emb(b_texts).T  # (len_a, len_b), normalized → cosine

    def faithfulness(self, answer: str, contexts: List[str]) -> Optional[float]:
        sents = split_sentences_zh(answer) or ([answer.strip()] if (answer or "").strip() else [])
        M = self._sim(sents, contexts)
        return None if M is None else float(np.clip(M.max(axis=1).mean(), 0.0, 1.0))

    def answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        if not (question or "").strip() or not (answer or "").strip():
            return None
        M = self._sim([answer], [question])
        return None if M is None else float(np.clip(M[0, 0], 0.0, 1.0))

    def context_recall(self, contexts: List[str], ground_truths: List[str]) -> Optional[float]:
        M = self._sim(ground_truths, contexts)  # (gt, ctx)
        return None if M is None else float((M.max(axis=1) >= self.threshold).mean())

    def context_precision(self, contexts: List[str], ground_truths: List[str]) -> Optional[float]:
        M = self._sim(contexts, ground_truths)  # (ctx, gt), contexts already in rank order
        if M is None:
            return None
        rel = (M.max(axis=1) >= self.threshold).astype(float)
        if rel.sum() == 0:
            return 0.0
        hits, precision_sum = 0, 0.0
        for i, r in enumerate(rel, start=1):
            if r > 0:
                hits += 1
                precision_sum += hits / i
        return float(precision_sum / hits)

    def score(self, question: str, answer: str, contexts: List[str], ground_truths: List[str]) -> Dict[str, Optional[float]]:
        return {
            "ragas_faithfulness": self.faithfulness(answer, contexts),
            "ragas_answer_relevancy": self.answer_relevancy(question, answer),
            "ragas_context_precision": self.context_precision(contexts, ground_truths),
            "ragas_context_recall": self.context_recall(contexts, ground_truths),
        }


class LlmRagasScorer:
    """RAGAS metrics judged by the local Qwen LLM (the SAME loaded model used for answer
    generation, reused via call_llm_json — no reload, no external API).

    This follows the canonical RAGAS LLM-as-judge formulation (statement-level support /
    attribution / per-context relevance), unlike the embedding-similarity proxy in
    RagasScorer. It is heavier — up to 4 judge calls per question — but every score is
    checkpointed, so a Colab disconnect mid-eval still resumes. A metric is None when its
    judge output can't be parsed (small models occasionally emit invalid JSON) or its
    inputs are missing, so the summary mean stays honest.
    """

    def __init__(self, max_contexts: Optional[int] = None, ctx_chars: Optional[int] = None):
        self.max_contexts = int(os.getenv("RAGAS_LLM_MAX_CONTEXTS", str(max_contexts or 6)))
        self.ctx_chars = int(os.getenv("RAGAS_LLM_CTX_CHARS", str(ctx_chars or 400)))

    def available(self) -> bool:
        # Relies on call_llm_json's own None-fallback; the model is already loaded by the
        # answer-generation pass, so we don't pre-load it here.
        return True

    def _fmt_contexts(self, contexts: List[str]) -> str:
        return "\n".join(f"[{i}] {str(c)[:self.ctx_chars]}"
                         for i, c in enumerate(contexts[:self.max_contexts], start=1))

    @staticmethod
    def _mean_bool(items: Any, key: str) -> Optional[float]:
        if not isinstance(items, list) or not items:
            return None
        vals = [1.0 if bool(it.get(key)) else 0.0 for it in items if isinstance(it, dict)]
        return float(sum(vals) / len(vals)) if vals else None

    def faithfulness(self, answer: str, contexts: List[str]) -> Optional[float]:
        if not (answer or "").strip() or not contexts:
            return None
        res = call_llm_json(
            "你是嚴格的事實查核員。只能依據提供的參考內容判斷,不可臆測或引入外部知識。",
            f"參考內容:\n{self._fmt_contexts(contexts)}\n\n答案:\n{answer}\n\n"
            "請把答案拆成數個獨立陳述,逐一判斷每個陳述是否能由參考內容支持。"
            '輸出 JSON:{"statements":[{"text":"...","supported":true}]}',
            default={})
        return self._mean_bool(res.get("statements") if isinstance(res, dict) else None, "supported")

    def context_recall(self, contexts: List[str], ground_truths: List[str]) -> Optional[float]:
        if not ground_truths or not contexts:
            return None
        pts = "\n".join(f"{i}. {p}" for i, p in enumerate(ground_truths, start=1))
        res = call_llm_json(
            "你是嚴格的查核員,只依參考內容判斷,不可臆測。",
            f"參考內容:\n{self._fmt_contexts(contexts)}\n\n標準答案要點:\n{pts}\n\n"
            "逐點判斷該要點是否能在參考內容中找到依據。"
            '輸出 JSON:{"points":[{"text":"...","attributable":true}]}',
            default={})
        return self._mean_bool(res.get("points") if isinstance(res, dict) else None, "attributable")

    def context_precision(self, contexts: List[str], ground_truths: List[str]) -> Optional[float]:
        if not contexts or not ground_truths:
            return None
        n = min(len(contexts), self.max_contexts)
        res = call_llm_json(
            "你判斷每段檢索段落對回答此問題是否相關有用,只輸出 JSON。",
            f"標準答案要點:{'；'.join(ground_truths)}\n\n檢索段落(已依檢索排名):\n"
            f"{self._fmt_contexts(contexts)}\n\n逐段判斷該段落是否與標準答案相關。"
            '輸出 JSON:{"contexts":[{"index":1,"relevant":true}]}',
            default={})
        items = res.get("contexts") if isinstance(res, dict) else None
        if not isinstance(items, list) or not items:
            return None
        rel = [0.0] * n
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("index"), (int, float)):
                idx = int(it["index"])
                if 1 <= idx <= n:
                    rel[idx - 1] = 1.0 if bool(it.get("relevant")) else 0.0
        if sum(rel) == 0:
            return 0.0
        hits, precision_sum = 0, 0.0
        for i, r in enumerate(rel, start=1):
            if r > 0:
                hits += 1
                precision_sum += hits / i
        return float(precision_sum / hits)

    def answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        if not (question or "").strip() or not (answer or "").strip():
            return None
        res = call_llm_json(
            "你評估答案是否切題地回應問題,只輸出 JSON。",
            f"問題:{question}\n答案:{answer}\n\n"
            "請給 0 到 100 的整數分數(100=完全切題並直接回答問題,0=完全離題)。"
            '輸出 JSON:{"score":0}',
            default={})
        sc = res.get("score") if isinstance(res, dict) else None
        if not isinstance(sc, (int, float)) or isinstance(sc, bool):
            return None
        return float(min(1.0, max(0.0, sc / 100.0)))

    def score(self, question: str, answer: str, contexts: List[str], ground_truths: List[str]) -> Dict[str, Optional[float]]:
        return {
            "ragas_faithfulness": self.faithfulness(answer, contexts),
            "ragas_answer_relevancy": self.answer_relevancy(question, answer),
            "ragas_context_precision": self.context_precision(contexts, ground_truths),
            "ragas_context_recall": self.context_recall(contexts, ground_truths),
        }


def _make_ragas_scorer(assistant: "HRAssistantGraph"):
    """Build the RAGAS scorer chosen by RAGAS_BACKEND, unless USE_RAGAS=false.

    - RAGAS_BACKEND=llm (default): the local Qwen as judge (reuses the loaded model;
                                   ~4 extra LLM calls per question — slower, canonical RAGAS).
    - RAGAS_BACKEND=embedding:     fast, embedding-similarity proxy (bge-m3, no extra LLM).
    """
    if os.getenv("USE_RAGAS", "true").lower() != "true":
        return None
    backend = os.getenv("RAGAS_BACKEND", "llm").strip().lower()
    if backend == "llm":
        return LlmRagasScorer()
    embedder = getattr(getattr(assistant, "retriever", None), "embedder", None)
    scorer = RagasScorer(embedder)
    if not scorer.available():
        print("[ragas] 找不到 embedder,RAGAS metrics 將為 None(設 USE_RAGAS=false 可關閉,"
              "或用 RAGAS_BACKEND=llm 改用 Qwen 當 judge)")
        return None
    return scorer


def _build_eval_entry(row: pd.Series, result: Dict[str, Any], latency: float,
                      scorer: Optional[RagasScorer] = None) -> Dict[str, Any]:
    """One unified per-question entry; both the detail row and the full record derive
    from it, so the checkpoint stores everything needed to rebuild outputs on resume."""
    q = row["question"]
    citations = result.get("citations", [])
    ragas = {k: None for k in RAGAS_FIELDS}
    if scorer is not None:
        contexts = [str(c.get("content", "")) for c in result.get("retrieved_chunks", [])
                    if str(c.get("content", "")).strip()][:8]
        ground_truths = _resolve_ground_truths(row)
        try:
            ragas = scorer.score(q, result.get("answer", ""), contexts, ground_truths)
        except Exception as e:  # never let metric scoring break the eval loop
            print(f"[ragas] scoring failed for id={row.get('id')}: {e!r}")
    expected_route = row.get("expected_route", "answer")
    expected_category = row.get("expected_category", "")
    retrieval_hit = citation_retrieval_hit(row.get("expected_citations", []), citations)
    src_hit = source_type_hit(row.get("expected_source_type", []), citations)
    return {
        "id": row.get("id"),
        "question_type": row.get("question_type"),
        "test_dimension": row.get("test_dimension"),
        "question": q,
        "expected_category": expected_category,
        "actual_category": result.get("category"),
        "category_correct": category_matches(expected_category, result.get("category")),
        "expected_route": expected_route,
        "actual_route": result.get("route"),
        "route_correct": result.get("route") == expected_route,
        "expected_citations": row.get("expected_citations", []),
        "retrieval_hit": retrieval_hit,
        "expected_source_type": row.get("expected_source_type", []),
        "source_type_hit": src_hit,
        "citation_present": bool(citations) if result.get("route") not in ["clarify"] else True,
        "risk_level": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "faithfulness_score": result.get("faithfulness_score"),
        "latency_sec": round(latency, 3),
        "answer": result.get("answer", ""),     # complete generated answer
        "citations": citations,                  # complete reference sources
        **{k: ragas.get(k) for k in RAGAS_FIELDS},
    }


def _entry_to_row(e: Dict[str, Any]) -> Dict[str, Any]:
    ans = e.get("answer", "") or ""
    return {
        "id": e.get("id"), "question_type": e.get("question_type"), "test_dimension": e.get("test_dimension"),
        "question": e.get("question"), "expected_category": e.get("expected_category"),
        "actual_category": e.get("actual_category"), "category_correct": e.get("category_correct"),
        "expected_route": e.get("expected_route"), "actual_route": e.get("actual_route"),
        "route_correct": e.get("route_correct"), "expected_citations": e.get("expected_citations"),
        "retrieval_hit": e.get("retrieval_hit"), "expected_source_type": e.get("expected_source_type"),
        "source_type_hit": e.get("source_type_hit"), "citation_present": e.get("citation_present"),
        "confidence": e.get("confidence"), "faithfulness_score": e.get("faithfulness_score"),
        "latency_sec": e.get("latency_sec"), "answer_preview": ans[:240],
        "answer_full": ans, "citations_json": safe_json_dumps(e.get("citations", [])),
        **{k: e.get(k) for k in RAGAS_FIELDS},
    }


def _entry_to_record(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": e.get("id"), "question_type": e.get("question_type"), "test_dimension": e.get("test_dimension"),
        "question": e.get("question"), "expected_category": e.get("expected_category"),
        "actual_category": e.get("actual_category"), "category_correct": e.get("category_correct"),
        "expected_route": e.get("expected_route"), "actual_route": e.get("actual_route"),
        "route_correct": e.get("route_correct"), "risk_level": e.get("risk_level"),
        "confidence": e.get("confidence"), "faithfulness_score": e.get("faithfulness_score"),
        "retrieval_hit": e.get("retrieval_hit"), "source_type_hit": e.get("source_type_hit"),
        "citation_present": e.get("citation_present"), "latency_sec": e.get("latency_sec"),
        **{k: e.get(k) for k in RAGAS_FIELDS},
        "answer": e.get("answer", ""), "citations": e.get("citations", []),
    }


def _load_eval_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a JSONL checkpoint into {id: entry}. Tolerates a half-written trailing line
    from a hard Colab disconnect by skipping any line that fails to parse."""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue  # partial last line after a crash — drop it, it will be re-run
        if e.get("id") is not None:
            done[str(e["id"])] = e  # later line wins, so a re-run overwrites cleanly
    return done


def eval_checkpoint_count(checkpoint_path: Optional[Path] = None) -> int:
    """How many questions are already saved in the checkpoint (for resume-aware callers)."""
    path = Path(checkpoint_path) if checkpoint_path is not None else EVAL_CHECKPOINT_PATH
    return len(_load_eval_checkpoint(path))


def evaluate_assistant(
    assistant: HRAssistantGraph,
    golden_df: pd.DataFrame,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every golden question, checkpointing after each one so a Colab disconnect
    loses at most the in-flight question.

    Each finished question is appended (flushed + fsync'd) as one JSONL line to
    `checkpoint_path` (default EVAL_CHECKPOINT_PATH, inside OUTPUT_DIR — point OUTPUT_DIR at
    Google Drive to survive the 50-min runtime cap). On the next run, completed questions
    are loaded from the checkpoint and skipped, so the LLM only generates the remaining
    ones; the final detail/summary are always computed over ALL questions (resumed + new).
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else EVAL_CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_eval_checkpoint(checkpoint_path) if resume else {}
    if done:
        print(f"[checkpoint] resume:已從 checkpoint 載入 {len(done)} 題,將略過這些題目 → {checkpoint_path}")

    scorer = _make_ragas_scorer(assistant)  # RAGAS metrics scorer (None if disabled)
    if isinstance(scorer, LlmRagasScorer):
        print("[ragas] RAGAS metrics 啟用 — backend=llm(Qwen as judge,每題約 4 次 LLM 呼叫)")
    elif scorer is not None:
        print(f"[ragas] RAGAS metrics 啟用 — backend=embedding(bge-m3, sim_threshold={scorer.threshold})")

    entries: List[Dict[str, Any]] = []  # in golden order, for deterministic output
    n_new = 0
    fh = open(checkpoint_path, "a", encoding="utf-8")  # append: prior lines survive
    try:
        for _, row in tqdm(golden_df.iterrows(), total=len(golden_df), desc="Evaluating"):
            rid = str(row.get("id"))
            if rid in done:
                entries.append(done[rid])  # reuse the saved answer, no LLM call
                continue
            start = time.time()
            result = assistant.ask(row["question"])
            latency = time.time() - start
            entry = _to_native(_build_eval_entry(row, result, latency, scorer))
            entries.append(entry)
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # force the line to disk/Drive before the next question
            n_new += 1
    finally:
        fh.close()
    print(f"[checkpoint] 本次新跑 {n_new} 題,checkpoint 共 {len(entries)} 題 → {checkpoint_path}")

    rows = [_entry_to_row(e) for e in entries]
    records = [_entry_to_record(e) for e in entries]
    detail = pd.DataFrame(rows)
    detail.attrs["full_records"] = records  # carried to save_outputs for the JSON export
    valid_hits = [x for x in detail["retrieval_hit"].tolist() if isinstance(x, (bool, np.bool_))]
    valid_src_hits = [x for x in detail["source_type_hit"].tolist() if isinstance(x, (bool, np.bool_))]

    def _mean_opt(col: str) -> Optional[float]:
        """Mean over non-null numeric values (RAGAS metrics are None when not computable)."""
        vals = [v for v in detail[col].tolist()
                if isinstance(v, (int, float, np.floating, np.integer))
                and not (isinstance(v, float) and np.isnan(v))]
        return (sum(map(float, vals)) / len(vals)) if vals else None

    summary = pd.DataFrame([{
        "Total Questions": len(detail),
        "Category Accuracy": detail["category_correct"].mean(),
        "Route Accuracy": detail["route_correct"].mean(),
        "Retrieval Hit Rate": (sum(valid_hits) / len(valid_hits)) if valid_hits else None,
        "Source Type Hit Rate": (sum(valid_src_hits) / len(valid_src_hits)) if valid_src_hits else None,
        "Citation Present Rate": detail["citation_present"].mean(),
        "Avg Faithfulness Score": detail["faithfulness_score"].mean(),
        # RAGAS-style metrics (embedding-based; None-skipping means)
        "RAGAS Context Recall": _mean_opt("ragas_context_recall"),
        "RAGAS Context Precision": _mean_opt("ragas_context_precision"),
        "RAGAS Faithfulness": _mean_opt("ragas_faithfulness"),
        "RAGAS Answer Relevancy": _mean_opt("ragas_answer_relevancy"),
        "Avg Latency Sec": detail["latency_sec"].mean(),
    }])
    return detail, summary


def log_feedback(output_dir: Path, question: str, answer: str, helpful: bool, correctness_score: int, completeness_score: int, comment: str = "") -> pd.DataFrame:
    path = output_dir / "feedback_log.csv"
    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "question": question,
        "answer": answer,
        "helpful": helpful,
        "correctness_score": correctness_score,
        "completeness_score": completeness_score,
        "comment": comment,
    }
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df

# -----------------------------
# 10. Colab Runner
# -----------------------------
