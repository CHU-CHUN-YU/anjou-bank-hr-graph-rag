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


def _build_eval_entry(row: pd.Series, result: Dict[str, Any], latency: float) -> Dict[str, Any]:
    """One unified per-question entry; both the detail row and the full record derive
    from it, so the checkpoint stores everything needed to rebuild outputs on resume."""
    q = row["question"]
    citations = result.get("citations", [])
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
            entry = _to_native(_build_eval_entry(row, result, latency))
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
    summary = pd.DataFrame([{
        "Total Questions": len(detail),
        "Category Accuracy": detail["category_correct"].mean(),
        "Route Accuracy": detail["route_correct"].mean(),
        "Retrieval Hit Rate": (sum(valid_hits) / len(valid_hits)) if valid_hits else None,
        "Source Type Hit Rate": (sum(valid_src_hits) / len(valid_src_hits)) if valid_src_hits else None,
        "Citation Present Rate": detail["citation_present"].mean(),
        "Avg Faithfulness Score": detail["faithfulness_score"].mean(),
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
