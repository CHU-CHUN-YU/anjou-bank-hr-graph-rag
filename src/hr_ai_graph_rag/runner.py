# ============================================================
# runner — 輸入處理、結果顯示/存檔、main() 進入點
#
# 解析或上傳三個輸入檔、組裝整條 pipeline、顯示與輸出結果。
# 以 `python -m hr_ai_graph_rag` 執行(見 __main__.py)。
# 依賴:其餘所有模組。
# ============================================================

from .config import *
from .utils import *
from .artifacts import *
from .ingestion import *
from .graph import *
from .retrieval import *
from .workflow import *
from .evaluation import *


def _resolve_existing_path(path_str: str) -> Optional[str]:
    if path_str and Path(path_str).exists():
        return str(Path(path_str))
    return None


def _identify_uploaded_files(uploaded_names: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    docx_files = [fn for fn in uploaded_names if fn.lower().endswith(".docx")]
    json_files = [fn for fn in uploaded_names if fn.lower().endswith(".json")]

    law_docx = None
    policy_docx = None
    golden_json = json_files[0] if json_files else None

    for fn in docx_files:
        if any(k in fn for k in ["勞動基準法", "勞基法", "labor"]):
            law_docx = fn
        elif any(k in fn for k in ["內規", "規章", "員工工作", "員工手冊", "policy", "福利規章"]):
            policy_docx = fn

    # Fallback by order if names are ambiguous.
    remaining = [fn for fn in docx_files if fn not in {law_docx, policy_docx}]
    if law_docx is None and remaining:
        law_docx = remaining.pop(0)
    if policy_docx is None and remaining:
        policy_docx = remaining.pop(0)

    if IN_COLAB:
        law_docx = f"/content/{law_docx}" if law_docx else None
        policy_docx = f"/content/{policy_docx}" if policy_docx else None
        golden_json = f"/content/{golden_json}" if golden_json else None
    return law_docx, policy_docx, golden_json


def prepare_input_files() -> Tuple[str, str, str]:
    """Resolve or upload the three required user-provided files.

    Required:
    1. 勞動基準法 DOCX
    2. 模擬銀行內規 DOCX
    3. Golden Dataset JSON
    """
    law_docx = _resolve_existing_path(LABOR_LAW_DOCX_PATH)
    policy_docx = _resolve_existing_path(INTERNAL_POLICY_DOCX_PATH)
    golden_json = _resolve_existing_path(GOLDEN_DATASET_JSON_PATH)

    if law_docx and policy_docx and golden_json:
        return law_docx, policy_docx, golden_json

    if IN_COLAB:
        print("請一次上傳三個檔案：")
        print("1) 勞動基準法 DOCX")
        print("2) 模擬銀行員工內部規章 DOCX")
        print("3) Golden Dataset JSON")
        uploaded = files.upload()
        ulaw, upolicy, ugolden = _identify_uploaded_files(list(uploaded.keys()))
        law_docx = law_docx or ulaw
        policy_docx = policy_docx or upolicy
        golden_json = golden_json or ugolden
    else:
        # Local convenience: search current directory and /mnt/data if paths are not set.
        candidates = list(Path(".").glob("*")) + list(Path("/mnt/data").glob("*"))
        names = [str(p) for p in candidates]
        ulaw, upolicy, ugolden = _identify_uploaded_files(names)
        law_docx = law_docx or ulaw
        policy_docx = policy_docx or upolicy
        golden_json = golden_json or ugolden

        # The repo ships the internal policy DOCX + golden JSON but not the official
        # 勞動基準法 DOCX (that is an external legal source). For a zero-config local
        # run, fall back to the built-in sample labor-law DOCX so the pipeline can run
        # end-to-end. Provide a real 勞動基準法 DOCX via LABOR_LAW_DOCX_PATH for production.
        if not (law_docx and Path(law_docx).exists()):
            sample_path = str(OUTPUT_DIR / "參考資料_勞動基準法_sample.docx")
            print("未提供勞動基準法 DOCX，改用內建 sample 條文 ->", sample_path)
            create_sample_labor_law_docx(sample_path)
            law_docx = sample_path

    missing = []
    if not law_docx or not Path(law_docx).exists():
        missing.append("勞動基準法 DOCX")
    if not policy_docx or not Path(policy_docx).exists():
        missing.append("模擬銀行內規 DOCX")
    if not golden_json or not Path(golden_json).exists():
        missing.append("Golden Dataset JSON")
    if missing:
        raise FileNotFoundError("缺少必要檔案：" + ", ".join(missing) + "。請上傳或設定環境變數 LABOR_LAW_DOCX_PATH / INTERNAL_POLICY_DOCX_PATH / GOLDEN_DATASET_JSON_PATH。")

    return str(law_docx), str(policy_docx), str(golden_json)


def display_result(result: HRState):
    display(Markdown("## 使用者問題"))
    display(Markdown(result.get("question", "")))
    display(Markdown("## AI 回答"))
    display(Markdown(result.get("answer", "")))
    meta = {
        "intent": result.get("intent"),
        "category": result.get("category"),
        "risk_level": result.get("risk_level"),
        "answer_policy": result.get("answer_policy"),
        "route": result.get("route"),
        "confidence": result.get("confidence"),
        "faithfulness_score": result.get("faithfulness_score"),
    }
    display(Markdown("## 系統判斷"))
    display(pd.DataFrame([meta]))
    display(Markdown("## 引用來源"))
    if result.get("citations"):
        display(pd.DataFrame(result["citations"]))
    else:
        display(Markdown("無 citation。"))
    display(Markdown("## Debug Trace"))
    for d in result.get("debug", []):
        print("-", d)


def save_outputs(output_dir: Path, articles, chunks, kg: HRKnowledgeGraph, demo_results: List[Dict[str, Any]], golden_df, eval_detail, eval_summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    with open(output_dir / "chunks_3layer_faq.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(output_dir / "demo_results.json", "w", encoding="utf-8") as f:
        json.dump(demo_results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(chunks).drop(columns=["embedding_text"], errors="ignore").to_csv(output_dir / "chunks_3layer_faq.csv", index=False, encoding="utf-8-sig")
    golden_df.to_csv(output_dir / "golden_dataset.csv", index=False, encoding="utf-8-sig")
    eval_detail.to_csv(output_dir / "evaluation_detail.csv", index=False, encoding="utf-8-sig")
    eval_summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    kg.save_graph_files(output_dir)
    readme = f"""
# 安久銀行 HR AI 智能助理 - Colab Technical Demo

## 技術內容
- 使用者上傳模擬員工內規 DOCX：補足企業內部規則層
- Policy-aware Hierarchical Chunking：Document-level / Article-level / Semantic sub-chunk
- FAQ Chunk：預設不使用 Golden Dataset 進知識庫；若設 USE_GOLDEN_AS_FAQ_CHUNKS=true 才做實驗性 FAQ chunk
- Hybrid Retrieval：Vector + BM25 + keyword + metadata priority
- Offline Artifacts：讀取 concept_nodes / risk_policy / query_patterns / rewrite_rules / relation_schema / graph_relation_candidates JSON
- Knowledge Graph / Graph RAG：建立 law / internal_policy / concept 節點與 approved graph relations
- LangGraph Workflow：Runtime Local LLM Classification -> Retrieval Orchestrator -> Deterministic Guardrails -> Answer / Disclaimer / Clarify / Escalate
- HuggingFace Local LLM：Colab GPU 載入 Instruct model，不需 OpenAI API key；負責分類 signal 與 grounded answer generation
- 使用者上傳 Golden Dataset JSON Evaluation：category, route, retrieval, source type, citation, faithfulness, latency

## 主要輸出
- articles.json
- chunks_3layer_faq.json / csv
- kg_nodes.csv / kg_edges.csv / hr_knowledge_graph.gexf
- golden_dataset.csv
- evaluation_detail.csv
- evaluation_summary.csv
- demo_results.json
- feedback_log.csv
""".strip()
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    zip_path = output_dir / "hr_ai_graph_rag_outputs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for fp in output_dir.glob("*"):
            if fp.is_file() and fp.name != zip_path.name:
                z.write(fp, arcname=fp.name)
    print("Saved outputs to:", output_dir)
    print("ZIP:", zip_path)
    if IN_COLAB:
        try:
            files.download(str(zip_path))
        except Exception as e:
            print("Download failed:", repr(e))
    return zip_path


def main():
    labor_docx_path, policy_docx_path, golden_json_path = prepare_input_files()
    print("Labor Law DOCX path:", labor_docx_path)
    print("Internal Policy DOCX path:", policy_docx_path)
    print("Golden Dataset JSON path:", golden_json_path)

    offline_artifacts = load_offline_artifacts()
    golden_df = load_golden_dataset(golden_json_path)
    print(f"Golden Dataset questions: {len(golden_df)}")
    display(Markdown("# Uploaded Golden Dataset Preview"))
    display(golden_df.head(10))

    builder = HRKnowledgeBuilder(ChunkConfig(version="PoC-v1", effective_date="2026-06-23"))
    articles, chunks = builder.build_chunks(
        labor_law_docx_path=labor_docx_path,
        internal_policy_docx_path=policy_docx_path,
        golden_df=golden_df,
    )
    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(chunks)}")
    display(Markdown("# Knowledge Chunks Preview"))
    display(pd.DataFrame(chunks).drop(columns=["embedding_text"], errors="ignore").head(12))

    kg = HRKnowledgeGraph(articles, chunks, artifacts=offline_artifacts)
    print("KG nodes:", kg.G.number_of_nodes(), "edges:", kg.G.number_of_edges())

    retriever = HybridRetriever(chunks)
    assistant = HRAssistantGraph(retriever, kg, artifacts=offline_artifacts)

    # Use first 9 questions from the uploaded Golden Dataset as demo questions.
    demo_questions = golden_df["question"].dropna().astype(str).head(9).tolist()

    demo_results = []
    print("\nRunning demo questions...")
    for q in demo_questions:
        r = assistant.ask(q)
        demo_results.append(r)

    # Show first 3 detailed results
    for r in demo_results[:3]:
        display_result(r)

    batch_df = pd.DataFrame([{
        "question": r.get("question"),
        "intent": r.get("intent"),
        "category": r.get("category"),
        "route": r.get("route"),
        "risk_level": r.get("risk_level"),
        "confidence": r.get("confidence"),
        "faithfulness_score": r.get("faithfulness_score"),
        "answer_preview": r.get("answer", "")[:160],
    } for r in demo_results])
    display(Markdown("# Batch Demo Summary"))
    display(batch_df)

    # Evaluation on uploaded Golden Dataset JSON
    eval_detail, eval_summary = evaluate_assistant(assistant, golden_df)
    display(Markdown("# Evaluation Detail"))
    display(eval_detail)
    display(Markdown("# Evaluation Summary"))
    display(eval_summary)

    # Feedback example based on first demo question
    if demo_results:
        log_feedback(
            OUTPUT_DIR,
            question=demo_results[0]["question"],
            answer=demo_results[0]["answer"],
            helpful=True,
            correctness_score=5,
            completeness_score=4,
            comment="Demo feedback: external DOCX/JSON ingestion version.",
        )

    # Save loaded offline artifact summary for auditability before packaging outputs.
    with open(OUTPUT_DIR / "loaded_offline_artifacts.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_dir": str(offline_artifacts.artifact_dir), "loaded_files": offline_artifacts.loaded_files}, f, ensure_ascii=False, indent=2)

    save_outputs(OUTPUT_DIR, articles, chunks, kg, demo_results, golden_df, eval_detail, eval_summary)
    print("\n全部流程完成。")
    return assistant, articles, chunks, kg, retriever, golden_df, offline_artifacts

# main() is the pipeline entry point. It runs only when invoked explicitly —
# `PYTHONPATH=src python -m hr_ai_graph_rag` (see __main__.py), `runner.main()`, or in
# Colab. Importing the package does not run it, so the data / parsing / chunking / graph
# layers can be reused and tested without the full GPU + LLM pipeline.
