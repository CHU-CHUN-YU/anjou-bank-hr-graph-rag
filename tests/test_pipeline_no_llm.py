"""No-LLM smoke test for the HR AI Graph-RAG pipeline.

This exercises every layer that does NOT require the GPU / generative-LLM stack
(torch, faiss, sentence-transformers, transformers, langgraph):

    DOCX parsing -> hierarchical chunking -> offline artifact loading ->
    knowledge-graph construction -> concept / risk / query-pattern matching ->
    golden-dataset loading.

It runs on the real data files bundled under ``data/`` and asserts non-trivial
output, so a green run is concrete evidence the core pipeline executes.

Run:  python tests/test_pipeline_no_llm.py
Deps: python-docx, networkx, rank-bm25, pandas, numpy, tqdm  (see requirements-core.txt)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_ai_graph_rag as hr  # noqa: E402

DATA = REPO_ROOT / "data"
POLICY_DOCX = DATA / "policies" / "安久銀行員工工作與福利規章辦法_模擬版.docx"
GOLDEN_JSON = DATA / "golden" / "anjou_bank_hr_ai_golden_dataset_50.json"
ARTIFACT_DIR = DATA / "hr_offline_artifacts"

_checks = 0


def check(cond, msg):
    global _checks
    if not cond:
        raise AssertionError("FAIL: " + msg)
    _checks += 1
    print("  ok -", msg)


def test_utility_helpers():
    print("[1] utility helpers")
    check(hr.detect_category("我想請特休假") == "leave", "detect_category leave")
    check(hr.detect_category("加班費怎麼算") == "overtime", "detect_category overtime")
    check(hr.article_id_from_no("law", "第 38 條") == "law_38", "article_id_from_no law")
    check(hr.normalize_article_no("第38條") == "第 38 條", "normalize_article_no")


def test_offline_artifacts():
    print("[2] offline artifacts loader")
    arts = hr.OfflineArtifacts(str(ARTIFACT_DIR)).load()
    check(len(arts.concept_nodes) > 0, f"concept_nodes loaded ({len(arts.concept_nodes)})")
    check(len(arts.risk_policies) > 0, f"risk_policies loaded ({len(arts.risk_policies)})")
    check(len(arts.query_patterns) > 0, f"query_patterns loaded ({len(arts.query_patterns)})")
    check(len(arts.rewrite_rules) > 0, f"rewrite_rules loaded ({len(arts.rewrite_rules)})")
    check(len(arts.relation_types) > 0, f"relation_types loaded ({len(arts.relation_types)})")
    check(len(arts.graph_edge_candidates) > 0, f"graph edge candidates ({len(arts.graph_edge_candidates)})")
    return arts


def test_chunking():
    print("[3] DOCX parse + hierarchical chunking")
    builder = hr.HRKnowledgeBuilder(hr.ChunkConfig(version="test", effective_date="2026-06-26"))
    # Use built-in sample for the labor law (external source not bundled), real DOCX for policy.
    law_path = str(REPO_ROOT / "hr_ai_graph_rag_outputs_test_law.docx")
    hr.create_sample_labor_law_docx(law_path)
    articles, chunks = builder.build_chunks(
        labor_law_docx_path=law_path,
        internal_policy_docx_path=str(POLICY_DOCX),
        golden_df=None,
    )
    Path(law_path).unlink(missing_ok=True)
    law_arts = [a for a in articles if a["source_type"] == "law"]
    pol_arts = [a for a in articles if a["source_type"] == "internal_policy"]
    check(len(law_arts) > 0, f"parsed law articles ({len(law_arts)})")
    check(len(pol_arts) > 0, f"parsed internal-policy articles ({len(pol_arts)})")
    types = {c["chunk_type"] for c in chunks}
    check({"document", "article"}.issubset(types), f"chunk types present ({sorted(types)})")
    check(all(c.get("global_chunk_id") for c in chunks), "every chunk has global_chunk_id")
    return articles, chunks


def test_knowledge_graph(articles, chunks, arts):
    print("[4] knowledge graph construction")
    kg = hr.HRKnowledgeGraph(articles, chunks, artifacts=arts)
    n, e = kg.G.number_of_nodes(), kg.G.number_of_edges()
    check(n > 0, f"graph has nodes ({n})")
    check(e > 0, f"graph has edges ({e})")
    # Graph expansion should run deterministically without the embedding/LLM layers.
    seeds = [a["article_id"] for a in articles[:3]]
    ctx = kg.expand(seeds, "內規與法規的差異為何", hops=1, max_nodes=10)
    check(isinstance(ctx, dict) and "nodes" in ctx, "expand() returns context dict")
    return kg


def test_matchers(arts):
    print("[5] concept / risk / query-pattern matching")
    g = hr.HRAssistantGraph.__new__(hr.HRAssistantGraph)  # bypass __init__ (no retriever/kg needed)
    g.artifacts = arts
    concepts = g._match_concepts("特別休假可以遞延嗎")
    check(len(concepts) > 0, f"concept match for 特別休假 ({len(concepts)})")
    risks = g._match_risk_policies("公司這樣是否合法，我可以提告嗎")
    check(len(risks) > 0, f"risk policy match for legal-judgment query ({len(risks)})")
    understanding = g._heuristic_understanding("我想請假")
    check(understanding["answer_policy"] == "clarify", "ambiguous query routes to clarify")


def test_golden_dataset():
    print("[6] golden dataset loading")
    df = hr.load_golden_dataset(str(GOLDEN_JSON))
    check(len(df) == 50, f"golden dataset has 50 rows ({len(df)})")
    for col in ["question", "expected_route", "expected_category"]:
        check(col in df.columns, f"golden column present: {col}")


def main():
    print("=" * 60)
    print("HR AI Graph-RAG — no-LLM pipeline smoke test")
    print("=" * 60)
    test_utility_helpers()
    arts = test_offline_artifacts()
    articles, chunks = test_chunking()
    test_knowledge_graph(articles, chunks, arts)
    test_matchers(arts)
    test_golden_dataset()
    print("=" * 60)
    print(f"ALL PASSED — {_checks} checks")
    print("=" * 60)


if __name__ == "__main__":
    main()
