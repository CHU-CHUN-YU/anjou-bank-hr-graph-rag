"""Integration test: hybrid retrieval + cross-encoder rerank stage.

Requires the embeddings stack (sentence-transformers, faiss). It downloads the
configured embedding + reranker models, so it is slower than the no-LLM smoke test.

By default it uses the repo's configured models (BGE-M3 + bge-reranker-v2-m3). For a
fast CI check, override with small stand-ins:

    EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    python tests/test_retrieval_rerank.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_ai_graph_rag as hr  # noqa: E402

POLICY_DOCX = REPO_ROOT / "data" / "policies" / "安久銀行員工工作與福利規章辦法_模擬版.docx"


def main():
    print("Embedding model:", hr.EMBEDDING_MODEL_NAME)
    print("Reranker model :", hr.RERANKER_MODEL_NAME, "| enabled:", hr.USE_RERANKER)

    builder = hr.HRKnowledgeBuilder(hr.ChunkConfig(version="test", effective_date="2026-06-26"))
    law_path = str(REPO_ROOT / "_test_law.docx")
    hr.create_sample_labor_law_docx(law_path)
    _, chunks = builder.build_chunks(str(law_path), str(POLICY_DOCX), golden_df=None)
    Path(law_path).unlink(missing_ok=True)
    print(f"chunks: {len(chunks)}")

    retriever = hr.HybridRetriever(chunks)
    assert retriever.reranker is not None, "reranker failed to load (USE_RERANKER=true expected a model)"

    query = "特別休假可以遞延到隔年嗎？"
    results = retriever.search(query, category="leave", top_k=5)
    assert results, "no results returned"
    assert any("rerank_score" in r for r in results), "rerank_score missing — rerank stage did not run"

    print(f"\nQuery: {query}\nTop {len(results)} results (after rerank):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.get('source_type')}] {r.get('article_no')} {r.get('title','')}"
              f"  final={r.get('final_score')}  rerank={r.get('rerank_score')}  pre={r.get('pre_rerank_score')}")

    print("\nRETRIEVAL + RERANK OK")


if __name__ == "__main__":
    main()
