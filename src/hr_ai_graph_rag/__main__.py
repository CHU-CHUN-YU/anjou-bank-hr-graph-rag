# ============================================================
# 套件進入點:python -m hr_ai_graph_rag
#
# 對應原本 `python src/hr_ai_graph_rag.py` 的執行入口。
# ============================================================

from .runner import main

if __name__ == "__main__":
    # 回傳 (assistant, articles, chunks, kg, retriever, golden_df, offline_artifacts)
    main()
