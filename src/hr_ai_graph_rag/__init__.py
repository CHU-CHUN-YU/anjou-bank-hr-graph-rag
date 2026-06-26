# ============================================================
# 安久銀行 HR AI — Graph RAG Assistant (package)
#
# Traditional-Chinese HR Q&A:hybrid retrieval + knowledge graph +
# LangGraph workflow + deterministic guardrails + local HF LLM。
#
# 本 package 由原單一模組 hr_ai_graph_rag.py 依職責拆分而成。各子模組
# 在此 re-export,故 `import hr_ai_graph_rag as hr` 後沿用 hr.* 介面不變。
# ============================================================

from .config import *      # noqa: F401,F403  環境設定 / 常數
from .utils import *       # noqa: F401,F403  文字 / route 工具
from .artifacts import *   # noqa: F401,F403  離線 artifacts
from .ingestion import *   # noqa: F401,F403  DOCX 解析 / 切塊 / 建知識
from .graph import *       # noqa: F401,F403  生成知識圖
from .retrieval import *   # noqa: F401,F403  混合檢索 / 重排序
from .llm import *         # noqa: F401,F403  本地生成模型
from .workflow import *    # noqa: F401,F403  對話流程
from .evaluation import *  # noqa: F401,F403  評估
from .runner import *      # noqa: F401,F403  進入點
