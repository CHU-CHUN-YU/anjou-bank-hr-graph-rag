# 安久銀行 HR AI Graph RAG — Runtime Local LLM Pattern + Hybrid Rewrite 版

這份 package 是依照成熟 workflow 調整後的 Colab 版本：

- 勞基法 DOCX、模擬銀行內規 DOCX、Golden Dataset JSON 由外部上傳。
- Offline LLM artifacts 以 JSON 接入：`concept_nodes`, `risk_policy`, `query_patterns`, `rewrite_rules`, `relation_schema`, `graph_relation_candidates`。
- Runtime local LLM 使用 Colab GPU HuggingFace model，負責：category / intent / concept matching / query pattern / ambiguity / missing slots / normalized query / candidate risk signal / optional rewrite terms。
- `query_patterns`：offline JSON 提供 schema + examples + fallback clarification；runtime local LLM 判斷 matched pattern 與 missing slots。
- `rewrite_rules`：offline JSON 提供 mandatory terms 與法條/內規 hints；runtime local LLM 只補 optional semantic terms，不可新增法條號或內規條號。
- Final route 仍由 deterministic guardrails 決定：risk_policy high-risk override、missing_slots clarify、retrieval confidence threshold、medium risk disclaimer。

## Colab 使用方式

1. Runtime 選 T4 GPU。
2. 上傳或指定：
   - 勞動基準法 DOCX
   - 安久銀行模擬內規 DOCX
   - Golden Dataset JSON
3. 解壓縮 package，或把 `hr_offline_artifacts/` 放在 `/content/hr_offline_artifacts`。
4. 執行 notebook 或：

```python
%run /content/colab_hr_ai_graph_rag_hf_local_runtime_pattern_rewrite.py
```

可選環境變數：

```python
import os
os.environ["LABOR_LAW_DOCX_PATH"] = "/content/參考資料_勞動基準法.docx"
os.environ["INTERNAL_POLICY_DOCX_PATH"] = "/content/安久銀行員工工作與福利規章辦法_模擬版.docx"
os.environ["GOLDEN_DATASET_JSON_PATH"] = "/content/anjou_bank_hr_ai_golden_dataset_50.json"
os.environ["OFFLINE_ARTIFACT_DIR"] = "/content/hr_offline_artifacts"
os.environ["USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING"] = "true"
os.environ["LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS"] = "5"
```

## 核心分工

| 模組 | 做法 |
|---|---|
| Concept taxonomy | offline LLM artifact + 人工審核 |
| Risk policy | offline LLM artifact + HR/法遵審核，runtime deterministic override |
| Query patterns | local LLM runtime 判斷 + offline fallback schema |
| Rewrite rules | offline mandatory terms + local LLM optional semantic terms |
| Graph relations | offline candidates + approved edge only |
| Final route | deterministic guardrails |
| Final answer | local LLM grounded generation |

## 重要安全原則

Local LLM 是 signal provider，不是 final decision maker。它可以理解問題、補語意詞、產生回答，但不負責最終是否轉人工、法律判定或高風險 override。
