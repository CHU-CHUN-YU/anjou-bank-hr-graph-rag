# 安久銀行 HR AI — Metadata 範例與欄位說明

> 本文整理這份 repo 內各種資料物件的 **metadata 欄位與實際範例**(全部用 `data/` 內真實資料產生)。
> 對應程式碼:chunk 見 `src/hr_ai_graph_rag/ingestion.py`、知識圖 node/edge 見 `src/hr_ai_graph_rag/graph.py`。

涵蓋三類:

1. [Chunk metadata](#1-chunk-metadata)(document / article / semantic / faq)
2. [知識圖 Node metadata](#2-知識圖-node-metadata)(concept / law_article / internal_policy_article)
3. [知識圖 Edge metadata](#3-知識圖-edge-metadata)

---

## 1. Chunk metadata

切塊在 `HRKnowledgeBuilder.build_chunks` 產生(見 `docs/WORKFLOW.md` 附錄 C)。每個 chunk 是一個 dict,**所有 chunk_type 共用同一組 18 個欄位**。實際資料的型別分佈:`document × 2`、`article × 148`、`semantic × 96`(FAQ 預設關閉)。

### 1.1 欄位總表

| 欄位 | 意義 | 範例值 |
|---|---|---|
| `chunk_id` | 該型別內的序號 id | `article_0107` |
| `global_chunk_id` | 全域序號(跨型別唯一) | `G00109` |
| `chunk_type` | 粒度:`document` / `article` / `semantic` / `faq` | `article` |
| `source_type` | 來源:`law` / `internal_policy` / `golden_dataset_faq_experiment` | `internal_policy` |
| `document_name` | 來源檔名 | `安久銀行…_模擬版.docx` |
| `article_id` | 正規化條文 id(semantic 為 `母條::s{n}`) | `policy_5` / `policy_5::s1` |
| `article_no` | 條號(document 層為 `DOCUMENT`、faq 為題號) | `第 5 條` |
| `chapter` | 章別 | `第二章　工作時間與出勤制度` |
| `title` | 條文標題 | `正常工作時間` |
| `category` | 關鍵字判定的類別(`detect_category`,啟發式) | `leave` |
| `priority` | 法規=1、內規=2(檢索排序加權用) | `2` |
| `version` | 版本 | `PoC-v1` |
| `effective_date` | 生效日 | `2026-06-23` |
| `content` | chunk 內文 | 條文全文 |
| `parent_id` | 上層 chunk(article→document、semantic→article) | `policy_5` |
| `related_articles` | 關聯條文 id(內規→勞基法) | `["law_30","law_5"]` |
| `keywords` | 抽取的關鍵字(供 BM25 / keyword bonus) | `["工作時間", …]` |
| `embedding_text` | **真正被向量化 / BM25 的文本**(把結構欄位串成一段) | 見 §1.6 |

> FAQ chunk 另有兩個專屬欄位:`question`、`answer`(見 §1.5)。

### 1.2 `document` chunk(文件層,整份摘要)

```json
{
  "chunk_id": "doc_0002",
  "global_chunk_id": "G00002",
  "chunk_type": "document",
  "source_type": "internal_policy",
  "document_name": "安久銀行員工工作與福利規章辦法_模擬版.docx",
  "article_id": "document::安久銀行員工工作與福利規章辦法_模擬版.docx",
  "article_no": "DOCUMENT",
  "chapter": "Document-level",
  "title": "安久銀行員工工作與福利規章辦法_模擬版.docx 文件摘要",
  "category": "attendance,general,leave,overtime,privacy_sensitive,salary,welfare",
  "priority": 2,
  "version": "PoC-v1",
  "effective_date": "2026-06-23",
  "content": "第 1 條 目的：…\n第 2 條 適用範圍：…  （前 20 條，每條取前 160 字）",
  "parent_id": null,
  "related_articles": [],
  "keywords": ["請假", "特別休假", "病假", "加班", "薪資", "福利", "申訴", …]
}
```

特徵:`article_no="DOCUMENT"`、`parent_id=null`、`category` 為全文件類別的**聯集**、`priority` 取全文件**最大值**。

### 1.3 `article` chunk(條文層,1:1 完整條文)— 最常命中

```json
{
  "chunk_id": "article_0107",
  "global_chunk_id": "G00109",
  "chunk_type": "article",
  "source_type": "internal_policy",
  "document_name": "安久銀行員工工作與福利規章辦法_模擬版.docx",
  "article_id": "policy_1",
  "article_no": "第 1 條",
  "chapter": "第一章　總則",
  "title": "目的",
  "category": "overtime",
  "priority": 2,
  "version": "PoC-v1",
  "effective_date": "2026-06-23",
  "content": "第 1 條 目的\n為建立本行員工工作時間、請假、加班、補休及福利制度…特訂定本辦法。",
  "parent_id": "document::安久銀行員工工作與福利規章辦法_模擬版.docx",
  "related_articles": ["law_1"],
  "keywords": ["請假", "補休", "加班", "工作時間", "福利", "勞動基準法", …]
}
```

特徵:`article_id` 為正規化條文 id(`policy_1`)、`parent_id` 指回文件層、`related_articles` 為內規條文中抓到的勞基法條對應 id。

### 1.4 `semantic` chunk(語意子句層,約 120 字一組)

```json
{
  "chunk_id": "semantic_0081",
  "global_chunk_id": "G00231",
  "chunk_type": "semantic",
  "source_type": "internal_policy",
  "document_name": "安久銀行員工工作與福利規章辦法_模擬版.docx",
  "article_id": "policy_5::s1",
  "article_no": "第 5 條",
  "chapter": "第二章　工作時間與出勤制度",
  "title": "正常工作時間",
  "category": "overtime",
  "priority": 2,
  "version": "PoC-v1",
  "effective_date": "2026-06-23",
  "content": "第 5 條 正常工作時間\n本行一般員工之正常工作時間為每日 8 小時，每週 40 小時。…",
  "parent_id": "policy_5",
  "related_articles": ["law_30", "law_5"],
  "keywords": ["工作時間", "正常工作時間", "每週", "40", "中午休息", …]
}
```

特徵:`article_id` 為 `母條::s{n}`(如 `policy_5::s1`)、`parent_id` 指回母條(`policy_5`)。短條文(切不出第二組且 < 220 字)不產生 semantic chunk。

### 1.5 `faq` chunk(預設關閉,`USE_GOLDEN_AS_FAQ_CHUNKS=true` 才產生)

由 golden 題目衍生,多兩個欄位 `question` / `answer`,`source_type` 標為實驗用以利區隔:

```json
{
  "chunk_id": "faq_0001",
  "global_chunk_id": "G00247",
  "chunk_type": "faq",
  "source_type": "golden_dataset_faq_experiment",
  "document_name": "Golden Dataset derived FAQ chunks - experimental",
  "article_id": "faq::golden::G001",
  "article_no": "G001",
  "chapter": "FAQ Chunk",
  "title": "勞基法規定正常工時一天最多幾小時？",
  "category": "working_hours",
  "priority": 1,
  "version": "PoC-v1",
  "effective_date": "2026-06-23",
  "content": "FAQ 問題：勞基法規定正常工時一天最多幾小時？\nFAQ 回答重點：每日正常工作時間不得超過8小時；每週正常工作時間不得超過40小時",
  "question": "勞基法規定正常工時一天最多幾小時？",
  "answer": "每日正常工作時間不得超過8小時；每週正常工作時間不得超過40小時",
  "parent_id": null,
  "related_articles": [],
  "keywords": ["工時", "工作時間", "FAQ", "每日正常工作時間不得超過8小時", …]
}
```

> ⚠️ 正式評估請維持 `USE_GOLDEN_AS_FAQ_CHUNKS=false`,否則等於把答案放進知識庫,指標會虛高。

### 1.6 `embedding_text`(真正被嵌入 / 檢索的文本)

`content` 不直接拿去嵌入;系統把結構化欄位串成下面這段,讓向量同時帶「型別 / 來源 / 類別」訊號:

```
Chunk Type: article
Source Type: internal_policy
Document: 安久銀行員工工作與福利規章辦法_模擬版.docx
Article: 第 1 條
Title: 目的
Category: overtime
Priority: 2
Content: 第 1 條 目的\n為建立本行員工工作時間…特訂定本辦法。
Keywords: 請假, 補休, 加班, 工作時間, 福利, 勞動基準法, …
```

### 1.7 三種型別的關鍵差異

| | `article_no` | `article_id` | `parent_id` |
|---|---|---|---|
| **document** | `DOCUMENT` | `document::<檔名>` | `null` |
| **article** | `第 5 條` | `policy_5` | `document::<檔名>` |
| **semantic** | `第 5 條` | `policy_5::s1` | `policy_5` |
| **faq** | `G001`(題號) | `faq::golden::G001` | `null` |

> 小細節:`category` 由關鍵字 `detect_category` 推得,所以像「第 1 條 目的」因含「加班 / 補休」字樣被標成 `overtime`——它是啟發式分類,不一定等於人工歸類。

---

## 2. 知識圖 Node metadata

知識圖在 `HRKnowledgeGraph._build_graph` 建立(見 `docs/WORKFLOW.md` 附錄 B)。實際資料約 **172 節點**,三種 `node_type`:`concept × 25`、`law_article × 105`、`internal_policy_article × 42`。

### 2.1 `concept` 節點(來自 `concept_nodes.json` artifact)

```json
{
  "node_id": "concept_special_leave",
  "node_type": "concept",
  "label": "特別休假",
  "category": "leave",
  "risk_level": "low",
  "default_answer_policy": "answer",
  "content": "依年資給予特別休假，安久銀行內規提供優於勞基法最低標準之天數。",
  "aliases": "特休｜年假｜特別假｜休年假",
  "graph_expansion_priority": "high"
}
```

| 欄位 | 意義 |
|---|---|
| `node_type` | `concept` |
| `label` | 概念顯示名稱 |
| `category` | 概念類別 |
| `risk_level` | 風險等級(`low`/`medium`/`high`) |
| `default_answer_policy` | 預設回答政策(`answer`/`with_disclaimer`/`clarify`/`escalate`) |
| `content` | 概念說明 |
| `aliases` | 別名(以 `｜` 分隔,供查詢比對) |
| `graph_expansion_priority` | 圖擴展優先度(`high`/`medium`/`low`) |

### 2.2 `law_article` 節點(勞基法條文)

```json
{
  "node_id": "law_38",
  "node_type": "law_article",
  "label": "第 38 條 ",
  "source_type": "law",
  "article_no": "第 38 條",
  "category": "leave",
  "priority": 1,
  "content": "第 38 條\n勞工在同一雇主或事業單位，繼續工作滿一定期間者，應依下列規定給予特別休假：…"
}
```

### 2.3 `internal_policy_article` 節點(內規條文)

```json
{
  "node_id": "policy_11",
  "node_type": "internal_policy_article",
  "label": "第 11 條 特別休假",
  "source_type": "internal_policy",
  "article_no": "第 11 條",
  "category": "leave",
  "priority": 2,
  "content": "第 11 條 特別休假\n員工於本行連續服務滿一定期間者，得依下列標準享有特別休假：…本條對應外規關係：參照《勞動基準法》第 38 條…"
}
```

| 欄位(law / policy 條文節點) | 意義 |
|---|---|
| `node_type` | `law_article` / `internal_policy_article` |
| `label` | `條號 + 標題` |
| `source_type` | `law` / `internal_policy` |
| `article_no` | 條號 |
| `category` | 類別 |
| `priority` | 法規=1、內規=2 |
| `content` | 條文全文 |

---

## 3. 知識圖 Edge metadata

邊由「概念↔條文對應」「內規條文對法規的引用」「人工核准的候選邊」三條管道產生。實際資料約 **239 條邊**。

### 3.1 簡單邊(概念↔條文、文件內參照)

`has_rule` / `related_to` / `parent_of` / `child_of` / `refers_to` 通常只有 `relation` 一個屬性:

```json
{ "source": "concept_special_leave", "target": "law_38", "relation": "has_rule" }
```

### 3.2 候選邊(`graph_relation_candidates.json`,帶治理稽核欄位)

`overrides` / `supplements` 等候選邊**只載入 `review_status="approved"`** 的,並保留稽核欄位:

```json
{
  "source": "policy_3",
  "target": "concept_internal_policy_priority",
  "relation": "supplements",
  "evidence": "本辦法如提供優於法令最低標準之工作條件或福利措施，員工得依本辦法適用較有利之規定。",
  "confidence": 0.96,
  "review_status": "approved",
  "human_review_required": false
}
```

| 欄位 | 意義 |
|---|---|
| `relation` | 關係型別(見下表) |
| `evidence` | 支持這條關係的原文證據 |
| `confidence` | 信心分數(0–1) |
| `review_status` | 審核狀態(預設只載 `approved`;`pending` 需 `LOAD_PENDING_GRAPH_EDGES=true`) |
| `human_review_required` | 是否仍需人工複核 |

### 3.3 relation 型別語意

| relation | 意義 |
|---|---|
| `overrides` | 內規優於法規最低標準(最有價值的差異關係) |
| `supplements` | 內規補充法規 |
| `has_rule` / `related_to` | 概念 ↔ 條文的雙向關聯 |
| `refers_to` | 條文參照另一條文 |
| `parent_of` / `child_of` | 概念階層 |

---

## 4. 相關文件

- 切塊原理與更多範例:`docs/WORKFLOW.md` 附錄 C
- 知識圖建置與走訪原理:`docs/WORKFLOW.md` 附錄 B
- 知識圖視覺化範例:`docs/knowledge_graph_example.md`
