# 安久銀行 HR AI — 系統與 Workflow 詳細說明

> 本文用文字詳細說明這份 repo 在做什麼、資料怎麼流動、每個階段的輸入與輸出。
> 對應程式碼為 `src/hr_ai_graph_rag/` 套件。閱讀時可搭配執行時印出的 `[STAGE] …` 日誌對照。

---

## 1. 這份 repo 在做什麼

這是一個**繁體中文的 HR 問答助理(虛構的「安久銀行」)**概念驗證(PoC)。使用者問一個 HR / 勞動法規問題(例如「特休可以遞延嗎?」),系統會:

1. 從**台灣勞動基準法**(外部法規)與**銀行內部規章**(內規)中,檢索相關條文;
2. 用**知識圖**補上條文之間的關聯(尤其「內規 vs 法規」的差異);
3. 經過**確定性的護欄(guardrails)**決定要「直接回答 / 加聲明回答 / 反問澄清 / 轉人工」;
4. 若要回答,才用**本地 LLM**(預設 Qwen2.5-1.5B-Instruct)生成**有引用來源**的答案;
5. 最後對著一份 **50 題的 golden 評估集**量測各項指標。

它整合了三種技術:**混合檢索(Hybrid Retrieval)+ 知識圖(Knowledge Graph)+ LangGraph 工作流(deterministic guardrails)**。生成模型是本地 HuggingFace LLM,**不需要 OpenAI API key**,設計給 Colab T4 GPU 跑。

---

## 2. 核心設計原則

> **LLM 只是「訊號提供者」,永遠不是最終決策者。路由(回答 / 聲明 / 澄清 / 轉人工)由確定性規則決定;風險只能被「升高」,不能被「降低」。**

這在 HR / 法規這種高敏感場景非常重要:

- LLM 可以**建議**分類、風險、偏好關係,但**不能單獨決定**最終要不要回答。
- 任何高風險訊號(來自風險政策或 LLM)都能把路由升級成「加聲明」或「轉人工」,但**沒有任何邏輯能把高風險降級成直接回答**。
- 圖的走訪只走**人工核准過的邊**;LLM 不能改變圖。
- 生成答案時硬性要求**只能根據檢索到的 context、必須附引用、不得自行編造、不做法律判定**。

---

## 3. 系統全貌

整個系統分兩大階段:

```
【階段 A:離線建構】 整個程式啟動時做一次
  輸入資料 → 切塊 → 建知識圖 → 建檢索器 → 組裝 assistant

【階段 B:執行時 workflow】 每一題問答各跑一次
  query → 查詢理解 → 檢索(混合+rerank)+圖擴展 → 護欄路由 → 生成/澄清/轉人工 → 忠實度檢查
```

### Package 模組對照

| 模組 | 角色 |
|---|---|
| `config.py` | 環境變數、模型名稱、全域常數、`stage_log()` 日誌工具 |
| `utils.py` | 零依賴工具:中文正規化、tokenize、category 偵測、route/policy 轉換、條號擷取 |
| `artifacts.py` | 載入 9 個離線知識 JSON(`OfflineArtifacts`) |
| `ingestion.py` | DOCX 解析、階層式切塊、`HRKnowledgeBuilder` |
| `graph.py` | `HRKnowledgeGraph`:建圖 + 執行時 `expand()` 擴展 |
| `retrieval.py` | `HybridRetriever`:向量 + BM25 + metadata 混合 + cross-encoder rerank |
| `llm.py` | `LocalHFLLM` + `call_llm_text` / `call_llm_json` |
| `workflow.py` | `HRAssistantGraph`:LangGraph 工作流(階段 B 的核心) |
| `evaluation.py` | golden 評估指標 + 使用者回饋紀錄 |
| `runner.py` | 串起整條 pipeline + `main()` 進入點 |

`__init__.py` 把所有公開符號 re-export,所以 `import hr_ai_graph_rag as hr` 後可直接用 `hr.HRAssistantGraph`、`hr.main()` 等扁平介面。

---

## 4. 輸入資料

| 資料 | 來源 | 預設 | 說明 |
|---|---|---|---|
| 勞動基準法 DOCX | 外部法律 | 放在 `data/policies/`(檔名含 勞動基準法/勞基法/labor)→ **自動探索**;無則用內建 sample | 沒提供時用內建關鍵條文 sample 讓流程能跑完;可用 `LABOR_LAW_DOCX_PATH` 覆蓋 |
| 安久銀行內規 DOCX | 內附 `data/policies/` | 自動探索 | 虛構的銀行內部規章 |
| Golden Dataset JSON | 內附 `data/golden/` | 自動探索 | 50 題評估集 |
| 離線 artifacts(9 個 JSON) | 內附 `data/hr_offline_artifacts/` | 自動探索 | 概念節點、風險政策、查詢樣式、改寫規則、關係 schema、圖邊候選、角色對應、本地 LLM 使用政策、清單 manifest |

**離線 artifacts 是這套系統的關鍵治理層**:把「概念分類、風險政策、可信圖邊、改寫詞」這些需要人工把關的東西,事先整理成 JSON,讓執行時的判斷有可控依據,而不是全靠 LLM 即興發揮。

---

## 5. 階段 A:離線建構(`runner.main()`)

依序執行(對應 `[STAGE]` 日誌):

```
prepare_input_files       → 解析/定位三個輸入檔(缺勞基法則用 sample)
load_offline_artifacts    → 載入 9 個 JSON,並把 concept_nodes 併入分類關鍵字
load_golden_dataset       → 載入 50 題
HRKnowledgeBuilder.build_chunks → 產生 articles + chunks
HRKnowledgeGraph(...)      → 建知識圖(一次)
HybridRetriever(chunks)   → 建向量索引 + BM25(+ 載入 reranker)
HRAssistantGraph(...)     → 組裝 LangGraph 工作流
→ 跑 demo 9 題 → 評估 50 題 → 存檔打包
```

### A1. 輸入準備(`prepare_input_files`)
- 先用環境變數 / 內附路徑解析三個檔;
- 內規 DOCX、golden JSON、artifacts 都會**自動探索**(在 Colab clone 的 repo 也適用);
- 勞基法 DOCX:若 `data/policies/` 有(檔名含 勞動基準法/勞基法/labor)會自動探索,**沒有才產生內建 sample 條文**;
- 只有「真的缺檔 **且** 在互動式 notebook kernel」時才會跳出上傳介面(用 `python -m` 子行程跑時不會卡住)。

### A2. 離線 artifacts(`load_offline_artifacts`)
- 用 `OfflineArtifacts` 載入資料夾 / ZIP 內的 9 個 JSON;
- 把 `concept_nodes` 當成「外部化的分類詞表」,**合併進** `CATEGORY_KEYWORDS`,讓後續分類更準。

### A3. 切塊(`HRKnowledgeBuilder.build_chunks`,`ingestion.py`)
先解析條文,再做 **Policy-aware 階層式切塊**,產生 4 種 chunk:

| chunk_type | 來源 | 內容 | priority |
|---|---|---|---|
| `document` | 整份文件 | 同一文件全部條文彙整 | 取該文件條文最高 priority |
| `article` | 單一條文 | 一條條文 | 法規=1、內規=2 |
| `semantic` | 條文內句子 | 把長條文切成語意子句(帶 `parent_id`) | 同母條文 |
| `faq` | golden 題 | 只有 `USE_GOLDEN_AS_FAQ_CHUNKS=true` 才產生(預設 false,避免評估作弊) | 1 |

- **內規 priority=2 > 法規 priority=1**:呼應「內規優於法規最低標準」的精神,排序時內規會被加權往前。
- 每個 chunk 最後組一段 `embedding_text`(含 type/source/條號/標題/類別/內容/關鍵字),作為**被檢索的文本**。
- 產出:`articles`(條文清單)+ `chunks`(4 種混合的可檢索單元)。

### A4. 建知識圖(`HRKnowledgeGraph._build_graph`,`graph.py`)
建立一張 networkx 有向圖,**整個程式只建一次**,分 4 步(對應 `[STAGE] graph:*` 日誌):

1. **概念節點**(`graph:concept_nodes`):從 `concept_nodes` artifact 建概念節點(label、category、risk_level、aliases…);有 `parent_concept_id` 就加 `parent_of`/`child_of` 邊。無 artifact 時用內建骨架。
2. **條文節點**(`graph:article_nodes`):把每條法規/內規條文加成節點(`law_article` / `internal_policy_article`),並加上文件內解析到的 `refers_to` 與 `graph_edges` 邊。
3. **概念↔條文邊**(`graph:concept_article_edges`):依 artifact 把概念連到對應條文(`has_rule` / `related_to`)。
4. **artifact 候選邊**(`graph:artifact_edges`):載入 `graph_relation_candidates` 裡**已核准(approved)**的邊(`overrides`/`supplements`…);未核准的預設不載入(除非 `LOAD_PENDING_GRAPH_EDGES=true`)。

最後印 `graph:built — nodes=… edges=…` 與各 relation 的邊數。建好後整張圖固定不動。

### A5. 建檢索器(`HybridRetriever.__init__`,`retrieval.py`)
- 用 `sentence-transformers` 載入 **bge-m3**,把所有 chunk 的 `embedding_text` 編成正規化向量,建 **FAISS `IndexFlatIP`**(內積=cosine)索引;
- 同時用 `tokenize_zh` 斷詞建 **BM25**;
- (選配)載入 cross-encoder reranker **bge-reranker-v2-m3**,載入失敗會 graceful 降級成純混合檢索。

---

## 6. 階段 B:執行時 workflow(LangGraph,`HRAssistantGraph`)

這是系統核心。`assistant.ask(question)` 會跑一條 LangGraph,**每題一次**。

### 流程圖

```
                       START
                         │
                         ▼
            ① query_understanding         查詢理解(heuristic + 選配 LLM)
                         │
                         ▼
            ② retrieval_orchestrator      混合檢索 → rerank → graph 擴展
                         │
                         ▼
            ③ guardrails (確定性路由)      決定 route
                         │
         ┌───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼
     answer          disclaimer       escalate         clarify
         │               │               │               │
         └───────┬───────┘               │               │
                 ▼                        ▼               ▼
        ④ generate_answer          ⑥ escalate      ⑤ clarify
           (本地 LLM,有引用)         (轉人工模板)     (反問模板)
                 │                        │               │
                 ▼                        └──────┬────────┘
        ⑦ faithfulness_check                     │
                 │                                │
                 ▼                                ▼
                              END
```

LangGraph 的邊(`_build_graph`):START → query_understanding → retrieval_orchestrator → guardrails;guardrails 用**條件邊**依 `route` 分流到 generate_answer / escalate / clarify;generate_answer → faithfulness_check → END;clarify、escalate 直接 → END。

### state(在節點間累積的狀態 `HRState`)
每個節點讀取上游放進 state 的欄位,計算後再寫回新的欄位。重要欄位:
`question, category, intent, risk_level, answer_policy, matched_concepts, missing_slots, risk_matches, llm_risk_signal, preferred_relations, rewritten_query, retrieved_chunks, graph_context, confidence, route, answer, citations, faithfulness_score, debug`。

---

### 節點 ① query_understanding(查詢理解)

**目的**:把自然語言問題轉成結構化訊號。先跑 heuristic,再(選配)用 LLM 補強。

**Step 1 — heuristic(`_heuristic_understanding`)**:純規則,不需 GPU。
- `detect_category` 用關鍵字判 category;再用 `_match_concepts` 命中的概念修正 category;
- 依 category 給 intent(工時與加班 / 請假與休假 / 薪資與工資 …);
- `_match_risk_policies` 比對風險政策、`_match_query_patterns` 比對查詢樣式;
- 算出**初步 answer_policy**(這只是「建議」,最終由 guardrails 定):
  - 模糊(命中 ambiguous 樣式、或問題 ≤ 5 字、或在預設模糊清單)→ `clarify`
  - 命中高風險 / 概念高風險 / 含 `RISK_KEYWORDS`(違法、申訴、提告…)→ `escalate`
  - 含「是否合法、違法嗎、可以告、申訴、主管不給、被逼」→ `escalate`
  - 概念政策含 with_disclaimer、或含「我這種情況、如果、主管、個案、薪資明細」→ `with_disclaimer`
  - 其餘 → `direct`
- 依 policy 給 `risk_level`(高/中/低);
- 產生 **rewritten_query**(query expansion):把 offline 改寫規則的必備詞 + heuristic 補充詞混進原問題,提升檢索召回;
- 收集 `missing_slots`(查詢樣式指出缺哪些資訊,如假別/期間/原因)。

**Step 2 — 本地 LLM 補強(`_llm_runtime_understanding`,選配)**:`USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING=true` 時。
- 把 concept taxonomy、query pattern schema、risk policy 當「受控選項」塞進 prompt(見 §8 Prompt ①),呼叫 `call_llm_json` 要 LLM **只輸出 JSON**;
- LLM 給的 category/intent/概念/缺漏槽位/候選風險會**補強**結果,並產生 `llm_risk_signal`、`preferred_relations`(圖擴展偏好的關係);
- 但**風險只升不降**:LLM 不能移除既有的高風險命中。

**輸出**:`category, intent, risk_level, answer_policy, matched_concepts, missing_slots, risk_matches, llm_risk_signal, preferred_relations, rewritten_query`。
**日誌**:`[STAGE] query_understanding — category=… intent=… risk=… concepts=… risk_matches=… missing_slots=…`。

---

### 節點 ② retrieval_orchestrator(檢索編排)

**目的**:用改寫後的 query 找到最相關的條文,並用圖補上關聯。**這一節依序做三件事:混合檢索 → rerank → graph 擴展。**(詳見 §7)

1. `retriever.search(rewritten_query, category, top_k=8)`:回傳已**混合檢索 + rerank** 後的 top-8 chunks。
2. 取前 5 個 chunk 對應的條文當 **seed**(語意/FAQ chunk 取 `parent_id`,否則取 `article_id`,並含 `related_articles`)。
3. `kg.expand(seed_ids, question, hops=1, max_nodes=14, preferred_relations=…)`:從種子沿**已核准的邊**走 1 hop,補上關聯節點與邊,組成 `graph_context`。

**輸出**:`retrieved_chunks`、`graph_context`。
**日誌**:`[STAGE] retrieval — chunks=8 graph_nodes=X graph_edges=Y use_graph=…`。

---

### 節點 ③ guardrails(確定性護欄與路由)★最關鍵★

**目的**:這裡「LLM 不做主」。先算 `confidence`(由 top chunk 的分數差距決定),再**按固定優先序**決定最終 `route`:

```
1. 命中高風險政策                       → escalate(風險=高)
2. LLM 訊號判高風險                      → escalate(高)
3. answer_policy=clarify 或有 missing_slots → clarify
4. answer_policy=escalate                → escalate(高)
5. confidence < 0.18(檢索沒把握)        → escalate(中)
6. answer_policy=with_disclaimer 或 LLM 判中風險 → disclaimer(中)
7. 其餘                                   → answer(低)
```

> 注意:從上到下,風險訊號只會把路由**往嚴格的方向**推(answer → disclaimer → escalate),**沒有任何分支把高風險降成直接回答**。

**輸出**:`confidence, route, risk_level`。
**日誌**:`[STAGE] guardrails — policy=… confidence=… → route=… risk=…`。

---

### 節點 ④ generate_answer(生成答案,只在 answer / disclaimer 路徑)

**目的**:根據檢索 + 圖 context 生成**有引用的中文答案**。
- 先用 `_make_context` 組 context:取前 6 個 chunk 的內容(`[S1]…[S6]`)+ `graph_context.context`(截斷 2500 字,標為 `[Graph-enhanced Context]`);
- 若 `USE_LLM=false` → 直接用 `_fallback_answer` 模板;
- 否則用結構化 prompt(見 §8 Prompt ②)呼叫 `call_llm_text`(本地 Qwen),要求:優先內規、其次法規當最低標準、必附 `[S#]` 引用、說明內規 vs 法規差異、情境題加聲明、不做法律判定;
- LLM 失敗 → 退回模板答案;
- 用 `_build_citations` 產生引用清單。

**輸出**:`answer, citations`。
**日誌**:`[STAGE] generate_answer — route=… llm=on/off answer_len=… citations=…`。

---

### 節點 ⑤ clarify(反問澄清,不呼叫 LLM)

當 route=clarify:從 query_patterns 取對應 category 的澄清問題(無則用內建三問),組成「我需要再確認一些資訊…」的回覆,列出可能缺少的槽位與要補充的問題。**不經 LLM**,`faithfulness_score=1.0`。
日誌:`[STAGE] clarify — missing_slots=… clarify_questions=…`。

### 節點 ⑥ escalate(轉人工,不呼叫 LLM)

當 route=escalate:組「此問題不適合 AI 直接判定,建議轉 HR/法遵」的回覆,附上判斷類型、風險等級、命中的風險政策、以及前 3 條可能相關依據。**不經 LLM**,`faithfulness_score=1.0`。
日誌:`[STAGE] escalate — risk=… risk_policies=…`。

### 節點 ⑦ faithfulness_check(忠實度檢查,只在生成路徑後)

對生成的答案做啟發式打分:
- 基礎 0.82;答案含 `[S#]` 引用 → +0.10;答案有提到引用條號 → +0.05;
- escalate/clarify 路徑直接拉到 ≥0.95;上限 1.0。

**輸出**:`faithfulness_score`。日誌:`[STAGE] faithfulness_check — score=… has_citation=… mentions_article=…`。

---

## 7. 檢索三步詳解:混合檢索 → rerank → graph

三步是**串聯、有依賴**的:前一步的輸出是下一步的輸入。

### Step 1 — 混合檢索(`HybridRetriever.search`)
**兩路召回**取聯集:
- 向量(bge-m3 / FAISS,cosine)取 `top_k*5`;
- BM25 全量算分後 min-max 正規化,取前段。

**對每個候選算 `final_score`**:
```
final_score = 0.62·vector_score      # 語意相似(cosine)
            + 0.28·bm25_score        # 字面關鍵字
            + keyword_bonus          # 命中查詢關鍵字,每個 +0.02
            + category_bonus         # chunk 類別 == 查詢類別 +0.06
            + priority_bonus         # 0.04 × priority(內規=2 → 加成較高)
            + faq_bonus              # faq chunk +0.05
            + article_bonus          # article chunk +0.03
```
其中 `keyword/category/priority/faq/article` 加成就是「**metadata 加成**」。依 `final_score` 排序。

> 為什麼要混合:**向量檢索懂「語意/同義」**(問「特休」能找到「特別休假」),**BM25 懂「精確字面」**(條號、專名)。兩者互補,召回更全。

### Step 2 — rerank(cross-encoder,接在混合之後,仍在 `search()` 內)
- 取前 `RERANK_CANDIDATES`(預設 20)個候選;
- 用 bge-reranker-v2-m3 把 `[query, content]` 一起餵進模型算相關性(比 bi-encoder 準但慢,所以只精排前段);
- min-max 正規化後**融合**:
```
final_score = 0.7·rerank_norm + 0.3·hybrid_score   # RERANK_WEIGHT=0.7
```
- 重新排序,回傳 `top_k`(預設 8)。

### Step 3 — graph 擴展(`kg.expand`,在 rerank 之後)
- 用 rerank 後的 top chunks 的條文當 **seed**;
- 只在問題含「差、比較、為什麼、內規、法規、公司、優於、補休、依據、哪個、關係」這類字時才啟用(否則略過,避免雜訊);
- 沿**已核准的邊**走 1 hop(雙向 successors+predecessors),最多 14 個節點;若 LLM 給了 `preferred_relations` 就只保留那些關係的邊;
- **收集的資訊**:
  - 每個**節點**:node_id、node_type、label、article_no、source_type、category、risk_level、default_answer_policy、content(前 500 字);
  - 每條**邊**:source、relation、target、evidence;
- 組成文字 `context`(`[Graph Nodes]` / `[Graph Relations]` 兩段),連同 nodes/edges 一起回傳。

**這段 graph context 會被加進 §6 節點④ 的 LLM 生成 prompt 的 `[Graph-enhanced Context]` 區塊**,讓生成時能看到「policy_18 --overrides--> law_38」這種關聯,寫出正確的內規 vs 法規差異。

**順序總結**:先用混合檢索撈候選 → 用 rerank 精排成 top_k → 再用 top_k 的條文當種子做 graph 擴展。rerank 夾在中間(屬檢索收尾),graph 永遠在最後。

---

## 8. Prompt 層次:2 個業務 prompt + 2 層包裝

整套系統**每題最多只呼叫 LLM 兩次**(分類一次、生成一次);clarify / escalate 路徑完全不呼叫 LLM。

### 業務 Prompt ① — Query Understanding(`call_llm_json`)
- **system**:「你是銀行 HR AI 助理的 Runtime Query Understanding 模組…判斷 category/intent/matched_concepts/是否模糊/missing_slots/candidate risk…**只能輸出 valid JSON,不要 markdown,不要解釋。**」
- **user**(動態組成):使用者問題 + concept taxonomy JSON + query pattern schema JSON + risk policy JSON + 要輸出的 JSON schema 範本。
- 特色:把離線知識當「**受控選項**」喂給 LLM,限制它在範圍內分類。

### 業務 Prompt ② — Answer Generation(`call_llm_text`)
- **system**:角色 + 硬規則(只依 context、優先內規其次法規、必附 `[S#]`、說明內規 vs 法規、情境加聲明、不做法律判定)。
- **user**:員工問題 + 上游算出的 Intent/Category/Risk/Route + Retrieval/Graph Context + **固定回答格式**(簡短結論 / 適用條件 / 依據 Citation / 規範差異 / 白話說明 / 注意事項聲明 / 下一步建議)。

### 包裝層 ③ — JSON 強制(`call_llm_json`,`llm.py`)
呼叫 Prompt ① 時,在 system 後追加「請只輸出 valid JSON,不要輸出 markdown」,temperature=0.0;回來用 regex 抽 JSON,抽不到回 default。

### 包裝層 ④ — Chat template 組裝(`_format_messages`,`llm.py`)
送進模型前,把 system+user 套成模型對話格式:先試標準 `[system, user]`;模型不接受 system 角色(如 Gemma)就**合併進 user turn**;沒有 chat template 的模型用純文字 `System/User/Assistant` 格式。

---

## 9. 評估(`evaluation.evaluate_assistant`)

對 golden 50 題逐題跑 `assistant.ask`,計時並比對,彙總成 summary:

| 指標 | 意義 |
|---|---|
| Category Accuracy | 類別判對率 |
| Route Accuracy | 路由判對率(answer/disclaimer/clarify/escalate) |
| Retrieval Hit Rate | 期望引用是否被檢索到 |
| Source Type Hit Rate | 來源型別(法規/內規)是否命中 |
| Citation Present Rate | 是否有引用(clarify 視為通過) |
| Avg Faithfulness Score | 平均忠實度(啟發式,見 §6 節點⑦) |
| RAGAS Context Recall | 標準答案要點是否被檢索到(平均) |
| RAGAS Context Precision | 相關 context 是否排在前面(平均 AP) |
| RAGAS Faithfulness | 答案是否 grounding 在 context(平均) |
| RAGAS Answer Relevancy | 答案是否切題回應問題(平均) |
| Avg Latency Sec | 平均每題耗時 |

> **四個 RAGAS-style 指標**是這版新增的,**不需要 OpenAI key**。**預設 backend 為 `llm`(本地 Qwen 當 judge)**;也可設 `RAGAS_BACKEND=embedding` 改用 bge-m3 相似度 proxy(較快)。完整實作原理見 **附錄 D**。

評估流程具備 **per-question checkpoint(可續跑)**:每題答完即把結果(含 RAGAS 指標)append 進 `OUTPUT_DIR/eval_checkpoint.jsonl` 並 fsync,Colab 50 分鐘斷線最多只損失「正在跑的那一題」;下次執行自動跳過已完成題,最終 summary 仍涵蓋全部 50 題(細節見附錄 D §D6)。

另有 `log_feedback` 紀錄使用者回饋(helpful、正確度、完整度、評論)到 `feedback_log.csv`。

---

## 10. 輸出檔案(`runner.save_outputs`)

一次完整執行會把結果寫到 `OUTPUT_DIR`(`./hr_ai_graph_rag_outputs`,Colab 為 `/content/...`,已 gitignore):

- `articles.json`、`chunks_3layer_faq.json` / `.csv`、`demo_results.json`
- `kg_nodes.csv`、`kg_edges.csv`、`hr_knowledge_graph.gexf`(可用 Gephi 開)
- `golden_dataset.csv`、`evaluation_detail.csv`(含每題 `answer_full` + `citations_json`,及四個 RAGAS 欄位 `ragas_*`)、`evaluation_summary.csv`(含 RAGAS 平均)
- `eval_checkpoint.jsonl`(可續跑的逐題 checkpoint,見附錄 D §D6)
- `evaluation_records.json`(彙總:每題**完整答案 + 完整參考來源** + summary)
- `evaluation_records/<id>.json`(每題各一個 JSON,完整答案 + 完整來源)
- `feedback_log.csv`、`loaded_offline_artifacts.json`、產出說明 `README.md`
- 全部打包成 `hr_ai_graph_rag_outputs.zip`(Colab 會自動下載)

---

## 11. 觀測:`[STAGE]` 分階段日誌

每個階段都會印一行 `[STAGE] <名稱> — <事實>`(必要時附內容預覽 `↳ …`),讓你看到每一步產出什麼。涵蓋:離線的 `input_files / offline_artifacts / golden_dataset / ingestion / graph:* / retriever / demo / evaluation`,以及每題的 `ask / query_understanding / retrieval / guardrails / generate_answer / clarify / escalate / faithfulness_check`。

用 `STAGE_LOG=false` 可完全靜音;`STAGE_LOG_PREVIEW` 調整預覽長度。

---

## 12. 執行方式

```bash
# A. 無 LLM 快速驗證(CPU,只需 requirements-core.txt)
python tests/test_pipeline_no_llm.py        # 期望 ALL PASSED — 24 checks

# B. 完整流程(GPU / Colab,需 requirements.txt)
PYTHONPATH=src python -m hr_ai_graph_rag
# 或:bash scripts/run_local.sh
```

預設 LLM 是 **Qwen2.5-1.5B-Instruct**(非 gated,免 HF token)。常用環境變數見專案根目錄 `README.md` 的設定表(模型、rerank 權重、USE_LLM、STAGE_LOG 等都可覆蓋)。

---

## 13. 一個問題如何走完整條 workflow(範例)

以「**公司特休是不是比勞基法多?**」為例:

1. **query_understanding**:category=leave、intent=請假與休假;含「公司」「勞基法」→ rewritten_query 補上「特別休假、勞基法第38條、員工內規」;非高風險、非模糊 → answer_policy=direct。
2. **retrieval**:混合檢索找到內規特休條文 + 勞基法第38條 → rerank 精排 → 因含「公司/勞基法/差」觸發 graph 擴展,補進 `policy_特休 --overrides--> law_38` 這條邊。
3. **guardrails**:confidence 足夠、無高風險 → route=answer。
4. **generate_answer**:LLM 看到檢索條文 + 圖關聯,產出「簡短結論:公司特休優於勞基法最低標準… 依據 [S1][S2]… 規範差異(內規 vs 法規)…」並附引用。
5. **faithfulness_check**:有 `[S#]` 且提到條號 → 分數 ~0.97。

若問題改成「**主管不准我請假是不是違法?可以申訴嗎?**」→ 命中 `RISK_KEYWORDS`(違法/申訴)→ heuristic policy=escalate → guardrails route=escalate → 走 ⑥ 直接回「建議轉 HR/法遵」,**不呼叫生成 LLM**。

---

## 14. 設計原則總結

1. **LLM 是訊號,不是裁判**:分類與生成都靠 LLM,但路由由確定性 guardrails 決定。
2. **風險只升不降**:任何風險訊號只能讓路由更嚴格。
3. **圖只走核准邊**:LLM 可建議偏好關係,不能改圖。
4. **生成必須有依據**:只依檢索 context、強制引用、不得編造、不做法律判定。
5. **內規優於法規最低標準**:從 priority 加權到 prompt 規則一以貫之。
6. **可觀測、可稽核**:每階段有日誌,圖/檢索/評估結果全部存檔。

---

# 附錄 A:Prompt 完整內容與 Context Engineering

本附錄逐字列出兩個業務 prompt 的 system / user 內容,並說明「context 怎麼被組裝、限制、注入」的工程細節。

## A1. Query Understanding Prompt(`_llm_runtime_understanding`,`call_llm_json`)

### system prompt(逐字)
```
你是銀行 HR AI 助理的 Runtime Query Understanding 模組。
你的任務是做 structured classification，不是回答問題。
請根據 concept taxonomy、query pattern schema 與 risk policy，判斷使用者問題的 category、intent、matched_concepts、matched_query_pattern_ids、是否模糊、missing_slots、candidate risk。
你可以做口語/錯字 normalization，也可以提供 optional_rewrite_terms，但不得自行創造法條號或內規條號；法條/內規條號會由 offline rewrite_rules 提供。
重要：risk_level / recommended_route 只是 signal，最終路由會由 deterministic guardrails 決定。
只能輸出 valid JSON，不要 markdown，不要解釋。
```
> 再經包裝層 `call_llm_json` 在尾端追加一句:`請只輸出 valid JSON，不要輸出 markdown。`,並用 temperature=0.0、max_new_tokens=384 呼叫。

### user prompt(模板;`{}` 為執行時填入)
```
使用者問題：{question}

可用 categories：general, working_hours, attendance, leave, overtime, salary, welfare, termination, occupational_accident, privacy_sensitive, high_risk, governance
可用 answer_policy：direct, with_disclaimer, clarify, escalate
可用 relation types：has_rule, related_to, refers_to, supplements, overrides, parent_of, child_of

Offline concept taxonomy（節錄）：
{concepts_for_prompt JSON}

Offline query patterns（節錄）：
{patterns_for_prompt JSON}

Offline risk policies（節錄）：
{risk_for_prompt JSON}

Heuristic baseline：
{base JSON}

請輸出 JSON schema：
{ intent, category, matched_concepts[], matched_query_pattern_ids[],
  is_ambiguous, missing_slots[], normalized_query, risk_level,
  risk_reasons[], answer_policy, recommended_route,
  preferred_relations[], optional_rewrite_terms[], confidence }
```

### Context engineering 細節(這個 prompt)
1. **受控選項注入**:把離線知識當「白名單」放進 user prompt,讓 LLM 只能在範圍內分類,而不是自由發揮:
   - `concepts_for_prompt`:概念表**最多 80 個**,每個只保留 concept_id / label / category / **aliases(最多 6 個)** / risk_level / default_answer_policy。
   - `risk_for_prompt`:風險政策**最多 40 個**,每個只留 id / category / risk_level / **trigger_phrases(最多 8 個)** / default_route。
   - `patterns_for_prompt`:查詢樣式**最多 40 個**,每個留 pattern_id / type / category / **examples(最多 6 個)** / missing_slots / default_route。
   - 這些「上限」是**刻意的 token 預算控制**:避免把整個 artifact 塞爆 context window。
2. **Heuristic baseline 注入**:把規則層算好的 `base`(category/intent/risk/policy…)也放進 prompt,讓 LLM 是在「修正既有判斷」而非「從零猜」,提升穩定度。
3. **明確禁令**:system 寫死「不得自行創造法條號/內規條號」「risk 只是 signal」「只輸出 JSON」——這是把「LLM 不做主」的原則寫進 prompt。
4. **輸出後的受控合併**(`_llm_runtime_understanding` 第 309–354 行)是 context engineering 的下半場,規則:
   - category / intent / matched_concepts / pattern_ids / normalized_query / preferred_relations:**可採用** LLM 的值(各有數量上限,如 concepts 最多 8 個)。
   - missing_slots:LLM 與 heuristic **取聯集**(只增不減)。
   - **risk 只升不降**:把 policy 對應成 route,用優先序 `answer<disclaimer<clarify<escalate` 比較,只有 LLM 提議的 route **≥** 目前 route 才採用;**LLM 想把高風險降成 direct 會被拒絕**。
   - `is_ambiguous=true` → 直接設 clarify。
   - risk 訊號全部存進 `llm_risk_signal`(供 guardrails 參考,但不直接決定)。
   - optional_rewrite_terms 最多 5 個,且**不得是條號**,只當語意補充詞。

## A2. Answer Generation Prompt(`_node_generate_answer`,`call_llm_text`)

### system prompt(逐字)
```
你是安久銀行 HR AI 智能助理。
你只能根據提供的 Retrieval Context 與 Graph Context 回答，不得自行編造資料。
回答時必須：
1. 優先使用 internal_policy，其次使用 law 作為最低標準。
2. 必須引用來源，格式使用 [S1], [S2]。
3. 若內規優於法規，請說明「內規 vs 法規」差異。
4. 若是情境型問題，需加風險聲明。
5. 不得做法律判定；高風險或個案爭議需建議洽 HR。
```

### user prompt(模板)
```
員工問題：{question}
Intent: {intent}
Category: {category}
Risk: {risk_level}
Route: {route}

Retrieval / Graph Context:
{context}      ← 由 _make_context 組裝(見下)

請用以下格式回答：
簡短結論：
適用條件：
依據 Citation：
規範差異（內規 vs 法規，如有）：
白話說明：
注意事項 / 聲明：
下一步建議：
```
> 用 temperature=0.1 呼叫;LLM 失敗或 `USE_LLM=false` 時改用 `_fallback_answer` 模板。

### Context engineering 細節(`_make_context`)
生成時的 `{context}` 是這樣組出來的:
1. **取前 6 個** `retrieved_chunks`(已是混合檢索+rerank 後的順序),每個組成一個 `[S{i}]` 區塊,欄位包含:`chunk_type / source_type / document / article_no / title / category / priority / content`,其中 **content 截斷到 900 字**。
2. 區塊之間以 `\n\n---\n\n` 分隔,**讓 LLM 能用 `[S1]…[S6]` 對齊引用**(引用編號 = chunk 在 context 裡的順位)。
3. 在最後**附上圖 context**:`graph_context.context`(即 §7 Step 3 組出的 `[Graph Nodes]` / `[Graph Relations]` 文字),標題為 `[Graph-enhanced Context]`,**截斷到 2500 字**。
4. `_build_citations` 另外產生結構化引用清單(前 6 個 chunk 的 source_id / 條號 / 來源 / 分數 / 內容預覽),存進 `state["citations"]`,供顯示與評估比對。

### 為何這樣設計(context engineering 思路)
- **編號對齊**:context 用 `[S1..S6]`、prompt 規則要求「必須用 [S#] 引用」、citations 也用同樣 source_id —— 三者對齊,才能在 faithfulness_check 驗證「答案是否真的引用了檢索到的來源」。
- **長度上限**:chunk content 900 字、graph context 2500 字,是 1.5B 小模型 context window 與生成品質的折衷。
- **欄位即訊號**:把 source_type / priority / category 一起放進 context,讓 LLM 知道「哪些是內規(優先)、哪些是法規(最低標準)」,呼應 system 規則第 1、3 條。
- **格式強約束**:固定 7 段回答格式,讓輸出可預期、好稽核,也逼模型分開「結論 / 依據 / 內規 vs 法規差異 / 聲明」。

## A3. 兩個包裝層回顧
- **JSON 強制層**(`call_llm_json`):僅 Prompt ① 經過,追加「只輸出 JSON」、temperature=0,回傳後用 regex 抽 JSON,失敗回 default。
- **Chat template 組裝層**(`_format_messages`):兩個 prompt 都經過;先試 `[{system},{user}]`,模型不收 system 角色就**合併進 user**,沒有 chat_template 的模型用 `System/User/Assistant` 純文字。

---

# 附錄 B:知識圖譜建置原理詳解

本附錄說明「圖是怎麼從原始 DOCX 與 artifacts 長出來的」,以及執行時擴展的走訪原理。

## B1. 節點與邊的三個來源
知識圖的素材來自三條獨立管道,在 `_build_graph` 匯流:

| 來源 | 產生什麼 | 在哪 |
|---|---|---|
| **DOCX 條文解析** | `law_article` / `internal_policy_article` 節點;條文內參照的 `refers_to`、`overrides`/`refers_to` 邊 | `ingestion.parse_*_articles` |
| **concept_nodes artifact** | `concept` 節點;`parent_of`/`child_of`;概念↔條文 `has_rule`/`related_to` | `graph._add_concept_nodes` / `_add_concept_article_edges` |
| **graph_relation_candidates artifact** | 額外的、人工核准過的條文間關係邊(`overrides`/`supplements`…) | `graph._add_artifact_edge_candidates` |

## B2. 條文解析如何生出「圖的素材」(`ingestion.py`)
- **法規(`parse_labor_law_articles`)**:用 `CHAPTER_PATTERN` / `ARTICLE_PATTERN` 切章節與條文,產生 `source_type="law"`、`priority=1`、`related_articles=[]` 的條文記錄。法規本身不主動連出邊(它是「被引用」的最低標準)。
- **內規(`parse_internal_policy_articles`)**:是**圖關係的主要來源**。每條內規 `flush()` 時:
  1. `extract_related_law_ids(content)`:用正則從內規條文裡抓「勞動基準法…第 X 條」,轉成對應的 **law article_id**,存進 `related_articles`。
  2. 對每個 related law id,用 `relation_from_policy_content(content)` 決定關係型別:
     - 內容含「優於 / 較有利 / 不低於 / 最低標準」→ **`overrides`**(內規優於法規);
     - 否則 → **`refers_to`**(僅參照)。
  3. 組成 `graph_edges = [(relation, law_id), …]`,連同 `priority=2` 一起存進條文記錄。
- 之後 `_build_graph` 第 2 步就把這些 `related_articles`(refers_to)與 `graph_edges`(overrides/refers_to)實際加到圖上。**這就是「內規 overrides 法規」這條關鍵邊的真正出處。**
- 解析失敗(非條文格式)時有 fallback:用 900 字、120 重疊的固定切塊,仍盡量抓 related law ids。

## B3. 條號正規化與比對(把文字對應到圖節點的關鍵)
圖能不能正確連邊,取決於「能不能把各種寫法的條號對應到同一個節點」。三段機制:
1. **`article_id_from_no(source_type, article_no)`**:把「第 38 條」正規化成穩定 id,如 `law_38`、`policy_11`(「之」轉 `_`)。節點 id 與引用都用它,確保一致。
2. **`_build_article_no_lookup`**:建一張 `(source_type, 正規化條號) → article_id` 的查表,並額外建 `("any", 條號)` 的萬用對應。
3. **`_resolve_article_ref(ref, preferred_source)`**:三段式解析,把「任何形式的條文參照」對應到圖節點:
   - 直接就是 article_id → 用它;
   - 形如 `policy_article_11` / `law_article_38` → 轉來源+條號查表;
   - 人類可讀的「勞動基準法第38條 / 安久銀行…第11條」→ 用 `extract_article_refs` 抓條號,並依字串含「勞動基準法/安久銀行/內規」決定 preferred source,再查表。
   - 這支函式是 concept↔article 與 artifact 候選邊「**端點對齊**」的核心。

## B4. 建圖四步(`_build_graph`)與每步的邊
1. **概念節點**:對每個 concept_node 建 `concept` 節點(帶 label/category/risk_level/aliases/graph_expansion_priority…);有 `parent_concept_id` 就雙向加 `parent_of` / `child_of`。無 artifact 時用 8 個內建概念骨架。
2. **條文節點 + 文件內邊**:把每條 article 加成節點;加 `related_articles` 的 `refers_to`;加 `graph_edges`(內規→法規的 overrides/refers_to)。
3. **概念↔條文邊**:對每個概念,把 `related_law_articles` / `related_policy_articles` 經 `_resolve_article_ref` 對應到條文節點,雙向加 `has_rule`(概念→條文)與 `related_to`(條文→概念)。無 concept artifact 時用 category 對應的 fallback 邊。
4. **artifact 候選邊**:從 `graph_relation_candidates` 載入額外關係,但有**雙重把關**(見 B6)。

## B5. relation 型別語意
| relation | 意義 |
|---|---|
| `overrides` | 內規優於法規最低標準(最有價值的差異關係) |
| `supplements` | 內規補充法規 |
| `has_rule` / `related_to` | 概念 ↔ 條文的雙向關聯 |
| `refers_to` | 條文參照另一條文 |
| `parent_of` / `child_of` | 概念階層 |

## B6. 圖治理(為什麼可信)
- **只載已核准邊**:`_add_artifact_edge_candidates` 預設只收 `review_status=="approved"` 的候選邊;未核准(pending)的**不載入**,除非設 `LOAD_PENDING_GRAPH_EDGES=true`。
- **relation 白名單**:候選邊的 relation 必須在 `relation_schema` 標記為 `runtime_expandable` 的集合內(否則用內建白名單),不在白名單的關係被丟棄。
- **端點必須存在**:來源/目標都得能對應到圖中既有節點才加邊。
- **保留稽核欄位**:邊上存 `evidence`、`confidence`、`review_status`、`human_review_required`,可事後追溯。
- 這些機制讓圖是「**人工治理過的可信關係**」,而非 LLM 即興產生。

## B7. 執行時擴展的走訪原理(`expand`)
- **觸發條件**:問題含「差 / 比較 / 為什麼 / 內規 / 法規 / 公司 / 優於 / 補休 / 依據 / 哪個 / 關係」才啟用圖(否則回空,避免雜訊)——因為只有「比較/關聯型」問題才需要圖。
- **走訪**:以 rerank 後 top chunks 的條文為種子,沿 **successors + predecessors(雙向)** 走 `hops=1`,最多 `max_nodes=14` 個節點(BFS,達上限即停)。
- **關係過濾**:若上游 LLM 給了 `preferred_relations`,**只保留**這些關係的邊(其餘鄰居跳過);沒給就全收。這是 LLM 唯一能影響圖的方式——**只能「偏好」,不能「新增」**。
- **產出**:節點(node_type/label/條號/來源/類別/風險/content 前 500 字)+ 邊(source/relation/target/evidence)+ 組好的文字 context,回傳並注入生成 prompt(見 A2)。

## B8. 一句話總結圖譜原理
**節點來自「解析後的條文 + 概念 artifact」,邊主要來自「內規條文裡對法規的引用(overrides/refers_to)+ 概念對應 + 人工核准的候選邊」;靠條號正規化與三段式 `_resolve_article_ref` 對齊端點;只載核准邊、relation 白名單把關;執行時針對比較型問題做 1-hop 雙向擴展,LLM 只能偏好關係、不能改圖。**

---

# 附錄 C:切塊(Chunking)詳解與範例

切塊發生在離線建構階段的 `HRKnowledgeBuilder.build_chunks`(`ingestion.py`)。流程是:**先把 DOCX 解析成「條文(article)」,再從條文衍生出 4 種不同粒度的 chunk**,讓檢索能在「整份文件 / 單條條文 / 條文內語意片段 / FAQ」四個層次上比對。這就是所謂 **Policy-aware Hierarchical Chunking(政策感知的階層式切塊)**。

## C0. 為什麼要多粒度切塊
不同問題需要不同粒度的命中:
- 寬泛問題(「公司有哪些假?」)→ 適合 **document** 層的整份摘要;
- 明確條文問題(「特休幾天?」)→ 適合 **article** 層的完整條文;
- 細節語意問題(「未休完的特休可以遞延嗎?」)→ 適合 **semantic** 層的單一句子片段(向量檢索對短而集中的片段更準)。

把同一份資料同時切成多層,檢索時哪一層命中都算數(混合檢索取聯集),再靠 rerank 與 metadata 加成挑出最適合的。

## C1. 第一步:DOCX → 條文(parse)
- **法規**(`parse_labor_law_articles`):用章(`CHAPTER_PATTERN`)、條(`ARTICLE_PATTERN`)正則切;每條產生 `source_type="law"`、`priority=1`、`related_articles=[]`。
- **內規**(`parse_internal_policy_articles`):用內規條文格式正則切;**會額外算出圖的素材**——`related_articles`(從文中抓到的勞基法條號)與 `graph_edges`(overrides / refers_to);`priority=2`;跳過只有標題、沒有內文(< 8 字)的目錄列。
- 解析失敗時 fallback:用 **900 字、120 重疊** 的固定切塊,確保流程不中斷。
- 每條 article 是一個 dict,欄位:`source_type, document_name, article_id, article_no, chapter, title, category, priority, version, effective_date, content, related_articles(, graph_edges)`。

> `article_id` 由 `article_id_from_no` 正規化而來(例:第 18 條 → `policy_18`);`category` 由 `detect_category` 用關鍵字判定。

## C2. 第二步:條文 → 4 種 chunk

所有 chunk 共用欄位:`chunk_id, chunk_type, source_type, document_name, article_id, article_no, chapter, title, category, priority, version, effective_date, content, parent_id, related_articles, keywords`。

### ① document chunk(文件層)— `make_document_chunks`
- **每份文件一個** chunk;把該文件**前 20 條**條文各取 `「條號 標題：內容前 160 字」` 串成摘要。
- `priority` = 該文件所有條文的**最大值**;`category` = 所有條文類別的**聯集**(逗號分隔);`article_no="DOCUMENT"`;`parent_id=None`。
- 用途:整份文件的鳥瞰摘要,適合寬泛問題。

### ② article chunk(條文層)— `make_article_chunks`
- **每條條文一個** chunk,**1:1**,`content` 是該條**完整內文**。
- `parent_id = "document::<文件名>"`(指回所屬文件層);保留 `related_articles`。
- 用途:最常命中的主力層,單條條文精準回答。

### ③ semantic chunk(語意子句層)— `make_semantic_subchunks`
- 把條文 `content` 用 `split_sentences_zh`(以 `。！？；` 斷句)切成句子,**每累積到約 120 字**併成一組(group)。
- **跳過規則**:若整條只會切出 ≤ 1 組 **且** content < 220 字 → **不產生 semantic chunk**(太短沒必要再切)。
- 每組一個 chunk,`article_id = "<母條 id>::s{j}"`,`parent_id = 母條 article_id`。
- 用途:把長條文打散成短而集中的片段,**向量檢索對短片段更準**,利於細節語意問題。

### ④ faq chunk(FAQ 層,預設關閉)— `make_faq_chunks`
- **只有 `USE_GOLDEN_AS_FAQ_CHUNKS=true` 才產生**(預設 false,避免把評估資料洩漏進知識庫)。
- 從 golden 題的 `question` + `expected_key_points` 組成 `「FAQ 問題:… FAQ 回答重點:…」`;`source_type="golden_dataset_faq_experiment"`、`priority=1`。

## C3. 第三步:收尾(`build_chunks` 末段)
- 把 4 種 chunk 合併:`chunks = document + article + semantic + faq`;
- 給每個 chunk 一個全域序號 `global_chunk_id`(`G00001`…);
- 算 `embedding_text`(`chunk_to_text`)——**這才是真正被嵌入/檢索的文本**,格式:
  ```
  Chunk Type: …
  Source Type: …
  Document: …
  Article: …
  Title: …
  Category: …
  Priority: …
  Content: …
  Keywords: …
  ```
  把結構化欄位一起放進嵌入文本,讓向量同時帶有「類型/來源/類別」訊號。

## C4. 完整範例

假設內規 DOCX 裡有這麼一條(解析後的 `content`):

```
第 18 條 特別休假
員工到職滿六個月以上未滿一年者，得申請特別休假三日；滿一年以上未滿二年者七日；滿二年以上未滿三年者十日。
本公司特別休假日數優於勞動基準法第 38 條之最低標準，屬較有利之約定。
特別休假因年度終結或契約終止而未休之日數，雇主應發給工資；經勞雇雙方協商同意者，得遞延至次一年度實施。
員工申請特別休假應提前三個工作日於差勤系統提出，並經單位主管核准。
```

### 解析後的 article(C1 產出)
```
article_id      = policy_18
article_no      = 第 18 條
title           = 特別休假
source_type     = internal_policy
category        = leave            (detect_category 命中「特別休假/休假」)
priority        = 2
related_articles= [law_38]         (extract_related_law_ids 抓到「勞動基準法第 38 條」)
graph_edges     = [("overrides", law_38)]   (內容含「優於/最低標準/較有利」→ overrides)
```

### 這一條會衍生出的 chunk

**① document chunk(整份內規共用一個,這條只是其中一行)**
```
chunk_type = document
article_no = DOCUMENT
content    = "... 第 18 條 特別休假：員工到職滿六個月以上未滿一年者，得申請特別休假三日；滿一年...（前160字）\n 第 19 條 ...\n ..."
priority   = 2   (取全文件最大)
```

**② article chunk(這條一個,完整內文)**
```
chunk_id   = article_00xx
chunk_type = article
article_id = policy_18
content    = 「第 18 條 特別休假\n員工到職滿六個月…並經單位主管核准。」(整條)
parent_id  = document::安久銀行員工工作與福利規章辦法_模擬版.docx
related_articles = [law_38]
```

**③ semantic chunks(因為 content > 220 字,會切成多個語意組)**
以約 120 字為界,大致切成 2 組:
```
semantic_00xx  article_id=policy_18::s1  parent_id=policy_18
  content ≈ 「第 18 條 特別休假 員工到職滿六個月…十日。本公司特別休假日數優於勞動基準法第 38 條之最低標準，屬較有利之約定。」

semantic_00xx  article_id=policy_18::s2  parent_id=policy_18
  content ≈ 「特別休假因年度終結或契約終止而未休之日數，雇主應發給工資；經勞雇雙方協商同意者，得遞延至次一年度實施。員工申請特別休假應提前三個工作日於差勤系統提出，並經單位主管核准。」
```
> 若這條只有短短一句(< 220 字且切不出第二組),則**不會**產生 semantic chunk,只留 document + article 兩層。

**④ faq chunk**:預設不產生(除非開 `USE_GOLDEN_AS_FAQ_CHUNKS`)。

### 檢索時誰會被命中(把切塊和 §7 串起來)
- 問「**特休未休完可以遞延嗎?**」→ 含「遞延」的 **semantic ::s2** 片段向量相似度最高,容易被選中;
- 問「**公司特休比勞基法多嗎?**」→ **article**(policy_18)命中,且因含「公司/勞基法/優於」觸發圖擴展,沿 `policy_18 --overrides--> law_38` 補進勞基法第38條;
- 問「**公司有哪些假?**」→ **document** 摘要層較易整體命中。

## C5. 可調參數(都寫在 `ingestion.py`)
| 參數 | 預設 | 位置 / 作用 |
|---|---|---|
| document 摘要每條取字數 | 160 | `make_document_chunks`,每條摘要長度 |
| document 收錄條數 | 前 20 條 | `make_document_chunks` |
| semantic 併組門檻 | 120 字 | `make_semantic_subchunks`,累積到此長度就切一組 |
| semantic 跳過門檻 | 220 字 | `make_semantic_subchunks`,過短的條文不切 semantic |
| parse fallback 切塊 | 900 字 / 120 重疊 | `parse_*` 解析失敗時的固定切塊 |
| FAQ chunk | `USE_GOLDEN_AS_FAQ_CHUNKS=false` | 是否把 golden 轉成 FAQ chunk(預設關閉) |

> 這些是寫死的常數;若要調(例如 semantic 想切更細),直接改 `ingestion.py` 對應數值即可。

## C6. 一句話總結切塊
**先把 DOCX 解析成條文(法規 priority=1、內規 priority=2 並順便算出 overrides 圖邊),再對每條同時產生 document(整份摘要)/ article(完整條文)/ semantic(約 120 字語意片段,短條文略過)三種粒度的 chunk(FAQ 預設關閉),最後組成含結構化欄位的 `embedding_text` 供嵌入與混合檢索。**

---

# 附錄 D:RAGAS 評估指標實作詳解

本附錄說明這份 repo 如何在「**不依賴 OpenAI / 不額外接外部 API**」的前提下,實作 RAGAS 四大指標:**context recall、context precision、faithfulness、answer relevancy**。對應程式碼為 `evaluation.py` 的 `RagasScorer`(embedding backend)、`LlmRagasScorer`(LLM-as-judge backend)、`_make_ragas_scorer`、`_build_eval_entry`。

## D0. 為什麼不直接用官方 `ragas` 套件

官方 `ragas` 預設要一顆 LLM 當 judge **而且預設接 OpenAI**,還要 embeddings。這會破壞本專案三個前提:**(1) 無 API key、(2) Colab 50 分鐘時間限制、(3) 全離線/本地模型**。50 題 × LLM judge 也很慢。因此這裡**自己實作 RAGAS 的語意**,並提供兩種 backend,讓使用者按時間預算選擇。

## D1. 兩種 backend 與切換

| backend | 用什麼算 | 速度 | 何時用 |
|---|---|---|---|
| `llm`(預設) | **本地 Qwen 當 judge**(重用答題那顆,不重載) | 慢(每題約 4 次 LLM 呼叫) | canonical RAGAS LLM-as-judge(預設) |
| `embedding` | 已載入的 **bge-m3** embedder 算 cosine 相似度 proxy | 快(無額外 LLM 呼叫) | 想快速、可重現、省時間 |

切換用環境變數(`_make_ragas_scorer`):
```bash
USE_RAGAS=true            # 總開關(false=四指標全 None)
RAGAS_BACKEND=llm         # 預設(Qwen as judge);改 embedding 用 bge-m3 proxy
RAGAS_SIM_THRESHOLD=0.5   # embedding backend:precision/recall 的相關判定門檻
RAGAS_LLM_MAX_CONTEXTS=6  # llm backend:judge 每題看幾段 context
RAGAS_LLM_CTX_CHARS=400   # llm backend:每段 context 在 prompt 裡截斷字數
```

`_make_ragas_scorer(assistant)` 的決策:`USE_RAGAS!=true` → 回 `None`;`RAGAS_BACKEND`(預設 `llm`)為 `llm` → 回 `LlmRagasScorer`;為 `embedding` → 回 `RagasScorer`(從 `assistant.retriever.embedder` 取 bge-m3,取不到則回 `None` 並提示)。

## D2. 每題的輸入怎麼來(`_build_eval_entry`)

評估迴圈每題呼叫一次 `scorer.score(question, answer, contexts, ground_truths)`,四個輸入:

| 輸入 | 來源 | 備註 |
|---|---|---|
| `question` | golden 題目 | — |
| `answer` | `result["answer"]`(生成答案) | — |
| `contexts` | `result["retrieved_chunks"]` 各 chunk 的 `content`,**取前 8 段** | 即 §7 混合檢索+rerank 後的順序;空字串會過濾 |
| `ground_truths` | 由 `_resolve_ground_truths(row)` 決定(見下),回傳「一串標準答案陳述」 | 用於 context recall / precision;空 → 這兩個指標記 `None` |

**`_resolve_ground_truths` 的優先序**:本資料集只內建 `expected_key_points`,但若你在 golden JSON 為某題補上**完整答案全文欄位**(依序找 `expected_answer` / `reference_answer` / `ground_truth` / `golden_answer`),scorer 會**優先採用該全文**,並用 `split_sentences_zh` 把它**斷句成一句句陳述**(符合 RAGAS「把標準答案拆解成 statements 再逐句檢查」的語意,讓 recall/precision 保有逐句粒度);全文欄位是 list 就直接當陳述清單用;都沒有(或為空/NaN)才**退回 `expected_key_points`**。

> golden 50 題**每題都有 `expected_key_points`**(已是拆好的要點清單,本身就是這份資料的 ground truth),所以 context precision/recall 都算得出來。換言之「正確答案」就是這些要點;要改用整段答案,只要在該題加一個 `expected_answer` 欄位即可,無痛接上、現有資料不受影響。

## D3. Embedding backend(`RagasScorer`)— 公式

所有相似度都用 **normalize 後的 bge-m3 向量**,所以 cosine = 內積。`_sim(A,B)` 回傳 `A·Bᵀ` 矩陣。門檻 `τ = RAGAS_SIM_THRESHOLD`(預設 0.5)。

1. **faithfulness**(答案是否 grounding 在 context):
   - 把 `answer` 用 `split_sentences_zh` 斷成句子 `{s_i}`,context 為 `{c_j}`;
   - `faithfulness = mean_i ( max_j cos(s_i, c_j) )` — **每句對所有 context 取最大相似度再平均**。連續值 [0,1],不需門檻。
2. **answer_relevancy**(答案是否切題):
   - `answer_relevancy = cos(answer, question)` — 整段答案 vs 問題的 cosine。
3. **context_recall**(該檢索到的有沒有檢索到):
   - 對每個 ground-truth 要點 `g_k`,看是否有任一 context 夠相似:`covered_k = 1[ max_j cos(g_k, c_j) ≥ τ ]`;
   - `context_recall = mean_k covered_k` — **被涵蓋的要點比例**。
4. **context_precision**(相關 context 有沒有排前面):
   - 對排名順序的每段 context `c_j`,標記是否相關:`rel_j = 1[ max_k cos(c_j, g_k) ≥ τ ]`;
   - 對這個 0/1 序列算 **average precision(AP)**:`AP = (Σ_j rel_j · precision@j) / (Σ_j rel_j)`,其中 `precision@j` = 前 j 段裡相關的比例。相關 context 排越前,分數越高;全不相關時記 0.0。

## D4. LLM backend(`LlmRagasScorer`)— Qwen 當 judge

重用 `call_llm_json`(就是 §8 包裝層 ③,temperature=0、要求只輸出 JSON、regex 抽 JSON),呼叫的是**答題用的同一顆 Qwen**,不重載、不接外部 API。context 在 prompt 裡**最多 `RAGAS_LLM_MAX_CONTEXTS`(6)段、每段截 `RAGAS_LLM_CTX_CHARS`(400)字**,控制 token。

每個指標一次 judge 呼叫(每題最多 4 次):

1. **faithfulness**:給 context + answer,要 Qwen 把答案**拆成數個陳述**並逐一判斷能否由 context 支持 → JSON `{"statements":[{"text","supported":bool}]}` → `score = 支持數 / 總數`。
2. **context_recall**:給 context + 標準答案要點(逐點),要 Qwen 判斷每點**能否在 context 找到依據** → `{"points":[{"text","attributable":bool}]}` → `score = 可歸因數 / 總數`。
3. **context_precision**:給標準答案要點 + 排名後的檢索段落,要 Qwen 逐段判斷**是否相關** → `{"contexts":[{"index","relevant":bool}]}` → 依 index 還原 0/1 序列,算與 D3 相同的 **average precision**。
4. **answer_relevancy**:給 question + answer,要 Qwen 給 **0–100 整數分** → `{"score":int}` → `score/100` 並夾到 [0,1]。

**穩健性**:小模型偶爾吐壞 JSON。`call_llm_json` 抽不到就回預設 `{}`,對應指標記 `None`;`answer_relevancy` 的 `score` 非數字也記 `None`。`None` 不會灌進平均(見 D5),保持誠實。

## D5. 寫進哪、平均怎麼算

- 四個欄位名固定為 `RAGAS_FIELDS`:`ragas_faithfulness`、`ragas_answer_relevancy`、`ragas_context_precision`、`ragas_context_recall`。
- 每題透過 `_build_eval_entry` 併進該題 entry,流向 **checkpoint JSONL → `evaluation_detail.csv` → `evaluation_records*.json`**;`_entry_to_row` / `_entry_to_record` 都用 `.get()` 帶出,缺值為 `None`。
- summary 用 `_mean_opt(col)`:**只對非 None 的數值取平均**(`None` / `NaN` 一律跳過),得到 `RAGAS Context Recall / Precision / Faithfulness / Answer Relevancy` 四個平均。某指標整欄皆 None(如關閉或全部 parse 失敗)→ 平均為 `None`。

## D6. 與 checkpoint 續跑的關係

RAGAS 指標是**逐題隨答案一起寫進 `eval_checkpoint.jsonl`**(每行一題,fsync 落地)。所以:

- Colab 50 分鐘斷線最多損失「正在算的那一題」;下次執行 `_load_eval_checkpoint` 讀回已完成題、**跳過(連 RAGAS 一起跳過,不重算)**,只補跑剩下的;
- 最終 detail / summary 一律對**全部 50 題**(已完成 + 新跑)計算;
- 注意:**舊 checkpoint(本功能之前產生的)沒有 RAGAS 欄位,resume 時不會回填**(那些題會是 `None`);要全題都有 RAGAS,刪掉 `eval_checkpoint.jsonl` 重跑即可。

## D7. 一句話總結 RAGAS 實作

**四個 RAGAS 指標都在本地、無外部 API 下計算:預設用 bge-m3 cosine 相似度做 proxy(faithfulness=答案句對 context 的最大相似度均值、answer relevancy=答案對問題 cosine、context recall=要點被涵蓋比例、context precision=相關 context 的 average precision);或設 `RAGAS_BACKEND=llm` 改用本地 Qwen 當 judge 做 canonical 的拆陳述/判歸因/判相關;結果逐題寫入可續跑的 checkpoint,summary 以跳過 None 的方式取平均。**
