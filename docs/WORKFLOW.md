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
| 勞動基準法 DOCX | 外部法律 | **未內附**;無則用內建 sample | 屬外部法源,不隨 repo 散布;沒提供時用內建關鍵條文 sample 讓流程能跑完 |
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
- 勞基法 DOCX 永遠不內附 → **沒提供就產生內建 sample 條文**;
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
| Avg Faithfulness Score | 平均忠實度 |
| Avg Latency Sec | 平均每題耗時 |

另有 `log_feedback` 紀錄使用者回饋(helpful、正確度、完整度、評論)到 `feedback_log.csv`。

---

## 10. 輸出檔案(`runner.save_outputs`)

一次完整執行會把結果寫到 `OUTPUT_DIR`(`./hr_ai_graph_rag_outputs`,Colab 為 `/content/...`,已 gitignore):

- `articles.json`、`chunks_3layer_faq.json` / `.csv`、`demo_results.json`
- `kg_nodes.csv`、`kg_edges.csv`、`hr_knowledge_graph.gexf`(可用 Gephi 開)
- `golden_dataset.csv`、`evaluation_detail.csv`、`evaluation_summary.csv`
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
```
