# ============================================================
# workflow — 對話流程(HRAssistantGraph / LangGraph)
#
# 查詢理解 → 檢索編排(混合 + 圖擴展) → deterministic guardrails → 
# grounded 生成 → faithfulness 檢查。LLM 只提供訊號,最終 route 由規則決定。
# 依賴:config、utils、artifacts、llm、graph、retrieval。
# ============================================================

from .config import *
from .utils import *
from .artifacts import *
from .llm import *
from .graph import *
from .retrieval import *


class HRState(TypedDict, total=False):
    question: str
    intent: str
    category: str
    risk_level: str
    answer_policy: Literal["direct", "with_disclaimer", "escalate", "clarify"]
    matched_concepts: List[str]
    missing_slots: List[str]
    llm_risk_signal: Dict[str, Any]
    risk_matches: List[Dict[str, Any]]
    preferred_relations: List[str]
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    confidence: float
    route: Literal["answer", "disclaimer", "escalate", "clarify"]
    answer: str
    citations: List[Dict[str, Any]]
    faithfulness_score: float
    debug: List[str]


def add_debug(state: HRState, message: str) -> List[str]:
    return state.get("debug", []) + [message]


class HRAssistantGraph:
    def __init__(self, retriever: HybridRetriever, kg: HRKnowledgeGraph, artifacts: Optional[OfflineArtifacts] = None):
        self.retriever = retriever
        self.kg = kg
        self.artifacts = artifacts or OfflineArtifacts()
        self.app = self._build_graph()

    def _match_concepts(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = normalize_for_match(question)
        matches = []
        for c in self.artifacts.concept_nodes:
            terms = [c.get("label", "")] + (c.get("aliases", []) or []) + (c.get("retrieval_keywords", []) or [])
            score = 0
            matched_terms = []
            for t in terms:
                t_norm = normalize_for_match(t)
                if t_norm and t_norm in q:
                    score += 2 if t in (c.get("aliases", []) or []) else 1
                    matched_terms.append(t)
            if score > 0:
                matches.append({
                    "concept_id": c.get("concept_id"),
                    "label": c.get("label"),
                    "category": c.get("category"),
                    "risk_level": c.get("risk_level"),
                    "default_answer_policy": c.get("default_answer_policy", "answer"),
                    "score": score,
                    "matched_terms": matched_terms,
                })
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    def _match_risk_policies(self, question: str) -> List[Dict[str, Any]]:
        q = normalize_for_match(question)
        matches = []
        for p in self.artifacts.risk_policies:
            hits = []
            for t in p.get("trigger_phrases", []) or []:
                if normalize_for_match(t) in q:
                    hits.append(t)
            if hits:
                rec = dict(p)
                rec["matched_triggers"] = hits
                matches.append(rec)
        # high risk first
        order = {"high": 3, "medium": 2, "low": 1, "高": 3, "中": 2, "低": 1}
        matches.sort(key=lambda x: order.get(x.get("risk_level"), 0), reverse=True)
        return matches

    def _match_query_patterns(self, question: str) -> List[Dict[str, Any]]:
        q = normalize_for_match(question)
        matches = []
        for p in self.artifacts.query_patterns:
            for ex in p.get("examples", []) or []:
                exn = normalize_for_match(ex)
                if exn and (exn in q or q in exn):
                    matches.append(dict(p))
                    break
        return matches

    def _get_offline_rewrite_terms(self, matched_concepts: List[str], category: str) -> Dict[str, List[str]]:
        """Return controlled offline rewrite terms.

        These are mandatory / trusted expansion terms. Runtime local LLM is not allowed
        to invent law article IDs or policy article IDs; those should come from these
        offline artifacts or concept_nodes.
        """
        terms = []
        article_hints = []
        for r in self.artifacts.rewrite_rules:
            if r.get("concept_id") in matched_concepts or (category and r.get("category") == category):
                terms.extend(r.get("rewrite_terms", []) or [])
                article_hints.extend(r.get("article_hints", []) or [])
        # Add concept-node related article hints as another controlled source.
        concept_map = {c.get("concept_id"): c for c in self.artifacts.concept_nodes}
        for cid in matched_concepts or []:
            c = concept_map.get(cid) or {}
            terms.extend(c.get("retrieval_keywords", []) or [])
            article_hints.extend(c.get("related_law_articles", []) or [])
            article_hints.extend(c.get("related_policy_articles", []) or [])
        terms = list(dict.fromkeys([str(t) for t in terms if t]))[:28]
        article_hints = list(dict.fromkeys([str(t) for t in article_hints if t]))[:16]
        return {"mandatory_terms": terms, "article_hints": article_hints}

    def _rewrite_from_artifacts(
        self,
        question: str,
        matched_concepts: List[str],
        category: str,
        fallback: str,
        normalized_query: Optional[str] = None,
        optional_terms: Optional[List[str]] = None,
    ) -> str:
        """Hybrid query rewrite.

        Mature workflow:
        - Offline rewrite_rules / concept_nodes provide mandatory controlled terms.
        - Runtime local LLM may provide optional semantic terms only.
        - Deterministic merge builds the final retrieval query.
        """
        offline = self._get_offline_rewrite_terms(matched_concepts, category)
        mandatory_terms = offline["mandatory_terms"]
        article_hints = offline["article_hints"]
        optional_terms = list(dict.fromkeys([str(t) for t in (optional_terms or []) if t]))[:LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS]
        base_query = normalize_spaces(normalized_query or question)
        parts = [base_query]
        if mandatory_terms:
            parts.append(" ".join(mandatory_terms))
        if article_hints:
            parts.append(" ".join(article_hints))
        if optional_terms:
            parts.append(" ".join(optional_terms))
        if len(parts) > 1:
            return "｜".join(parts)
        return fallback

    def _heuristic_understanding(self, question: str) -> Dict[str, Any]:
        q = question.strip()
        category = detect_category(q)
        concept_matches = self._match_concepts(q)
        if concept_matches:
            category = concept_matches[0].get("category") or category
        if category == "overtime":
            intent = "工時與加班"
        elif category == "leave":
            intent = "請假與休假"
        elif category == "salary":
            intent = "薪資與工資"
        elif category == "termination":
            intent = "離職與資遣"
        elif category == "attendance":
            intent = "出勤與工時"
        elif category == "welfare":
            intent = "員工福利"
        elif category == "occupational_accident":
            intent = "職業災害"
        elif category == "privacy_sensitive":
            intent = "個資與敏感資訊"
        else:
            intent = "一般 HR 規章"

        risk_matches = self._match_risk_policies(q)
        pattern_matches = self._match_query_patterns(q)
        high_risk = any(str(r.get("risk_level", "")).lower() in ["high", "高"] for r in risk_matches)
        ambiguous = bool(pattern_matches) or q in AMBIGUOUS_SHORTS or len(q) <= 5

        # Concept priors can raise risk but should not alone decide final route when ambiguous.
        concept_policies = [m.get("default_answer_policy") for m in concept_matches if m.get("default_answer_policy")]
        concept_high = any(str(m.get("risk_level", "")).lower() in ["high", "高"] for m in concept_matches)

        if ambiguous:
            answer_policy = "clarify"
        elif high_risk or concept_high or any(k in q for k in RISK_KEYWORDS):
            answer_policy = "escalate"
        elif any(k in q for k in ["是否合法", "違法嗎", "可以告", "申訴", "主管不給", "被逼"]):
            answer_policy = "escalate"
        elif "with_disclaimer" in concept_policies or any(k in q for k in ["我這種情況", "如果", "主管", "個案", "薪資明細"]):
            answer_policy = "with_disclaimer"
        else:
            answer_policy = "direct"
        risk_level = "高" if answer_policy == "escalate" else "中" if answer_policy == "with_disclaimer" else "低"

        rewrite_terms = []
        if category == "overtime": rewrite_terms = ["加班", "延長工時", "加班費", "補休", "勞基法第24條"]
        elif category == "leave": rewrite_terms = ["請假", "特別休假", "病假", "事假", "員工內規", "勞基法第38條"]
        elif category == "salary": rewrite_terms = ["工資", "薪資給付", "扣薪", "加班費", "勞基法第22條"]
        elif category == "termination": rewrite_terms = ["資遣", "終止勞動契約", "預告期間", "勞基法第16條"]
        elif category == "welfare": rewrite_terms = ["員工福利", "旅遊補助", "HR公告", "內部規章"]
        fallback_rewrite = f"{intent}｜{q}｜{' '.join(rewrite_terms)}"
        matched_concepts = [m.get("concept_id") for m in concept_matches if m.get("concept_id")]
        rewritten_query = self._rewrite_from_artifacts(q, matched_concepts, category, fallback_rewrite)
        missing_slots = []
        for p in pattern_matches:
            missing_slots.extend(p.get("missing_slots", []) or [])
        missing_slots = list(dict.fromkeys(missing_slots))
        return {
            "intent": intent,
            "category": category,
            "risk_level": risk_level,
            "answer_policy": answer_policy,
            "matched_concepts": matched_concepts,
            "missing_slots": missing_slots,
            "llm_risk_signal": {},
            "risk_matches": risk_matches,
            "preferred_relations": [],
            "rewritten_query": rewritten_query,
        }

    def _llm_runtime_understanding(self, question: str, base: Dict[str, Any]) -> Dict[str, Any]:
        concepts_for_prompt = []
        for c in self.artifacts.concept_nodes[:80]:
            concepts_for_prompt.append({
                "concept_id": c.get("concept_id"),
                "label": c.get("label"),
                "category": c.get("category"),
                "aliases": (c.get("aliases", []) or [])[:6],
                "risk_level": c.get("risk_level"),
                "default_answer_policy": c.get("default_answer_policy"),
            })
        risk_for_prompt = []
        for p in self.artifacts.risk_policies[:40]:
            risk_for_prompt.append({
                "risk_policy_id": p.get("risk_policy_id"),
                "category": p.get("category"),
                "risk_level": p.get("risk_level"),
                "trigger_phrases": (p.get("trigger_phrases", []) or [])[:8],
                "default_route": p.get("default_route"),
            })
        patterns_for_prompt = []
        for p in self.artifacts.query_patterns[:40]:
            patterns_for_prompt.append({
                "pattern_id": p.get("pattern_id"),
                "pattern_type": p.get("pattern_type"),
                "category": p.get("category"),
                "examples": (p.get("examples", []) or [])[:6],
                "missing_slots": p.get("missing_slots", []) or [],
                "default_route": p.get("default_route"),
            })
        system = """
你是銀行 HR AI 助理的 Runtime Query Understanding 模組。
你的任務是做 structured classification，不是回答問題。
請根據 concept taxonomy、query pattern schema 與 risk policy，判斷使用者問題的 category、intent、matched_concepts、matched_query_pattern_ids、是否模糊、missing_slots、candidate risk。
你可以做口語/錯字 normalization，也可以提供 optional_rewrite_terms，但不得自行創造法條號或內規條號；法條/內規條號會由 offline rewrite_rules 提供。
重要：risk_level / recommended_route 只是 signal，最終路由會由 deterministic guardrails 決定。
只能輸出 valid JSON，不要 markdown，不要解釋。
"""
        user = f"""
使用者問題：{question}

可用 categories：general, working_hours, attendance, leave, overtime, salary, welfare, termination, occupational_accident, privacy_sensitive, high_risk, governance
可用 answer_policy：direct, with_disclaimer, clarify, escalate
可用 relation types：has_rule, related_to, refers_to, supplements, overrides, parent_of, child_of

Offline concept taxonomy（節錄）：
{json.dumps(concepts_for_prompt, ensure_ascii=False)}

Offline query patterns（節錄）：
{json.dumps(patterns_for_prompt, ensure_ascii=False)}

Offline risk policies（節錄）：
{json.dumps(risk_for_prompt, ensure_ascii=False)}

Heuristic baseline：
{json.dumps(base, ensure_ascii=False, default=str)}

請輸出 JSON schema：
{{
  "intent": "...",
  "category": "leave/overtime/salary/welfare/termination/working_hours/attendance/occupational_accident/privacy_sensitive/high_risk/general/governance",
  "matched_concepts": ["concept_id"],
  "matched_query_pattern_ids": ["pattern_id"],
  "is_ambiguous": true/false,
  "missing_slots": ["..."],
  "normalized_query": "口語或錯字修正後的問題，不要新增法條號",
  "risk_level": "低/中/高",
  "risk_reasons": ["..."],
  "answer_policy": "direct/with_disclaimer/clarify/escalate",
  "recommended_route": "answer/disclaimer/clarify/escalate",
  "preferred_relations": ["has_rule/related_to/refers_to/supplements/overrides"],
  "optional_rewrite_terms": ["最多5個語意補充詞，不要放法條號或內規條號"],
  "confidence": 0.0
}}
"""
        parsed = call_llm_json(system, user, default={})
        if not isinstance(parsed, dict) or not parsed:
            return base
        merged = dict(base)
        # Accept category/intent/concepts/ambiguity from local LLM.
        if parsed.get("intent"):
            merged["intent"] = str(parsed["intent"])
        if parsed.get("category"):
            merged["category"] = str(parsed["category"])
        if isinstance(parsed.get("matched_concepts"), list):
            merged["matched_concepts"] = list(dict.fromkeys([str(x) for x in parsed["matched_concepts"] if x]))[:8]
        if isinstance(parsed.get("matched_query_pattern_ids"), list):
            merged["matched_query_pattern_ids"] = list(dict.fromkeys([str(x) for x in parsed["matched_query_pattern_ids"] if x]))[:8]
        if isinstance(parsed.get("missing_slots"), list):
            merged["missing_slots"] = list(dict.fromkeys((base.get("missing_slots", []) or []) + [str(x) for x in parsed["missing_slots"] if x]))[:8]
        if parsed.get("normalized_query"):
            merged["normalized_query"] = str(parsed.get("normalized_query"))
        if isinstance(parsed.get("preferred_relations"), list):
            merged["preferred_relations"] = [str(x) for x in parsed["preferred_relations"] if x][:6]
        # Risk is stored as a signal. Final guardrails can override.
        merged["llm_risk_signal"] = {
            "risk_level": parsed.get("risk_level"),
            "risk_reasons": parsed.get("risk_reasons", []),
            "recommended_route": parsed.get("recommended_route"),
            "answer_policy": parsed.get("answer_policy"),
            "confidence": parsed.get("confidence"),
        }
        # Let LLM identify ambiguity, but final policy still goes through guardrails.
        if parsed.get("is_ambiguous") is True:
            merged["answer_policy"] = "clarify"
        elif parsed.get("answer_policy") in ["direct", "with_disclaimer", "clarify", "escalate"]:
            # Do not let LLM lower a heuristic high-risk/escalate decision.
            current_route = policy_to_route(merged.get("answer_policy"))
            proposed_route = policy_to_route(parsed.get("answer_policy"))
            priority = {"answer": 0, "disclaimer": 1, "clarify": 2, "escalate": 3}
            if priority.get(proposed_route, 0) >= priority.get(current_route, 0):
                merged["answer_policy"] = route_to_policy(proposed_route)
        # Hybrid query rewrite: offline mandatory terms + local LLM optional semantic terms.
        optional_terms = parsed.get("optional_rewrite_terms") if isinstance(parsed.get("optional_rewrite_terms"), list) else []
        merged["optional_rewrite_terms"] = [str(t) for t in optional_terms if t][:LOCAL_LLM_OPTIONAL_REWRITE_MAX_TERMS]
        fallback_rewrite = base.get("rewritten_query", question)
        merged["rewritten_query"] = self._rewrite_from_artifacts(
            question=question,
            matched_concepts=merged.get("matched_concepts", []) or [],
            category=merged.get("category", base.get("category", "general")),
            fallback=fallback_rewrite,
            normalized_query=merged.get("normalized_query"),
            optional_terms=merged.get("optional_rewrite_terms", []),
        )
        return merged

    def _node_query_understanding(self, state: HRState) -> HRState:
        q = state["question"]
        base = self._heuristic_understanding(q)
        if USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING:
            base = self._llm_runtime_understanding(q, base)
        # Refresh risk matches after local LLM concept/category changes; never remove existing high-risk matches.
        risk_matches = self._match_risk_policies(q)
        if risk_matches:
            base["risk_matches"] = risk_matches
        stage_log(
            "query_understanding",
            f"category={base.get('category')} intent={base.get('intent')} risk={base.get('risk_level')} "
            f"concepts={len(base.get('matched_concepts', []) or [])} "
            f"risk_matches={len(base.get('risk_matches', []) or [])} "
            f"missing_slots={base.get('missing_slots', []) or []}",
            preview=f"rewritten: {base.get('rewritten_query', q)}",
        )
        return {**base, "debug": add_debug(state, f"Query understanding: {base}")}

    def _node_retrieval_orchestrator(self, state: HRState) -> HRState:
        q = state.get("rewritten_query", state["question"])
        cat = state.get("category", "general")
        chunks = self.retriever.search(q, category=cat, top_k=8)
        seed_ids = []
        for c in chunks[:5]:
            aid = c.get("parent_id") if c.get("chunk_type") in ["semantic", "faq"] else c.get("article_id")
            if aid:
                seed_ids.append(aid)
            for rel in c.get("related_articles", []):
                seed_ids.append(rel)
        graph_ctx = self.kg.expand(seed_ids, state["question"], hops=1, max_nodes=14, preferred_relations=state.get("preferred_relations", []))
        top = chunks[0] if chunks else {}
        stage_log(
            "retrieval",
            f"chunks={len(chunks)} graph_nodes={len(graph_ctx.get('nodes', []))} "
            f"graph_edges={len(graph_ctx.get('edges', []))} use_graph={graph_ctx.get('use_graph')}",
            preview=(f"top: [{top.get('source_type')}] {top.get('article_no')} {top.get('title')} "
                     f"score={top.get('final_score')}") if top else "no chunks retrieved",
        )
        return {
            "retrieved_chunks": chunks,
            "graph_context": graph_ctx,
            "debug": add_debug(state, f"Retrieved {len(chunks)} chunks; graph_nodes={len(graph_ctx.get('nodes', []))}, graph_edges={len(graph_ctx.get('edges', []))}"),
        }

    def _node_guardrails_and_route(self, state: HRState) -> HRState:
        policy = state.get("answer_policy", "direct")
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            confidence = 0.0
        else:
            top = float(chunks[0].get("final_score", 0))
            second = float(chunks[1].get("final_score", 0)) if len(chunks) > 1 else 0
            confidence = min(1.0, max(0.0, top + max(0, top-second)*0.15))

        risk_matches = state.get("risk_matches", []) or []
        llm_signal = state.get("llm_risk_signal", {}) or {}
        missing_slots = state.get("missing_slots", []) or []

        # Final route is deterministic. Risk policy can only raise risk / route, not lower it.
        high_risk_policy_hit = any(str(r.get("risk_level", "")).lower() in ["high", "高"] or r.get("default_route") == "escalate" for r in risk_matches)
        llm_high_risk = str(llm_signal.get("risk_level", "")).lower() in ["high", "高"] or llm_signal.get("recommended_route") == "escalate"
        llm_medium_risk = str(llm_signal.get("risk_level", "")).lower() in ["medium", "中"] or llm_signal.get("recommended_route") == "disclaimer"

        if high_risk_policy_hit:
            route = "escalate"
            final_risk_level = "高"
        elif llm_high_risk:
            route = "escalate"
            final_risk_level = "高"
        elif policy == "clarify" or missing_slots:
            route = "clarify"
            final_risk_level = "中" if missing_slots else state.get("risk_level", "低")
        elif policy == "escalate":
            route = "escalate"
            final_risk_level = "高"
        elif confidence < 0.18:
            route = "escalate"
            final_risk_level = "中"
        elif policy == "with_disclaimer" or llm_medium_risk:
            route = "disclaimer"
            final_risk_level = "中"
        else:
            route = "answer"
            final_risk_level = state.get("risk_level", "低")

        stage_log(
            "guardrails",
            f"policy={policy} confidence={confidence:.4f} → route={route} risk={final_risk_level} "
            f"(risk_matches={len(risk_matches)} missing_slots={len(missing_slots)})",
        )
        return {
            "confidence": round(confidence, 4),
            "route": route,
            "risk_level": final_risk_level,
            "debug": add_debug(state, f"Guardrails: policy={policy}, risk_matches={len(risk_matches)}, llm_signal={llm_signal}, confidence={confidence:.4f}, route={route}"),
        }

    def _make_context(self, state: HRState, max_chars: int = 900) -> str:
        lines = []
        for i, c in enumerate(state.get("retrieved_chunks", [])[:6], start=1):
            lines.append(f"""
[S{i}]
chunk_type: {c.get('chunk_type')}
source_type: {c.get('source_type')}
document: {c.get('document_name')}
article_no: {c.get('article_no')}
title: {c.get('title')}
category: {c.get('category')}
priority: {c.get('priority')}
content: {c.get('content')[:max_chars]}
""".strip())
        graph_context = state.get("graph_context", {}).get("context", "")
        if graph_context:
            lines.append("\n[Graph-enhanced Context]\n" + graph_context[:2500])
        return "\n\n---\n\n".join(lines)

    def _build_citations(self, state: HRState) -> List[Dict[str, Any]]:
        cites = []
        for i, c in enumerate(state.get("retrieved_chunks", [])[:6], start=1):
            cites.append({
                "source_id": f"S{i}",
                "chunk_type": c.get("chunk_type"),
                "source_type": c.get("source_type"),
                "document_name": c.get("document_name"),
                "article_no": c.get("article_no"),
                "title": c.get("title"),
                "category": c.get("category"),
                "score": c.get("final_score"),
                "content_preview": c.get("content", "")[:180],
            })
        return cites

    def _fallback_answer(self, state: HRState, disclaimer: bool = False) -> str:
        q = state["question"]
        chunks = state.get("retrieved_chunks", [])[:4]
        if not chunks:
            return "目前知識庫沒有找到足夠依據可回答，建議洽 HR 確認。"
        top_internal = [c for c in chunks if c.get("source_type") == "internal_policy"]
        top_law = [c for c in chunks if c.get("source_type") == "law"]
        lines = []
        lines.append("簡短結論：")
        if top_internal:
            lines.append("依目前知識庫，應優先參考公司內部規章；若內規未明確規定，再參考勞動基準法作為最低標準。")
        else:
            lines.append("依目前檢索結果，以下為相關法規或規章整理。")
        lines.append("\n適用條件：")
        lines.append("需依實際身分、年資、班表、核准流程與公司最新公告確認。")
        lines.append("\n依據 Citation：")
        for i, c in enumerate(chunks, start=1):
            lines.append(f"- [S{i}] {c.get('source_type')}｜{c.get('document_name')}｜{c.get('article_no')}｜{c.get('title')}")
        if state.get("graph_context", {}).get("edges"):
            lines.append("\n規範差異 / Graph 關係：")
            for e in state["graph_context"]["edges"][:5]:
                lines.append(f"- {e['source']} --{e['relation']}--> {e['target']}")
        lines.append("\n白話說明：")
        for i, c in enumerate(chunks[:3], start=1):
            content = c.get("answer") or c.get("content", "")
            lines.append(f"- [S{i}] {content[:260]}...")
        lines.append("\n注意事項 / 聲明：")
        if disclaimer:
            lines.append("本回答依據現行知識庫提供一般性說明，實際適用仍需依個案情況與人力資源單位最終認定為準。")
        else:
            lines.append("若涉及個人薪資、主管指示、申訴爭議或公司最新公告，建議洽 HR 確認。")
        lines.append("\n下一步建議：")
        lines.append("若仍不確定，請提供更完整情境或轉 HR 人員確認。")
        return "\n".join(lines)

    def _node_generate_answer(self, state: HRState) -> HRState:
        context = self._make_context(state)
        disclaimer = state.get("route") == "disclaimer"
        if not USE_LLM:
            answer = self._fallback_answer(state, disclaimer=disclaimer)
        else:
            n_src = min(6, len(state.get("retrieved_chunks", [])))
            src_list = "、".join(f"[S{i}]" for i in range(1, n_src + 1)) or "（無可用來源）"
            system = f"""
你是安久銀行 HR AI 智能助理。
你只能根據提供的 Retrieval Context 與 Graph Context 回答，不得自行編造資料。
請務必一律使用「繁體中文（台灣用語）」回答，不得使用簡體字。
回答規則：
1. 優先使用 internal_policy，其次以 law 作為最低標準。
2. 每一段只要用到來源，句末必須標註對應編號，格式 [S1]、[S2]；可用來源僅限：{src_list}。
3. 直接在每個標題後填入「實際內容」，不要只列空白標題，也不要重印這份格式或規則說明。
4. 至少要在「依據 Citation」與「白話說明」兩段標註來源編號。
5. 若內規優於法規，請在「規範差異」段說明差異並各自標註來源。
6. 情境型問題需加風險聲明；不得做法律判定，高風險或個案爭議建議洽 HR。
"""
            user = f"""
員工問題：{state['question']}
Intent: {state.get('intent')}
Category: {state.get('category')}
Risk: {state.get('risk_level')}
Route: {state.get('route')}

Retrieval / Graph Context:
{context}

請直接在每個標題後填入內容（可同行或換行），每個有依據的段落都要標 [S#]：
簡短結論：
適用條件：
依據 Citation：
規範差異（內規 vs 法規，如有）：
白話說明：
注意事項 / 聲明：
下一步建議：
"""
            answer = call_llm_text(system, user, temperature=0.1)
            # Retry once if the model forgot to cite any [S#] source.
            if answer and "[S" not in answer:
                stage_log("generate_answer", "回答未標註 [S#]，以更嚴格指示重生一次")
                retry_hint = "\n注意：上一次回答未標註任何 [S#] 來源，請重寫，並確保每個有依據的段落都標註 [S1]、[S2] 等來源編號，且不要重印格式標題。"
                retry = call_llm_text(system + retry_hint, user, temperature=0.0)
                if retry and "[S" in retry:
                    answer = retry
            if not answer:
                answer = self._fallback_answer(state, disclaimer=disclaimer)
        citations = self._build_citations(state)
        stage_log(
            "generate_answer",
            f"route={state.get('route')} llm={'on' if USE_LLM else 'off(template)'} "
            f"answer_len={len(answer)} citations={len(citations)}",
            preview=answer,
        )
        return {"answer": answer, "citations": citations, "debug": add_debug(state, "Generated answer")}

    def _node_clarify(self, state: HRState) -> HRState:
        missing_slots = state.get("missing_slots", []) or []
        category = state.get("category", "general")
        questions = []
        for p in self.artifacts.query_patterns:
            if p.get("category") == category and p.get("pattern_type") == "ambiguous":
                questions.extend(p.get("clarification_questions", []) or [])
        if not questions:
            questions = [
                "請問你想查詢的是特休、病假、事假、加班、補休、薪資、資遣或福利？",
                "是否涉及個人薪資、主管核准、特殊班表或申訴爭議？",
                "需要查詢公司內規，還是只想了解勞基法最低規定？",
            ]
        questions = list(dict.fromkeys(questions))[:4]
        slot_text = "、".join(missing_slots) if missing_slots else "假別、期間、原因、是否涉及主管核准或個案爭議"
        answer = f"""
我需要再確認一些資訊，才能避免誤判。

您的問題是：「{state['question']}」

目前判斷類別：{state.get('category')}｜可能缺少資訊：{slot_text}

請補充以下資訊：
""".strip()
        for i, qu in enumerate(questions, start=1):
            answer += f"\n{i}. {qu}"
        answer += "\n\n在資訊不足時，系統不會直接推論答案，以降低 HR 法規誤判風險。"
        stage_log(
            "clarify",
            f"missing_slots={missing_slots or '—'} clarify_questions={len(questions)}",
            preview=answer,
        )
        return {"answer": answer, "citations": [], "faithfulness_score": 1.0, "debug": add_debug(state, "Clarification generated")}

    def _node_escalate(self, state: HRState) -> HRState:
        chunks = state.get("retrieved_chunks", [])[:3]
        refs = "\n".join([f"- [S{i+1}] {c.get('source_type')}｜{c.get('article_no')}｜{c.get('title')}" for i, c in enumerate(chunks)])
        answer = f"""
目前此問題不適合由 AI 直接判定，建議轉由 HR 或法遵人員處理。

問題：{state['question']}
判斷類型：{state.get('intent')}
風險等級：{state.get('risk_level')}

原因：
1. 問題可能涉及個案判斷、申訴爭議、薪資明細、主管處置、個資或法律責任。
2. AI 可提供一般規範整理，但不應直接做合法性或責任歸屬判斷。
3. 本題命中的風險政策：{', '.join([r.get('risk_policy_id','') for r in state.get('risk_matches', [])]) or '無明確風險政策，但路由判斷為高風險或低信心'}。

可能相關依據：
{refs if refs else '目前無足夠相關依據。'}

下一步建議：請洽 HR 服務窗口或依公司正式申訴 / 諮詢流程處理。
""".strip()
        stage_log(
            "escalate",
            f"risk={state.get('risk_level')} "
            f"risk_policies={[r.get('risk_policy_id', '') for r in state.get('risk_matches', [])] or '—'}",
            preview=answer,
        )
        return {"answer": answer, "citations": self._build_citations(state), "faithfulness_score": 1.0, "debug": add_debug(state, "Escalation generated")}

    def _node_faithfulness_check(self, state: HRState) -> HRState:
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        has_citation = bool(re.search(r"\[S\d+\]", answer))
        mentions_article = any(str(c.get("article_no", "")).replace(" ", "") in answer.replace(" ", "") for c in citations if c.get("article_no"))
        score = 0.82
        if has_citation: score += 0.10
        if mentions_article: score += 0.05
        if state.get("route") in ["escalate", "clarify"]: score = max(score, 0.95)
        score = min(1.0, score)
        stage_log("faithfulness_check", f"score={score:.4f} has_citation={has_citation} mentions_article={mentions_article}")
        return {"faithfulness_score": round(score, 4), "debug": add_debug(state, f"Faithfulness={score:.4f}")}

    def _route_after_guardrails(self, state: HRState) -> Literal["answer", "disclaimer", "escalate", "clarify"]:
        return state.get("route", "answer")

    def _build_graph(self):
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(HRState)
        graph.add_node("query_understanding", self._node_query_understanding)
        graph.add_node("retrieval_orchestrator", self._node_retrieval_orchestrator)
        graph.add_node("guardrails", self._node_guardrails_and_route)
        graph.add_node("generate_answer", self._node_generate_answer)
        graph.add_node("clarify", self._node_clarify)
        graph.add_node("escalate", self._node_escalate)
        graph.add_node("faithfulness_check", self._node_faithfulness_check)

        graph.add_edge(START, "query_understanding")
        graph.add_edge("query_understanding", "retrieval_orchestrator")
        graph.add_edge("retrieval_orchestrator", "guardrails")
        graph.add_conditional_edges(
            "guardrails",
            self._route_after_guardrails,
            {
                "answer": "generate_answer",
                "disclaimer": "generate_answer",
                "escalate": "escalate",
                "clarify": "clarify",
            },
        )
        graph.add_edge("generate_answer", "faithfulness_check")
        graph.add_edge("faithfulness_check", END)
        graph.add_edge("clarify", END)
        graph.add_edge("escalate", END)
        return graph.compile()

    def ask(self, question: str) -> HRState:
        stage_log("ask", "處理新問題", preview=question)
        return self.app.invoke({"question": question})

# -----------------------------
# 9. Evaluation + Feedback
# -----------------------------
