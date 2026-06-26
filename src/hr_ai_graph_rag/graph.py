# ============================================================
# graph — 生成知識圖(HRKnowledgeGraph)
#
# 由 articles/chunks + offline edge candidates 建立 networkx 知識圖,
# 提供 graph-enhanced retrieval 的鄰接擴展。
# 依賴:config、utils、artifacts(OfflineArtifacts 型別)。
# ============================================================

from .config import *
from .utils import *
from .artifacts import *


class HRKnowledgeGraph:
    def __init__(self, articles: List[Dict[str, Any]], chunks: List[Dict[str, Any]], artifacts: Optional[OfflineArtifacts] = None):
        self.articles = articles
        self.chunks = chunks
        self.artifacts = artifacts or OfflineArtifacts()
        self.G = nx.DiGraph()
        self._article_lookup = {a["article_id"]: a for a in articles}
        self._article_no_lookup = self._build_article_no_lookup()
        self._build_graph()

    def _build_article_no_lookup(self) -> Dict[Tuple[str, str], str]:
        lookup = {}
        for a in self.articles:
            source_type = a.get("source_type", "")
            no_norm = normalize_for_match(a.get("article_no", ""))
            if no_norm:
                lookup[(source_type, no_norm)] = a.get("article_id")
                lookup[("any", no_norm)] = a.get("article_id")
        return lookup

    def _resolve_article_ref(self, ref: str, preferred_source: Optional[str] = None) -> Optional[str]:
        # Direct article id
        if ref in self._article_lookup:
            return ref
        ref = str(ref or "")
        # Convert policy/law article hints like policy_article_11 / law_article_38
        m = re.search(r"(policy|law)_article_([0-9]+)", ref)
        if m:
            source = "internal_policy" if m.group(1) == "policy" else "law"
            no_norm = normalize_for_match(f"第{m.group(2)}條")
            return self._article_no_lookup.get((source, no_norm)) or self._article_no_lookup.get(("any", no_norm))
        # Convert human-readable ref like 安久銀行...第11條 or 勞動基準法第38條
        refs = extract_article_refs(ref)
        if refs:
            source = preferred_source
            if "勞動基準法" in ref or "勞基法" in ref:
                source = "law"
            elif "安久銀行" in ref or "內規" in ref or "規章" in ref:
                source = "internal_policy"
            for no_norm in refs:
                if source:
                    hit = self._article_no_lookup.get((source, no_norm))
                    if hit:
                        return hit
                hit = self._article_no_lookup.get(("any", no_norm))
                if hit:
                    return hit
        return None

    def _add_concept_nodes(self):
        if self.artifacts.concept_nodes:
            for c in self.artifacts.concept_nodes:
                cid = c.get("concept_id")
                if not cid:
                    continue
                self.G.add_node(
                    cid,
                    node_type="concept",
                    label=c.get("label", cid),
                    category=c.get("category", "general"),
                    risk_level=c.get("risk_level", "low"),
                    default_answer_policy=c.get("default_answer_policy", "answer"),
                    content=c.get("description", c.get("label", cid)),
                    aliases="｜".join(c.get("aliases", []) or []),
                    graph_expansion_priority=c.get("graph_expansion_priority", "medium"),
                )
                parent = c.get("parent_concept_id")
                if parent:
                    self.G.add_edge(parent, cid, relation="parent_of")
                    self.G.add_edge(cid, parent, relation="child_of")
            return

        # Fallback concept skeleton if no artifact is provided.
        concepts = {
            "concept_leave": "請假制度",
            "concept_special_leave": "特別休假",
            "concept_overtime": "加班",
            "concept_comp_time": "補休",
            "concept_working_hours": "工時",
            "concept_salary": "薪資",
            "concept_welfare": "福利",
            "concept_termination": "離職與資遣",
        }
        for cid, label in concepts.items():
            self.G.add_node(cid, node_type="concept", label=label, content=label)

    def _add_concept_article_edges(self):
        # Artifact-driven concept-to-article mapping
        for c in self.artifacts.concept_nodes:
            cid = c.get("concept_id")
            if not cid or cid not in self.G:
                continue
            for ref in c.get("related_law_articles", []) or []:
                aid = self._resolve_article_ref(ref, preferred_source="law")
                if aid:
                    self.G.add_edge(cid, aid, relation="has_rule")
                    self.G.add_edge(aid, cid, relation="related_to")
            for ref in c.get("related_policy_articles", []) or []:
                aid = self._resolve_article_ref(ref, preferred_source="internal_policy")
                if aid:
                    self.G.add_edge(cid, aid, relation="has_rule")
                    self.G.add_edge(aid, cid, relation="related_to")

        # Fallback category-related edges if no concept artifacts are available.
        if not self.artifacts.concept_nodes:
            concept_by_cat = {
                "leave": "concept_leave",
                "overtime": "concept_overtime",
                "attendance": "concept_working_hours",
                "salary": "concept_salary",
                "welfare": "concept_welfare",
                "termination": "concept_termination",
            }
            for a in self.articles:
                concept = concept_by_cat.get(a["category"])
                if concept:
                    self.G.add_edge(a["article_id"], concept, relation="related_to")
                    self.G.add_edge(concept, a["article_id"], relation="has_rule")

    def _add_artifact_edge_candidates(self):
        allowed_relations = {r.get("relation_type") for r in self.artifacts.relation_types if r.get("runtime_expandable", True)}
        if not allowed_relations:
            allowed_relations = {"has_rule", "related_to", "refers_to", "supplements", "overrides", "parent_of", "child_of"}
        for e in self.artifacts.graph_edge_candidates:
            status = e.get("review_status", "pending")
            if status != "approved" and not LOAD_PENDING_GRAPH_EDGES:
                continue
            rel = e.get("relation_type", "related_to")
            if rel not in allowed_relations:
                continue
            src = self._resolve_article_ref(e.get("source_node", "")) or e.get("source_node")
            tgt = self._resolve_article_ref(e.get("target_node", "")) or e.get("target_node")
            if src in self.G and tgt in self.G:
                self.G.add_edge(
                    src, tgt,
                    relation=rel,
                    evidence=e.get("evidence", ""),
                    confidence=e.get("confidence", None),
                    review_status=status,
                    human_review_required=e.get("human_review_required", False),
                )

    def _build_graph(self):
        self._add_concept_nodes()
        for a in self.articles:
            self.G.add_node(
                a["article_id"],
                node_type="law_article" if a["source_type"] == "law" else "internal_policy_article",
                label=f"{a['article_no']} {a.get('title','')}",
                source_type=a["source_type"],
                article_no=a["article_no"],
                category=a["category"],
                priority=a["priority"],
                content=a["content"],
            )
            # direct related articles parsed from document text
            for rel in a.get("related_articles", []):
                if rel:
                    self.G.add_edge(a["article_id"], rel, relation="refers_to")
            for relation, target in a.get("graph_edges", []):
                self.G.add_edge(a["article_id"], target, relation=relation)

        self._add_concept_article_edges()
        self._add_artifact_edge_candidates()

    def expand(self, seed_article_ids: List[str], question: str, hops: int = 1, max_nodes: int = 12, preferred_relations: Optional[List[str]] = None) -> Dict[str, Any]:
        # Runtime expansion is deterministic; local LLM can provide preferred relations, but traversal uses approved graph only.
        use_graph = any(k in question for k in ["差", "比較", "為什麼", "內規", "法規", "公司", "優於", "補休", "依據", "哪個", "關係"])
        if not use_graph and len(seed_article_ids) <= 0:
            return {"use_graph": False, "nodes": [], "edges": [], "context": ""}

        relation_filter = set(preferred_relations or [])
        nodes = []
        edges = []
        visited = set()
        frontier = [s for s in seed_article_ids if s in self.G]
        for s in frontier:
            visited.add(s)

        for _ in range(hops):
            new_frontier = []
            for u in frontier:
                neighbors = list(self.G.successors(u)) + list(self.G.predecessors(u))
                for v in neighbors:
                    candidate_edges = []
                    if self.G.has_edge(u, v):
                        candidate_edges.append((u, v, self.G[u][v].get("relation", "related_to")))
                    if self.G.has_edge(v, u):
                        candidate_edges.append((v, u, self.G[v][u].get("relation", "related_to")))
                    if relation_filter and not any(r in relation_filter for _, _, r in candidate_edges):
                        continue
                    if v not in visited:
                        visited.add(v)
                        new_frontier.append(v)
                    edges.extend(candidate_edges)
                    if len(visited) >= max_nodes:
                        break
                if len(visited) >= max_nodes:
                    break
            frontier = new_frontier
            if len(visited) >= max_nodes:
                break

        for n in list(visited)[:max_nodes]:
            data = self.G.nodes[n]
            nodes.append({
                "node_id": n,
                "node_type": data.get("node_type"),
                "label": data.get("label"),
                "article_no": data.get("article_no", ""),
                "source_type": data.get("source_type", ""),
                "category": data.get("category", ""),
                "risk_level": data.get("risk_level", ""),
                "default_answer_policy": data.get("default_answer_policy", ""),
                "content": data.get("content", "")[:500],
            })
        seen = set()
        edge_dicts = []
        for u, v, r in edges:
            key = (u, v, r)
            if key not in seen:
                seen.add(key)
                edata = self.G[u][v] if self.G.has_edge(u, v) else {}
                edge_dicts.append({"source": u, "target": v, "relation": r, "evidence": edata.get("evidence", "")})

        context_lines = []
        if nodes:
            context_lines.append("[Graph Nodes]")
            for n in nodes:
                context_lines.append(f"- {n['node_id']} ({n['node_type']}): {n['label']}｜{n['content']}")
        if edge_dicts:
            context_lines.append("[Graph Relations]")
            for e in edge_dicts:
                ev = f"｜evidence: {e['evidence']}" if e.get("evidence") else ""
                context_lines.append(f"- {e['source']} --{e['relation']}--> {e['target']}{ev}")
        return {
            "use_graph": use_graph,
            "nodes": nodes,
            "edges": edge_dicts,
            "context": "\n".join(context_lines),
        }

    def save_graph_files(self, output_dir: Path):
        nodes = []
        for n, d in self.G.nodes(data=True):
            rec = {"node_id": n, **d}
            nodes.append(rec)
        edges = []
        for u, v, d in self.G.edges(data=True):
            edges.append({"source": u, "target": v, **d})
        pd.DataFrame(nodes).to_csv(output_dir / "kg_nodes.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(edges).to_csv(output_dir / "kg_edges.csv", index=False, encoding="utf-8-sig")
        nx.write_gexf(self.G, output_dir / "hr_knowledge_graph.gexf")

# -----------------------------
# 6. Hybrid Retriever
# -----------------------------
