# ============================================================
# retrieval — 混合檢索 + 重排序(HybridRetriever)
#
# FAISS dense + BM25 + metadata 混合檢索,後接 cross-encoder reranker。
# torch/faiss/sentence-transformers 於使用時才延後載入。
# 依賴:config、utils。
# ============================================================

from .config import *
from .utils import *


class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        use_reranker: bool = USE_RERANKER,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer
        self._faiss = faiss
        self.chunks = chunks
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.texts = [c["embedding_text"] for c in chunks]
        self.tokenized = [tokenize_zh(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)
        self.index = None
        self.embeddings = None
        self._build_vector_index()

        # Optional cross-encoder reranker. Loaded lazily; if it fails to load, retrieval
        # degrades gracefully to the hybrid (vector + BM25) ranking.
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading reranker model: {reranker_model_name}")
                self.reranker = CrossEncoder(reranker_model_name)
            except Exception as e:
                print("Reranker load failed; continuing without rerank.", repr(e))
                self.reranker = None

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Re-score the top hybrid candidates with the cross-encoder and blend scores."""
        if not self.reranker or not candidates:
            return candidates[:top_k]
        pool = candidates[:max(RERANK_CANDIDATES, top_k)]
        pairs = [[query, c.get("content", "") or c.get("embedding_text", "")] for c in pool]
        try:
            scores = self.reranker.predict(pairs, convert_to_numpy=True)
        except Exception as e:
            print("Rerank predict failed; using hybrid order.", repr(e))
            return candidates[:top_k]
        s = np.asarray(scores, dtype="float32")
        # Min-max normalize reranker scores so they blend with the hybrid final_score.
        if s.size > 1 and float(s.max() - s.min()) > 1e-9:
            s_norm = (s - s.min()) / (s.max() - s.min())
        else:
            s_norm = np.zeros_like(s)
        for c, raw, norm in zip(pool, scores, s_norm):
            hybrid = float(c.get("final_score", 0.0))
            c["rerank_score"] = round(float(raw), 4)
            c["pre_rerank_score"] = hybrid
            c["final_score"] = round(RERANK_WEIGHT * float(norm) + (1 - RERANK_WEIGHT) * hybrid, 4)
        pool = sorted(pool, key=lambda x: x["final_score"], reverse=True)
        return pool[:top_k]

    def _build_vector_index(self):
        print("Building embeddings and FAISS index...")
        emb = self.embedder.encode(
            self.texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.embeddings = emb
        self.index = self._faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        print("FAISS index size:", self.index.ntotal)

    def search(self, query: str, category: str = "general", top_k: int = 8) -> List[Dict[str, Any]]:
        query_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        search_k = min(max(top_k * 5, top_k), len(self.chunks))
        vector_scores, vector_ids = self.index.search(query_emb, search_k)

        bm25_scores = self.bm25.get_scores(tokenize_zh(query))
        if len(bm25_scores) > 0:
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        else:
            bm25_norm = np.zeros(len(self.chunks))

        candidate_ids = set(vector_ids[0].tolist())
        candidate_ids.update(np.argsort(-bm25_norm)[:search_k].tolist())
        keywords = extract_keywords(query)

        results = []
        for idx in candidate_ids:
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = dict(self.chunks[idx])
            vscore = 0.0
            if idx in vector_ids[0]:
                pos = list(vector_ids[0]).index(idx)
                vscore = float(vector_scores[0][pos])
            bscore = float(bm25_norm[idx])
            keyword_bonus = sum(0.02 for kw in keywords if kw in c.get("embedding_text", ""))
            category_bonus = 0.06 if category != "general" and c.get("category") == category else 0.0
            priority_bonus = 0.04 * float(c.get("priority", 1))
            faq_bonus = 0.05 if c.get("chunk_type") == "faq" else 0.0
            article_bonus = 0.03 if c.get("chunk_type") == "article" else 0.0

            final_score = 0.62 * vscore + 0.28 * bscore + keyword_bonus + category_bonus + priority_bonus + faq_bonus + article_bonus
            c.update({
                "vector_score": round(vscore, 4),
                "bm25_score": round(bscore, 4),
                "keyword_bonus": round(keyword_bonus, 4),
                "category_bonus": round(category_bonus, 4),
                "priority_bonus": round(priority_bonus, 4),
                "final_score": round(float(final_score), 4),
            })
            results.append(c)
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        # Cross-encoder rerank stage (no-op if reranker is disabled/unavailable).
        if self.reranker:
            return self._rerank(query, results, top_k)
        return results[:top_k]

# -----------------------------
# 7. HuggingFace Local LLM Helpers
# -----------------------------
