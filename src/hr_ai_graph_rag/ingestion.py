# ============================================================
# ingestion — DOCX 解析 + 階層式切塊 + 建知識
#
# 讀取勞基法/內規 DOCX、產生 sample 法條、政策感知的階層式 chunking,
# 以及 HRKnowledgeBuilder(articles/chunks 建構)。
# 依賴:config、utils。
# ============================================================

from .config import *
from .utils import *


def read_docx_text(path: str) -> str:
    doc = Document(path)
    blocks = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            blocks.append(txt)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(blocks)


def create_sample_labor_law_docx(path: str) -> str:
    doc = Document()
    doc.add_heading("參考資料_勞動基準法 Sample", level=1)
    sample_articles = [
        ("第一章 總則", "第 2 條", "本法用詞，定義如下：勞工指受雇主僱用從事工作獲致工資者。工資指勞工因工作而獲得之報酬。"),
        ("第二章 勞動契約", "第 16 條", "雇主依規定終止勞動契約者，其預告期間依勞工工作年資而定。三個月以上一年未滿者十日前預告；一年以上三年未滿者二十日前預告；三年以上者三十日前預告。"),
        ("第三章 工資", "第 22 條", "工資應全額直接給付勞工。"),
        ("第三章 工資", "第 24 條", "雇主延長勞工工作時間者，延長工作時間在二小時以內者，按平日每小時工資額加給三分之一以上；再延長工作時間在二小時以內者，按平日每小時工資額加給三分之二以上。"),
        ("第四章 工作時間、休息、休假", "第 30 條", "勞工正常工作時間，每日不得超過八小時，每週不得超過四十小時。"),
        ("第四章 工作時間、休息、休假", "第 32 條", "雇主有使勞工在正常工作時間以外工作之必要者，經工會同意，如事業單位無工會者，經勞資會議同意後，得將工作時間延長之。"),
        ("第四章 工作時間、休息、休假", "第 36 條", "勞工每七日中應有二日之休息，其中一日為例假，一日為休息日。"),
        ("第四章 工作時間、休息、休假", "第 38 條", "勞工在同一雇主或事業單位，繼續工作滿一定期間者，應給予特別休假。六個月以上一年未滿者三日；一年以上二年未滿者七日；二年以上三年未滿者十日。"),
        ("第七章 職業災害補償", "第 59 條", "勞工因遭遇職業災害而致死亡、失能、傷害或疾病時，雇主應依規定予以補償。"),
    ]
    current_chapter = None
    for chapter, article, content in sample_articles:
        if chapter != current_chapter:
            doc.add_heading(chapter, level=2)
            current_chapter = chapter
        doc.add_paragraph(f"{article} {content}")
    doc.save(path)
    return path

CHAPTER_PATTERN = re.compile(rf"^第\s*[\d{CJK_NUM}]+\s*章.*")
ARTICLE_PATTERN = re.compile(
    rf"^(第\s*[\d{CJK_NUM}]+(?:\s*-\s*[\d{CJK_NUM}]+)?\s*條(?:\s*之\s*[\d{CJK_NUM}]+)?)[\s：:、]*(.*)"
)
POLICY_ARTICLE_PATTERN = re.compile(r"^(POLICY[-_][A-Za-z0-9]+|內規[-_][A-Za-z0-9]+|[A-Za-z]+[-_]\d+)[\s：:、]*(.*)", re.I)


@dataclass
class ChunkConfig:
    version: str = "v1.0"
    effective_date: str = "2026-06-23"


class HRKnowledgeBuilder:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def parse_labor_law_articles(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        articles = []
        current_chapter = "未標示章節"
        current_article_no = None
        current_content = []

        def flush():
            if current_article_no and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    article_no = normalize_article_no(current_article_no)
                    articles.append({
                        "source_type": "law",
                        "document_name": document_name,
                        "article_id": article_id_from_no("law", article_no),
                        "article_no": article_no,
                        "chapter": current_chapter,
                        "title": "",
                        "category": detect_category(content),
                        "priority": 1,
                        "version": self.config.version,
                        "effective_date": self.config.effective_date,
                        "content": content,
                        "related_articles": [],
                    })

        for line in lines:
            if CHAPTER_PATTERN.match(line):
                current_chapter = line
                continue
            m = ARTICLE_PATTERN.match(line)
            if m:
                flush()
                current_article_no = m.group(1)
                rest = m.group(2).strip()
                current_content = [f"{normalize_article_no(current_article_no)} {rest}".strip()]
            else:
                if current_article_no:
                    current_content.append(line)
        flush()

        # fallback: fixed chunks if article parser fails
        if not articles:
            raw = text
            size, overlap = 900, 120
            start = 0
            while start < len(raw):
                content = raw[start:start+size]
                article_no = f"chunk_{len(articles)+1}"
                articles.append({
                    "source_type": "law",
                    "document_name": document_name,
                    "article_id": f"law_chunk_{len(articles)+1}",
                    "article_no": article_no,
                    "chapter": "未標示章節",
                    "title": "",
                    "category": detect_category(content),
                    "priority": 1,
                    "version": self.config.version,
                    "effective_date": self.config.effective_date,
                    "content": content,
                    "related_articles": [],
                })
                start += size - overlap
        return articles

    def parse_internal_policy_articles(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Parse user-provided internal policy DOCX into article-level structured records.

        Expected heading examples:
        - 第 11 條｜特別休假
        - 第11條 特別休假
        - POLICY-LEAVE-001 特別休假

        The parser intentionally skips table-of-contents entries that have no body text.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        articles = []
        current_chapter = "未標示章節"
        current_article_no = None
        current_title = ""
        current_content = []

        internal_article_pattern = re.compile(
            rf"^(第\s*[\d{CJK_NUM}]+(?:\s*-\s*[\d{CJK_NUM}]+)?\s*條(?:\s*之\s*[\d{CJK_NUM}]+)?|POLICY[-_][A-Za-z0-9]+|內規[-_][A-Za-z0-9]+)[\s｜|：:、]*(.*)$",
            re.I,
        )

        def flush():
            if not current_article_no:
                return
            body = "\n".join(current_content).strip()
            # Skip TOC-only headings or empty articles.
            if not body or len(body) < 8:
                return
            article_no = normalize_article_no(current_article_no)
            title = normalize_spaces(current_title)
            content = f"{article_no} {title}\n{body}".strip()
            related = extract_related_law_ids(content)
            graph_edges = []
            for rid in related:
                graph_edges.append((relation_from_policy_content(content), rid))
            articles.append({
                "source_type": "internal_policy",
                "document_name": document_name,
                "article_id": article_id_from_no("internal_policy", article_no),
                "article_no": article_no,
                "chapter": current_chapter,
                "title": title,
                "category": detect_category(content + " " + title + " " + current_chapter),
                "priority": 2,
                "version": self.config.version,
                "effective_date": self.config.effective_date,
                "content": content,
                "related_articles": related,
                "graph_edges": graph_edges,
            })

        for line in lines:
            if CHAPTER_PATTERN.match(line):
                flush()
                current_chapter = line
                current_article_no = None
                current_title = ""
                current_content = []
                continue
            m = internal_article_pattern.match(line)
            if m:
                flush()
                current_article_no = m.group(1)
                current_title = m.group(2).strip()
                current_content = []
            else:
                if current_article_no:
                    current_content.append(line)
        flush()

        # Fallback: if the policy file is not article-formatted, create fixed article-like chunks.
        if not articles:
            raw = text
            size, overlap = 900, 120
            start = 0
            while start < len(raw):
                content = raw[start:start+size]
                article_no = f"policy_chunk_{len(articles)+1}"
                articles.append({
                    "source_type": "internal_policy",
                    "document_name": document_name,
                    "article_id": f"policy_chunk_{len(articles)+1}",
                    "article_no": article_no,
                    "chapter": "未標示章節",
                    "title": "",
                    "category": detect_category(content),
                    "priority": 2,
                    "version": self.config.version,
                    "effective_date": self.config.effective_date,
                    "content": content,
                    "related_articles": extract_related_law_ids(content),
                    "graph_edges": [],
                })
                start += size - overlap
        return articles

    def make_document_chunks(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_doc = defaultdict(list)
        for a in articles:
            by_doc[a["document_name"]].append(a)
        chunks = []
        for doc_name, arr in by_doc.items():
            source_type = arr[0]["source_type"]
            content = "\n".join([f"{a['article_no']} {a['title']}：{a['content'][:160]}" for a in arr[:20]])
            cats = sorted(set(a["category"] for a in arr))
            chunks.append({
                "chunk_id": f"doc_{len(chunks)+1:04d}",
                "chunk_type": "document",
                "source_type": source_type,
                "document_name": doc_name,
                "article_id": f"document::{doc_name}",
                "article_no": "DOCUMENT",
                "chapter": "Document-level",
                "title": f"{doc_name} 文件摘要",
                "category": ",".join(cats),
                "priority": max(a["priority"] for a in arr),
                "version": self.config.version,
                "effective_date": self.config.effective_date,
                "content": content,
                "parent_id": None,
                "related_articles": [],
                "keywords": extract_keywords(content),
            })
        return chunks

    def make_article_chunks(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for a in articles:
            chunks.append({
                "chunk_id": f"article_{len(chunks)+1:04d}",
                "chunk_type": "article",
                "source_type": a["source_type"],
                "document_name": a["document_name"],
                "article_id": a["article_id"],
                "article_no": a["article_no"],
                "chapter": a["chapter"],
                "title": a.get("title", ""),
                "category": a["category"],
                "priority": a["priority"],
                "version": a["version"],
                "effective_date": a["effective_date"],
                "content": a["content"],
                "parent_id": f"document::{a['document_name']}",
                "related_articles": a.get("related_articles", []),
                "keywords": extract_keywords(a["content"]),
            })
        return chunks

    def make_semantic_subchunks(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for a in articles:
            sentences = split_sentences_zh(a["content"])
            # group every 1-2 sentences if long
            groups = []
            buf = []
            for s in sentences:
                buf.append(s)
                if len("".join(buf)) >= 120:
                    groups.append("".join(buf))
                    buf = []
            if buf:
                groups.append("".join(buf))
            if len(groups) <= 1 and len(a["content"]) < 220:
                continue
            for j, g in enumerate(groups, start=1):
                chunks.append({
                    "chunk_id": f"semantic_{len(chunks)+1:04d}",
                    "chunk_type": "semantic",
                    "source_type": a["source_type"],
                    "document_name": a["document_name"],
                    "article_id": f"{a['article_id']}::s{j}",
                    "article_no": a["article_no"],
                    "chapter": a["chapter"],
                    "title": a.get("title", ""),
                    "category": a["category"],
                    "priority": a["priority"],
                    "version": a["version"],
                    "effective_date": a["effective_date"],
                    "content": g,
                    "parent_id": a["article_id"],
                    "related_articles": a.get("related_articles", []),
                    "keywords": extract_keywords(g),
                })
        return chunks

    def make_faq_chunks(self, articles: List[Dict[str, Any]], golden_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Build FAQ chunks only when explicitly enabled.

        Default is empty because Golden Dataset is evaluation data and should not be leaked
        into the retrieval knowledge base. If USE_GOLDEN_AS_FAQ_CHUNKS=true, the script
        converts user-provided Golden Dataset questions into experimental FAQ chunks.
        """
        if not USE_GOLDEN_AS_FAQ_CHUNKS or golden_df is None or golden_df.empty:
            return []
        chunks = []
        for _, row in golden_df.iterrows():
            q = str(row.get("question", "")).strip()
            if not q:
                continue
            key_points = row.get("expected_key_points", [])
            if isinstance(key_points, str):
                answer = key_points
            elif isinstance(key_points, list):
                answer = "；".join(map(str, key_points))
            else:
                answer = ""
            category = str(row.get("expected_category", detect_category(q)) or detect_category(q))
            content = f"FAQ 問題：{q}\nFAQ 回答重點：{answer}"
            chunks.append({
                "chunk_id": f"faq_{len(chunks)+1:04d}",
                "chunk_type": "faq",
                "source_type": "golden_dataset_faq_experiment",
                "document_name": "Golden Dataset derived FAQ chunks - experimental",
                "article_id": f"faq::golden::{row.get('id', len(chunks)+1)}",
                "article_no": str(row.get("id", f"FAQ-{len(chunks)+1}")),
                "chapter": "FAQ Chunk",
                "title": q,
                "category": category,
                "priority": 1,
                "version": self.config.version,
                "effective_date": self.config.effective_date,
                "content": content,
                "question": q,
                "answer": answer,
                "parent_id": None,
                "related_articles": [],
                "keywords": extract_keywords(content),
            })
        return chunks

    def build_chunks(
        self,
        labor_law_docx_path: str,
        internal_policy_docx_path: str,
        golden_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # 1) External regulation: user-provided 勞動基準法 DOCX
        law_text = read_docx_text(labor_law_docx_path)
        law_articles = self.parse_labor_law_articles(law_text, Path(labor_law_docx_path).name)

        # 2) Internal policy: user-provided 模擬銀行內規 DOCX
        policy_text = read_docx_text(internal_policy_docx_path)
        policy_articles = self.parse_internal_policy_articles(policy_text, Path(internal_policy_docx_path).name)

        all_articles = law_articles + policy_articles

        doc_chunks = self.make_document_chunks(all_articles)
        article_chunks = self.make_article_chunks(all_articles)
        semantic_chunks = self.make_semantic_subchunks(all_articles)
        faq_chunks = self.make_faq_chunks(all_articles, golden_df=golden_df)

        chunks = doc_chunks + article_chunks + semantic_chunks + faq_chunks
        # ensure unique ids
        for i, c in enumerate(chunks, start=1):
            c["global_chunk_id"] = f"G{i:05d}"
            c["embedding_text"] = self.chunk_to_text(c)
        return all_articles, chunks

    @staticmethod
    def chunk_to_text(c: Dict[str, Any]) -> str:
        return f"""
Chunk Type: {c.get('chunk_type')}
Source Type: {c.get('source_type')}
Document: {c.get('document_name')}
Article: {c.get('article_no')}
Title: {c.get('title')}
Category: {c.get('category')}
Priority: {c.get('priority')}
Content: {c.get('content')}
Keywords: {', '.join(c.get('keywords', []))}
""".strip()

# -----------------------------
# 5. Knowledge Graph / Graph RAG
# -----------------------------
