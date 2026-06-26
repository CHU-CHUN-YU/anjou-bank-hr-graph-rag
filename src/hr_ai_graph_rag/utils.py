# ============================================================
# utils — 零依賴文字 / 分類 / route 工具
#
# 純函式工具:中文正規化、tokenize、category 偵測、route/policy 轉換、
# 條號擷取等。刻意不依賴 package 內任何其他模組(最底層共用層)。
# ============================================================

import re
import json
from collections import defaultdict
from typing import Any, List, Dict, Optional


CJK_NUM = "一二三四五六七八九十百千零〇"

CATEGORY_KEYWORDS = {
    "leave": ["請假", "特休", "特別休假", "年假", "病假", "事假", "休假", "例假", "休息日", "補休", "假別"],
    "overtime": ["加班", "延長工時", "工時", "工作時間", "加班費", "補休", "換補休"],
    "salary": ["薪資", "薪水", "工資", "給付", "扣薪", "全勤", "獎金", "津貼"],
    "termination": ["資遣", "離職", "解僱", "終止契約", "預告", "遣散", "非自願離職"],
    "attendance": ["出勤", "遲到", "早退", "打卡", "曠職", "排班", "輪班"],
    "welfare": ["福利", "旅遊補助", "員工貸款", "餐補", "教育訓練", "生日", "健康檢查"],
    "occupational_accident": ["職災", "受傷", "職業災害", "補償", "醫療"],
    "privacy_sensitive": ["個資", "身分證", "病歷", "診斷證明", "申訴", "懲處", "性騷擾"],
}

RISK_KEYWORDS = ["違法", "申訴", "告", "提告", "主管逼", "強迫", "不給薪", "扣薪", "解僱", "懲處", "性騷擾", "個資", "歧視"]
AMBIGUOUS_SHORTS = ["我想請假", "請假規定", "休假規定", "可以嗎", "合法嗎", "怎麼辦", "我要請假"]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_article_no(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", "", x)
    x = x.replace("第", "第 ").replace("條", " 條")
    return normalize_spaces(x)


def detect_category(text: str) -> str:
    scores = defaultdict(int)
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[cat] += 1
    if not scores:
        return "general"
    return max(scores.items(), key=lambda x: x[1])[0]


def extract_keywords(text: str, max_n: int = 30) -> List[str]:
    kws = []
    for v in CATEGORY_KEYWORDS.values():
        for kw in v:
            if kw in text:
                kws.append(kw)
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    return list(dict.fromkeys(kws + tokens))[:max_n]


def tokenize_zh(text: str) -> List[str]:
    # Simple tokenizer for BM25: keywords + CJK bigrams + alnum tokens
    text = str(text)
    tokens = extract_keywords(text, max_n=50)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens += ["".join(cjk[i:i+2]) for i in range(max(0, len(cjk)-1))]
    tokens += re.findall(r"[A-Za-z0-9]{2,}", text.lower())
    return [t for t in tokens if t]


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def normalize_route(route: str) -> str:
    route = str(route or "").strip()
    mapping = {"direct": "answer", "with_disclaimer": "disclaimer", "answer": "answer", "clarify": "clarify", "escalate": "escalate", "disclaimer": "disclaimer"}
    return mapping.get(route, route or "answer")


def policy_to_route(policy: str) -> str:
    return normalize_route(policy)


def route_to_policy(route: str) -> str:
    route = normalize_route(route)
    return {"answer": "direct", "disclaimer": "with_disclaimer", "clarify": "clarify", "escalate": "escalate"}.get(route, "direct")


CATEGORY_ALIASES = {
    "working_hours": {"attendance", "overtime", "working_hours"},
    "high_risk": {"privacy_sensitive", "termination", "salary", "leave", "overtime", "general", "high_risk"},
    "privacy_sensitive": {"privacy_sensitive", "salary", "general"},
}


def category_matches(expected: Any, actual: Any) -> bool:
    expected = str(expected or "").strip()
    actual = str(actual or "").strip()
    if not expected:
        return True
    if expected == actual:
        return True
    return actual in CATEGORY_ALIASES.get(expected, set())


def normalize_for_match(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def extract_article_refs(text: Any) -> List[str]:
    """Extract normalized references like 第38條 from a citation string."""
    text = str(text or "")
    refs = []
    for raw in re.findall(rf"第\s*([\d{CJK_NUM}]+)\s*條", text):
        refs.append(normalize_for_match(f"第{raw}條"))
    return refs


def relation_from_policy_content(content: str) -> str:
    if any(k in content for k in ["優於", "較有利", "不低於", "最低標準"]):
        return "overrides"
    return "refers_to"


def law_article_id_from_ref(raw_no: str) -> str:
    return article_id_from_no("law", f"第 {raw_no} 條")


def extract_related_law_ids(content: str) -> List[str]:
    refs = []
    for raw in re.findall(rf"勞動基準法[^。；;\n]*第\s*([\d{CJK_NUM}]+)\s*條", content):
        refs.append(law_article_id_from_ref(raw))
    # Some policy clauses write only 「第 30 條」 after mentioning external regulation nearby.
    if "勞動基準法" in content:
        for raw in re.findall(rf"第\s*([\d{CJK_NUM}]+)\s*條", content):
            refs.append(law_article_id_from_ref(raw))
    return list(dict.fromkeys(refs))


def split_sentences_zh(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？；;])", text)
    return [p.strip() for p in parts if p.strip()]


def article_id_from_no(source_type: str, article_no: str) -> str:
    clean = re.sub(r"\s+", "", article_no)
    clean = clean.replace("第", "").replace("條", "")
    clean = clean.replace("之", "_")
    prefix = "law" if source_type == "law" else "policy"
    return f"{prefix}_{clean}"
