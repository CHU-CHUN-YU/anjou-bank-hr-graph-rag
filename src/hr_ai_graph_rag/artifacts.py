# ============================================================
# artifacts — 離線 LLM 輔助知識/設定載入
#
# 載入離線產生的 JSON/ZIP artifacts(concept_nodes、risk_policy、
# query_patterns、rewrite_rules、relation_schema、graph_relation_candidates 等)。
# 依賴:config(路徑/Colab)、utils(CATEGORY_KEYWORDS 合併)。
# ============================================================

from .config import *
from .utils import *


OFFLINE_ARTIFACT_FILENAMES = [
    "workflow_role_mapping.json",
    "concept_nodes.json",
    "risk_policy.json",
    "query_patterns.json",
    "rewrite_rules.json",
    "relation_schema.json",
    "graph_relation_candidates.json",
    "local_llm_usage_policy.json",
    "artifact_manifest.json",
]

class OfflineArtifacts:
    """Versioned offline LLM-assisted knowledge/config artifacts.

    These artifacts are generated or curated before runtime, then loaded as JSON.
    Runtime local LLM may use them for structured classification context, but final
    risk and route decisions are still deterministic.
    """
    def __init__(self, artifact_dir: Optional[str] = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.data: Dict[str, Any] = {}
        self.loaded_files: Dict[str, str] = {}

    @property
    def concept_nodes(self) -> List[Dict[str, Any]]:
        return self.data.get("concept_nodes", {}).get("concept_nodes", [])

    @property
    def risk_policies(self) -> List[Dict[str, Any]]:
        return self.data.get("risk_policy", {}).get("risk_policies", [])

    @property
    def query_patterns(self) -> List[Dict[str, Any]]:
        return self.data.get("query_patterns", {}).get("patterns", [])

    @property
    def rewrite_rules(self) -> List[Dict[str, Any]]:
        return self.data.get("rewrite_rules", {}).get("rewrite_rules", [])

    @property
    def relation_types(self) -> List[Dict[str, Any]]:
        return self.data.get("relation_schema", {}).get("relation_types", [])

    @property
    def graph_edge_candidates(self) -> List[Dict[str, Any]]:
        return self.data.get("graph_relation_candidates", {}).get("edge_candidates", [])

    def load(self) -> "OfflineArtifacts":
        if not self.artifact_dir or not self.artifact_dir.exists():
            print("No offline artifact folder found. Falling back to code defaults where available.")
            return self
        for fn in OFFLINE_ARTIFACT_FILENAMES:
            fp = self.artifact_dir / fn
            if fp.exists():
                key = fp.stem
                with open(fp, "r", encoding="utf-8") as f:
                    self.data[key] = json.load(f)
                self.loaded_files[key] = str(fp)
        print("Loaded offline artifacts:", sorted(self.loaded_files.keys()))
        return self

    def category_keywords(self) -> Dict[str, List[str]]:
        kw = defaultdict(list)
        for c in self.concept_nodes:
            cat = c.get("category") or "general"
            vals = []
            vals.append(c.get("label", ""))
            vals.extend(c.get("aliases", []) or [])
            vals.extend(c.get("retrieval_keywords", []) or [])
            for v in vals:
                if v and v not in kw[cat]:
                    kw[cat].append(str(v))
        return dict(kw)


def _extract_zip_to_dir(zip_path: str, target_dir: Path) -> Optional[Path]:
    zp = Path(zip_path)
    if not zp.exists() or not zp.suffix.lower() == ".zip":
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(target_dir)
    # If zip contains a single folder, use it; otherwise use target root.
    candidates = [p for p in target_dir.iterdir() if p.is_dir()]
    for cand in candidates:
        if (cand / "concept_nodes.json").exists():
            return cand
    return target_dir


def locate_offline_artifacts_dir() -> Optional[str]:
    # 1) Explicit directory
    if OFFLINE_ARTIFACT_DIR and Path(OFFLINE_ARTIFACT_DIR).exists():
        return str(Path(OFFLINE_ARTIFACT_DIR))

    # 2) Explicit ZIP
    if OFFLINE_ARTIFACT_ZIP_PATH and Path(OFFLINE_ARTIFACT_ZIP_PATH).exists():
        extracted = _extract_zip_to_dir(OFFLINE_ARTIFACT_ZIP_PATH, OUTPUT_DIR / "offline_artifacts_loaded")
        if extracted:
            return str(extracted)

    # 3) Common local/Colab locations
    candidates = [
        Path("/content/hr_offline_artifacts"),
        Path("/content/offline_artifacts"),
        Path("./hr_offline_artifacts"),
        Path("./offline_artifacts"),
        Path("/mnt/data/hr_offline_artifacts"),
    ]
    for p in candidates:
        if p.exists() and (p / "concept_nodes.json").exists():
            return str(p)

    # 4) Auto-detect ZIP in common locations
    zip_candidates = list(Path("/content").glob("*offline*artifacts*.zip")) if Path("/content").exists() else []
    zip_candidates += list(Path(".").glob("*offline*artifacts*.zip"))
    zip_candidates += list(Path("/mnt/data").glob("*offline*artifacts*.zip")) if Path("/mnt/data").exists() else []
    if zip_candidates:
        extracted = _extract_zip_to_dir(str(zip_candidates[0]), OUTPUT_DIR / "offline_artifacts_loaded")
        if extracted:
            return str(extracted)

    # 5) Optional upload in Colab
    if IN_COLAB:
        print("可選：上傳 offline artifacts ZIP（若略過，會使用程式內建 fallback 規則）。")
        try:
            uploaded = files.upload()
            for name in uploaded.keys():
                if name.lower().endswith(".zip"):
                    extracted = _extract_zip_to_dir(f"/content/{name}", OUTPUT_DIR / "offline_artifacts_loaded")
                    if extracted:
                        return str(extracted)
        except Exception as e:
            print("Offline artifact upload skipped or failed:", repr(e))
    return None


def load_offline_artifacts() -> OfflineArtifacts:
    artifact_dir = locate_offline_artifacts_dir()
    artifacts = OfflineArtifacts(artifact_dir).load()
    # Update global CATEGORY_KEYWORDS using concept_nodes as an externalized taxonomy.
    external_kw = artifacts.category_keywords()
    if external_kw:
        for cat, vals in external_kw.items():
            base = CATEGORY_KEYWORDS.setdefault(cat, [])
            for v in vals:
                if v and v not in base:
                    base.append(v)
    return artifacts

