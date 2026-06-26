#!/usr/bin/env bash
# Run the full pipeline locally (requires the full GPU/LLM stack from requirements.txt).
# For the no-LLM smoke test use:  python tests/test_pipeline_no_llm.py
set -euo pipefail
cd "$(dirname "$0")/.."

# Bundled data under data/ is auto-discovered. Override any of these if needed:
# export LABOR_LAW_DOCX_PATH=/path/to/勞動基準法.docx        # else a built-in sample is used
# export INTERNAL_POLICY_DOCX_PATH=data/policies/安久銀行員工工作與福利規章辦法_模擬版.docx
# export GOLDEN_DATASET_JSON_PATH=data/golden/anjou_bank_hr_ai_golden_dataset_50.json
# export OFFLINE_ARTIFACT_DIR=data/hr_offline_artifacts
# export USE_LOCAL_LLM_FOR_QUERY_UNDERSTANDING=true

python src/hr_ai_graph_rag.py
