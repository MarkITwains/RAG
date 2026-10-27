#!/usr/bin/env bash
# 供应商切换后的恢复链：全量重嵌入 → 校验 → 重跑第 1 组基线
# 背景：.env 换供应商后，新 API 向量与库内存量向量 cosine 仅 0.34（空间不兼容），
#       必须用新供应商重嵌入；块文本/node id 不变（内容派生），回归集继续有效。
set -uo pipefail
cd /root/rag_v1
source scripts/env_defaults.sh
pcb_rag_load_env_defaults .env.eval
export PYTHONUNBUFFERED=1
mkdir -p logs eval/reports

TS=$(date +%m%d_%H%M%S)

echo "==== [1/3] 全量重嵌入（INGEST_OVERWRITE=1, EMBED_BATCH_SIZE=64） ===="
INGEST_OVERWRITE=1 EMBED_BATCH_SIZE=64 .venv/bin/python -m pcb_rag.ingest \
  > "logs/reingest_${TS}.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "  ✗ 重嵌入失败 rc=$rc"; exit 1; }
grep -E "Ingest done" "logs/reingest_${TS}.log" | tail -1

echo "==== [2/3] 校验：row_count 与维度 ===="
OK=$(.venv/bin/python - <<'PYEOF'
import os
from pathlib import Path
from pymilvus import MilvusClient
from pcb_rag.revisions import RevisionStore, state_path
c = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
collection = os.environ["COLLECTION"]
field = os.getenv("MILVUS_EMBEDDING_FIELD", "embedding")
c.flush(collection)
n = c.get_collection_stats(collection).get("row_count")
d = c.describe_collection(collection)["fields"]
dim = next((f["params"].get("dim") for f in d if f["name"] == field), None)
active = RevisionStore(state_path(collection)).read().get("active", {})
expected = sum(1 for line in Path(active["artifacts"]["chunks"]).open(encoding="utf-8") if line.strip())
print("1" if int(n or 0) == expected and int(dim or 0) == int(os.environ["EMBED_DIM"]) else "0")
PYEOF
)
[ "$OK" = "1" ] || { echo "  ✗ 校验失败（条数/维度不符），中止"; exit 1; }
echo "  ✓ chunks 与当前发布快照一致 / dim 与 EMBED_DIM 一致"

echo "==== [3/3] 重跑第 1 组基线（两条路径） ===="
bash scripts/run_r1_baseline.sh
