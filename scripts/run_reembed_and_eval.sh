#!/usr/bin/env bash
# 供应商切换后的恢复链：全量重嵌入 → 校验 → 重跑第 1 组基线
# 背景：.env 换供应商后，新 API 向量与库内存量向量 cosine 仅 0.34（空间不兼容），
#       必须用新供应商重嵌入；块文本/node id 不变（内容派生），回归集继续有效。
set -uo pipefail
cd /root/rag_v1
set -a; source .env.eval; set +a
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
from pymilvus import MilvusClient
c = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
c.flush("pcb_kb")
n = c.get_collection_stats("pcb_kb").get("row_count")
d = c.describe_collection("pcb_kb")["fields"]
dim = next((f["params"].get("dim") for f in d if f["name"] == "embedding"), None)
print("1" if int(n or 0) == 5680 and int(dim or 0) == 2048 else "0")
PYEOF
)
[ "$OK" = "1" ] || { echo "  ✗ 校验失败（条数/维度不符），中止"; exit 1; }
echo "  ✓ 5680 chunks / dim 2048"

echo "==== [3/3] 重跑第 1 组基线（两条路径） ===="
bash scripts/run_r1_baseline.sh
