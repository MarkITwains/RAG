#!/usr/bin/env bash
# 第 1 组：回归基线两条路径（TestRule.md §二.1）
#   Path A: eval/evaluate_recall.py（与 docx/3月报告同源可比）
#   Path B: 生产路径 _retrieve_nodes（scripts/eval_production_path.py）
# 全程后台可跑；每步独立日志；一步失败不影响后续步骤。
set -uo pipefail
cd /root/rag_v1

# 生产配置唯一基准；env_loader 为 setdefault 语义，source 的值优先于 .env
set -a; source .env.eval; set +a
export PYTHONUNBUFFERED=1
mkdir -p eval/reports logs

PY=.venv/bin/python
DS=eval/datasets/regression_v1.json
TS=$(date +%m%d_%H%M%S)
FAIL=0

echo "==== [0/4] 热身：Path B 3 题（HyDE 冷启动 + 引擎初始化验证） ===="
CHUNK_EXPAND_ENABLED=0 $PY scripts/eval_production_path.py \
  --dataset "$DS" --top-k 20 --warmup 3 --limit 3 \
  --out eval/reports/r1_warmup_production.json \
  > "logs/r1_${TS}_warmup_production.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "  ✗ warmup B 失败 rc=$rc（中止，防止废报告）"; exit 1; } || echo "  ✓ warmup B ok"

echo "==== [1/4] 热身：Path A 3 题 ===="
$PY eval/evaluate_recall.py --dataset "$DS" --mode fusion_hyde_rerank \
  --recall-k 200 --rerank --rerank-top-n 10 --rrf-k 40 --ks 1,3,5,10 \
  --limit 3 --out eval/reports/r1_warmup_evalscript.json \
  > "logs/r1_${TS}_warmup_evalscript.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "  ✗ warmup A 失败 rc=$rc（中止）"; exit 1; } || echo "  ✓ warmup A ok"

echo "==== [2/4] Path A 全量 80 题：评测脚本路径 ===="
$PY eval/evaluate_recall.py --dataset "$DS" --mode fusion_hyde_rerank \
  --recall-k 200 --rerank --rerank-top-n 10 --rrf-k 40 --ks 1,3,5,10 \
  --out eval/reports/r1_baseline_evalscript.json \
  > "logs/r1_${TS}_evalscript.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "  ✗ Path A 失败 rc=$rc"; FAIL=1; } || echo "  ✓ Path A ok"

echo "==== [3/4] Path B 全量 80 题：生产路径（CHUNK_EXPAND_ENABLED=0 取纯排名） ===="
CHUNK_EXPAND_ENABLED=0 $PY scripts/eval_production_path.py \
  --dataset "$DS" --top-k 20 \
  --out eval/reports/r1_baseline_production.json \
  > "logs/r1_${TS}_production.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "  ✗ Path B 失败 rc=$rc"; FAIL=1; } || echo "  ✓ Path B ok"

echo "==== [4/4] 结果 ===="
$PY - <<'PYEOF'
import json, pathlib
for name in ["r1_baseline_evalscript", "r1_baseline_production"]:
    p = pathlib.Path(f"eval/reports/{name}.json")
    if not p.is_file():
        print(f"  {name}: 缺失"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d.get("summary") or d
    keys = ["mrr", "map", "hit@1", "hit@3", "hit@5", "hit@10", "ndcg@10"]
    vals = {k: s.get(k) for k in keys if s.get(k) is not None}
    print(f"  {name}: n={s.get('n')} " + " ".join(f"{k}={v}" for k, v in vals.items()))
PYEOF
echo "exit_flag=$FAIL"
