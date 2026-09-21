#!/usr/bin/env bash
# 第 2 组：消融矩阵（TestRule.md §二.2）——13 行，每行一个独立进程。
# 全部基于 .env.eval 生产配置 + 行内显式覆盖（不用累积逻辑，避免顺序 bug）；
# 矩阵统一 CHUNK_EXPAND_ENABLED=0 取纯排名（行 12 专门测扩展）；
# 每行先跑 2 题热身，结果全空则跳过该行（防废报告，见 2026-09-20 事故）。
set -uo pipefail
cd /root/rag_v1
set -a; source .env.eval; set +a
export PYTHONUNBUFFERED=1
export CHUNK_EXPAND_ENABLED=0
mkdir -p eval/reports logs

PY=.venv/bin/python
DS=eval/datasets/regression_v1.json
TS=$(date +%m%d_%H%M%S)
FAILED_ROWS=()

# 行格式：名称|环境覆盖（相对 .env.eval）
ROWS=(
"00_bm25_only|"  # 单独用 evaluate_recall 跑，见下方特例
"01_vector_only|LEXICAL_ENABLED=0 RERANK_ENABLED=0 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=0 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=0 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"02_vector_rerank|LEXICAL_ENABLED=0 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=0 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=0 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"03_fusion_rerank|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=0 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=0 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"04_expansion|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=0 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"05_multi_expand|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=0 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"06_routing|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=0 QUERY_STEP_BACK_ENABLED=0"
"07_decompose_stepback|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=0 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1"
"08_hyde_production|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=1 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1"
"09_recall50|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=1 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1 RECALL_TOP_K=50"
"10_reranktop50|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=1 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1 RERANK_TOP_N=50"
"11_rule_mode|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=1 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1 QUERY_UNDERSTANDING_MODE=rule"
"12_chunk_expand|LEXICAL_ENABLED=1 RERANK_ENABLED=1 HYDE_ENABLED=1 QUERY_EXPANSION_ENABLED=1 QUERY_ROUTING_ENABLED=1 MULTI_EXPAND_ENABLED=1 QUERY_DECOMPOSE_ENABLED=1 QUERY_STEP_BACK_ENABLED=1 CHUNK_EXPAND_ENABLED=1"
)

echo "==== 行 00：BM25 单路（evaluate_recall --mode bm25）===="
$PY eval/evaluate_recall.py --dataset "$DS" --mode bm25 --no-rerank \
  --recall-k 200 --ks 1,3,5,10 --out eval/reports/r2_00_bm25_only.json \
  > "logs/r2_${TS}_00_bm25_only.log" 2>&1 \
  && echo "  ✓ 00 ok" || { echo "  ✗ 00 失败"; FAILED_ROWS+=("00"); }

for row in "${ROWS[@]}"; do
  name="${row%%|*}"; [ "$name" = "00_bm25_only" ] && continue
  vars="${row#*|}"
  echo "==== 行 $name ===="
  # shellcheck disable=SC2086
  env $vars $PY scripts/eval_production_path.py \
    --dataset "$DS" --top-k 20 --warmup 2 \
    --out "eval/reports/r2_${name}.json" \
    > "logs/r2_${TS}_${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  ✗ $name 失败 rc=$rc（跳过，继续下一行）"
    FAILED_ROWS+=("$name")
  else
    echo "  ✓ $name ok"
  fi
done

echo "==== 消融矩阵汇总 ===="
$PY - <<'PYEOF'
import glob, json
rows = sorted(glob.glob("eval/reports/r2_*.json")) + ["eval/reports/r1_baseline_production.json"]
print(f"{'配置':24} {'n':>4} {'MRR':>7} {'Hit@1':>7} {'Hit@5':>7} {'NDCG@10':>8} {'p50(s)':>7} 空结果")
prev = None
for p in rows:
    try:
        d = json.loads(open(p, encoding="utf-8").read())
        s = d.get("summary", d)
        if "hit_rate" in s:  # evaluate_recall 格式
            vals = (s.get("mrr"), s["hit_rate"].get("1"), s["hit_rate"].get("5"), s.get("ndcg", {}).get("10"), "-", 0)
        else:
            vals = (s.get("mrr"), s.get("hit@1"), s.get("hit@5"), s.get("ndcg@10"),
                    s.get("wall_s_p50", "-"), s.get("empty_results", "-"))
        name = p.split("/")[-1].replace("r2_", "").replace(".json", "")
        mark = ""
        if prev is not None and isinstance(vals[0], float) and isinstance(prev, float):
            mark = f"  (ΔMRR {vals[0]-prev:+.3f})"
        print(f"{name:24} {s.get('n','-'):>4} {vals[0]:>7.4f} {vals[1]:>7.3f} {vals[2]:>7.3f} {vals[3]:>8.4f} {str(vals[4]):>7} {vals[5]}{mark}")
        prev = vals[0] if isinstance(vals[0], float) else prev
    except Exception as e:
        print(f"{p}: 读取失败 {e}")
PYEOF
echo "失败行: ${FAILED_ROWS[*]:-无}"
