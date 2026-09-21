#!/usr/bin/env bash
# 第 3 组：延迟与并发压测（TestRule.md §二.3）
# 起 /retrieval 与 /api/ask 并发 1/4/8 各 60 次；语义缓存关闭；结束抓服务端
# HyDE 跳过次数与 /metrics 快照。
set -uo pipefail
cd /root/rag_v1
set -a; source .env.eval; set +a
export PYTHONUNBUFFERED=1
export SEMANTIC_CACHE_ENABLED=0     # 压测前提：关语义缓存（TestRule §二.3）
mkdir -p logs eval/reports

PY=.venv/bin/python
PORT="${API_PORT:-8000}"
TOKEN="$DIFY_API_TOKEN"
TS=$(date +%m%d_%H%M%S)
SRV_LOG="logs/g3_server_${TS}.log"

echo "==== 启动服务 (uvicorn :$PORT, SEMANTIC_CACHE_ENABLED=0) ===="
# --loop asyncio：uvloop 与 query.py 的 nest_asyncio/run_until_complete 不兼容，
  #   默认 uvloop 下 build_index 会抛 'this event loop is already running'（2026-09-20 实测）
  $PY -m uvicorn pcb_rag.dify_external_api:app --host 127.0.0.1 --port "$PORT" --loop asyncio \
  > "$SRV_LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "✗ 服务未就绪，中止"; tail -20 "$SRV_LOG"; kill $SRV_PID 2>/dev/null; exit 1; }
echo "  ✓ 服务就绪 (pid=$SRV_PID)"

trap 'kill $SRV_PID 2>/dev/null' EXIT

for ep in retrieval ask; do
  for conc in 1 4 8; do
    echo "==== /$ep 并发 $conc × 60 ===="
    $PY scripts/load_test.py --endpoint "$ep" --concurrency "$conc" --total 60 \
      --token "$TOKEN" \
      --out "eval/reports/g3_${ep}_c${conc}.json" \
      > "logs/g3_${ep}_c${conc}.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && echo "  ✗ rc=$rc（含错误请求，见日志）" || echo "  ✓ ok"
    tail -1 "logs/g3_${ep}_c${conc}.log" | head -c 400; echo
  done
done

echo "==== 服务端观测 ===="
echo "  HyDE 跳过/未命中: $(grep -cE '全部未命中|无剩余预算|生成超时' "$SRV_LOG")"
echo "  [Vec]/[BM25] 检索失败: $(grep -cE '\[Vec\] 检索失败|\[BM25\] 检索失败' "$SRV_LOG")"
echo "  Rerank 429 重试: $(grep -c '429 限流' "$SRV_LOG")"
curl -s "http://127.0.0.1:$PORT/metrics" > "eval/reports/g3_metrics_snapshot.json" 2>/dev/null \
  && echo "  /metrics 快照: eval/reports/g3_metrics_snapshot.json"
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
echo "  GPU: ${GPU:-N/A（当前检索/精排/生成均为 API 后端，无本地 GPU 推理）}"

echo "==== 汇总 ===="
$PY - <<'PYEOF'
import glob, json
print(f"{'端点':10} {'并发':>4} {'p50':>7} {'p95':>7} {'p99':>7} {'吞吐rps':>8} {'错误率':>7}")
for p in sorted(glob.glob("eval/reports/g3_*_c*.json")):
    s = json.loads(open(p, encoding="utf-8").read())["summary"]
    print(f"{s['endpoint']:10} {s['concurrency']:>4} {s['latency_p50']:>7.2f} {s['latency_p95']:>7.2f} "
          f"{s['latency_p99']:>7.2f} {s['throughput_rps']:>8.3f} {s['error_rate']:>7.2%}")
PYEOF
