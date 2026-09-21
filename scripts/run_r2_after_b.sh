#!/usr/bin/env bash
# 接续链：等 Path B 重跑结束 → 闸门判定（TestRule §三：两路径 MRR 差 >0.03 先查原因）→ 通过则自动启动第 2 组消融矩阵
set -uo pipefail
cd /root/rag_v1

echo "==== 等待 pcb-eval-b（Path B 重跑）结束 ===="
for i in $(seq 1 90); do
  systemctl is-active --quiet pcb-eval-b.service 2>/dev/null || break
  sleep 60
done
systemctl is-active --quiet pcb-eval-b.service 2>/dev/null && { echo "等待超时（90 分钟），Path B 仍在跑，中止接续"; exit 1; }
echo "Path B 已结束"

echo "==== 闸门判定 ===="
.venv/bin/python - <<'PYEOF'
import json, sys
def mrr(p):
    d = json.loads(open(p, encoding="utf-8").read())
    s = d.get("summary", d)
    return float(s.get("mrr", 0.0)), s

ma, sa = mrr("eval/reports/r1_baseline_evalscript.json")
mb, sb = mrr("eval/reports/r1_baseline_production.json")
diff = abs(ma - mb)
print(f"  评测脚本路径 MRR = {ma:.4f} (n={sa.get('n')})")
print(f"  生产路径    MRR = {mb:.4f} (n={sb.get('n')}, 空结果={sb.get('empty_results','?')})")
print(f"  |ΔMRR| = {diff:.4f}  （闸门 0.03）")
if sb.get("n") != 80 or sb.get("empty_results") == 80 or mb <= 0.001:
    print("  ✗ 生产路径数据无效（n≠80 或全空/全零），中止")
    sys.exit(1)
if diff > 0.03:
    print("  ✗ 两路径差异超闸门 —— 按 TestRule §三，先查原因再往下走，消融矩阵不启动")
    sys.exit(1)
print("  ✓ 过闸，启动消融矩阵")
PYEOF
[ $? -ne 0 ] && exit 1

echo "==== 启动第 2 组消融矩阵 ===="
bash scripts/run_r2_ablation.sh
