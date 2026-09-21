#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 第 4 阶段（G4 + G5）一键脚本，TestRule §二.4 / §二.5
#
#   G4 生成质量：GLM 出题（golden.jsonl 不存在时生成）→ DeepSeek 答题 →
#               GLM 异源判卷（evaluate.py 四指标）
#   G5 软匹配：  检索只跑一次（与 r1 Path A 同配置），离线扫描 4 个阈值，
#               导出"软命中但硬未命中"样本供人工分类
#
# 用法: bash scripts/run_g4_g5.sh
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd /root/rag_v1

set -a; source .env.eval; set +a
# 本机代理（127.0.0.1:12345）当前不可用且 emoera 可直连，绕过代理；
# 否则 httpx/requests 会走死代理导致所有 API 调用连接失败
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# GLM-5.3 为 reasoning 模型，思考 token 也计入 max_tokens，给足余量
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS_OVERRIDE:-8192}"
export PYTHONUNBUFFERED=1
mkdir -p eval/reports logs

PY=.venv/bin/python
JUDGE_MODEL="${JUDGE_MODEL:-glm-5-3-260814}"
TS=$(date +%m%d_%H%M%S)

echo "==== [1/4] GLM 可用性验证（JUDGE_MODEL=$JUDGE_MODEL） ===="
if ! curl -sS --max-time 60 --noproxy "*" "$LLM_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"只回复两个字：OK\"}],\"max_tokens\":512}" \
    | grep -q '"model"'; then
  echo "  ✗ GLM 请求失败，中止"
  exit 1
fi
echo ""

echo "==== [2/4] G5 软匹配：检索一次 + 阈值扫描（后台日志 logs/g5_${TS}.log） ===="
CHUNK_EXPAND_ENABLED=0 nohup $PY scripts/softmatch_scan.py \
  > "logs/g5_${TS}.log" 2>&1 &
G5_PID=$!
echo "  G5 已启动 pid=$G5_PID"

echo "==== [3/4] G4 出题：GLM 生成 golden.jsonl（不存在时才生成，日志 logs/g4_golden_${TS}.log） ===="
if [ -s eval/datasets/golden.jsonl ]; then
  echo "  golden.jsonl 已存在，跳过生成（如需重造请先删除）"
else
  LLM_MODEL="$JUDGE_MODEL" nohup $PY eval/build_golden_dataset.py \
    --limit 100 --out eval/datasets/golden.jsonl \
    > "logs/g4_golden_${TS}.log" 2>&1 &
  GOLDEN_PID=$!
  echo "  出题已启动 pid=$GOLDEN_PID，等待完成..."
  wait $GOLDEN_PID || { echo "  ✗ 出题失败，G4 评测不启动"; wait $G5_PID 2>/dev/null; exit 1; }
  N=$(wc -l < eval/datasets/golden.jsonl)
  echo "  ✓ golden.jsonl 生成 $N 条"
  [ "$N" -lt 30 ] && { echo "  ✗ 有效题量不足 30，G4 评测不启动"; wait $G5_PID 2>/dev/null; exit 1; }
fi

echo "==== [4/4] G4 评测：DeepSeek 答题 + GLM 判卷（日志 logs/g4_eval_${TS}.log） ===="
JUDGE_MODEL="$JUDGE_MODEL" JUDGE_BACKEND=api nohup $PY eval/evaluate.py \
  --dataset eval/datasets/golden.jsonl \
  --top-k 5 --recall-k 200 \
  --out "eval/reports/g4_gen_quality.json" \
  > "logs/g4_eval_${TS}.log" 2>&1 &
G4_PID=$!
echo "  G4 评测已启动 pid=$G4_PID"
echo ""
echo "全部任务已派发："
echo "  G5: tail -f logs/g5_${TS}.log"
echo "  G4 出题: tail -f logs/g4_golden_${TS}.log"
echo "  G4 评测: tail -f logs/g4_eval_${TS}.log"
