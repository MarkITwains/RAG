#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  PCB-RAG 优化配置快速启动脚本 v2.0
#  
#  功能：交互式配置检索参数、入库、查询、评估
#  更新：2026-01-28
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ${BOLD}$1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}─────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────────────────────────────────────

set_defaults() {
    # 入库配置
    export NODE_PARSER_MODE="${NODE_PARSER_MODE:-structure}"
    export SEMANTIC_BUFFER_SIZE="${SEMANTIC_BUFFER_SIZE:-1}"
    export SEMANTIC_BREAKPOINT_THRESHOLD="${SEMANTIC_BREAKPOINT_THRESHOLD:-90}"
    export SEMANTIC_CHUNK_SIZE="${SEMANTIC_CHUNK_SIZE:-3000}"
    export STRUCTURE_MAX_CHUNK_SIZE="${STRUCTURE_MAX_CHUNK_SIZE:-2000}"
    export STRUCTURE_MIN_CHUNK_SIZE="${STRUCTURE_MIN_CHUNK_SIZE:-200}"
    export STRUCTURE_OVERLAP="${STRUCTURE_OVERLAP:-100}"
    export CHUNK_SIZE="${CHUNK_SIZE:-384}"
    export CHUNK_OVERLAP="${CHUNK_OVERLAP:-150}"
    export HIERARCHICAL_CHUNK_SIZES="${HIERARCHICAL_CHUNK_SIZES:-1024,384,128}"
    
    # 编码和文本清洗
    export AUTO_FIX_ENCODING="${AUTO_FIX_ENCODING:-1}"
    export CLEAN_OCR_GARBAGE="${CLEAN_OCR_GARBAGE:-1}"
    export NORMALIZE_TEXT="${NORMALIZE_TEXT:-1}"
    
    # 检索配置
    export FUSION_MODE="${FUSION_MODE:-DIST_BASED_SCORE}"
    export FUSION_WEIGHTS="${FUSION_WEIGHTS:-0.35,0.65}"
    export FUSION_NUM_QUERIES="${FUSION_NUM_QUERIES:-5}"
    export RECALL_TOP_K="${RECALL_TOP_K:-40}"
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED="${QUERY_EXPANSION_ENABLED:-1}"
    export QUERY_EXPANSION_MAX_TERMS="${QUERY_EXPANSION_MAX_TERMS:-5}"
    export QUERY_ROUTING_ENABLED="${QUERY_ROUTING_ENABLED:-1}"
    
    # BM25 配置
    export LEXICAL_ENABLED="${LEXICAL_ENABLED:-1}"
    
    # Rerank 配置
    export RERANK_ENABLED="${RERANK_ENABLED:-1}"
    export RERANK_BACKEND="${RERANK_BACKEND:-qwen3reranker}"
    export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-4B}"
    export RERANK_TOP_N="${RERANK_TOP_N:-8}"
    
    # ColBERT 配置（已禁用，生成式 reranker 不适用）
    export COLBERT_RERANK_ENABLED="${COLBERT_RERANK_ENABLED:-0}"
    
    # LLM 配置
    export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
    export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-120}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 主菜单
# ─────────────────────────────────────────────────────────────────────────────

show_main_menu() {
    print_header "PCB-RAG 检索优化系统 v2.0"
    
    echo "请选择操作："
    echo ""
    echo -e "  ${BOLD}[1]${NC} 🚀 快速启动（使用预设方案）"
    echo -e "  ${BOLD}[2]${NC} ⚙️  高级配置（自定义参数）"
    echo -e "  ${BOLD}[3]${NC} 📊 运行评估"
    echo -e "  ${BOLD}[4]${NC} 📥 仅入库数据"
    echo -e "  ${BOLD}[5]${NC} 🔍 仅启动查询"
    echo -e "  ${BOLD}[6]${NC} 📋 查看当前配置"
    echo -e "  ${BOLD}[7]${NC} 📖 查看帮助"
    echo -e "  ${BOLD}[0]${NC} 退出"
    echo ""
    read -p "请输入选项 [0-7]: " main_choice
}

# ─────────────────────────────────────────────────────────────────────────────
# 预设方案选择
# ─────────────────────────────────────────────────────────────────────────────

select_preset() {
    print_section "选择预设配置方案"
    
    echo ""
    echo -e "${BOLD}入库+检索一体化方案：${NC}"
    echo ""
    echo -e "  ${BOLD}[1]${NC} 🎯 ${GREEN}高精度方案${NC}（推荐）"
    echo "      结构感知切块 + 加权融合 + Rerank + 查询路由"
    echo "      适合：GB/T标准文档、追求最佳检索效果"
    echo ""
    echo -e "  ${BOLD}[2]${NC} ⚡ ${YELLOW}高速度方案${NC}"
    echo "      固定切块 + RRF融合 + 无Rerank"
    echo "      适合：低延迟场景、资源受限环境"
    echo ""
    echo -e "  ${BOLD}[3]${NC} 📚 ${BLUE}标准文档方案${NC}"
    echo "      结构感知切块 + BM25权重增强 + Rerank"
    echo "      适合：PCB标准/规范类文档查询"
    echo ""
    echo -e "  ${BOLD}[4]${NC} 🔬 ${CYAN}层次化方案${NC}"
    echo "      层次切块 + 加权融合 + Rerank"
    echo "      适合：结构化文档、需要上下文的场景"
    echo ""
    echo -e "${BOLD}仅检索优化方案（不重新入库）：${NC}"
    echo ""
    echo -e "  ${BOLD}[5]${NC} 📈 召回优先"
    echo "      大召回 + Rerank精排"
    echo ""
    echo -e "  ${BOLD}[6]${NC} 🎯 精度优先"
    echo "      查询路由 + 查询扩展 + 强Rerank"
    echo ""
    echo -e "  ${BOLD}[0]${NC} 返回主菜单"
    echo ""
    read -p "请选择方案 [0-6]: " preset_choice
    
    case $preset_choice in
        1) apply_preset_high_precision ;;
        2) apply_preset_high_speed ;;
        3) apply_preset_standard_docs ;;
        4) apply_preset_hierarchical ;;
        5) apply_preset_recall_first ;;
        6) apply_preset_precision_first ;;
        0) return 1 ;;
        *) print_error "无效选择"; return 1 ;;
    esac
    return 0
}

apply_preset_high_precision() {
    print_success "应用高精度方案"
    
    # 入库配置 - 使用结构感知切块
    export NODE_PARSER_MODE=structure
    export STRUCTURE_MAX_CHUNK_SIZE=2000
    export STRUCTURE_MIN_CHUNK_SIZE=200
    export STRUCTURE_OVERLAP=100
    
    # 编码修复和文本清洗
    export AUTO_FIX_ENCODING=1
    export CLEAN_OCR_GARBAGE=1
    
    # 检索配置
    export FUSION_MODE=DIST_BASED_SCORE
    export FUSION_WEIGHTS=0.35,0.65
    export FUSION_NUM_QUERIES=5
    export RECALL_TOP_K=50
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_EXPANSION_MAX_TERMS=5
    export QUERY_ROUTING_ENABLED=1
    
    # Rerank
    export RERANK_ENABLED=1
    export RERANK_BACKEND=qwen3reranker
    export RERANK_TOP_N=10
    
    NEED_REINGEST=true
}

apply_preset_high_speed() {
    print_success "应用高速度方案"
    
    # 入库配置
    export NODE_PARSER_MODE=sentence
    export CHUNK_SIZE=512
    export CHUNK_OVERLAP=100
    
    # 检索配置
    export FUSION_MODE=RECIPROCAL_RANK
    export FUSION_NUM_QUERIES=3
    export RECALL_TOP_K=20
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_ROUTING_ENABLED=0
    
    # Rerank（关闭）
    export RERANK_ENABLED=0
    
    NEED_REINGEST=true
}

apply_preset_standard_docs() {
    print_success "应用标准文档方案"
    
    # 入库配置 - 使用结构感知切块
    export NODE_PARSER_MODE=structure
    export STRUCTURE_MAX_CHUNK_SIZE=2000
    export STRUCTURE_MIN_CHUNK_SIZE=200
    export STRUCTURE_OVERLAP=100
    
    # 编码修复和文本清洗
    export AUTO_FIX_ENCODING=1
    export CLEAN_OCR_GARBAGE=1
    
    # 检索配置 - BM25权重更高
    export FUSION_MODE=DIST_BASED_SCORE
    export FUSION_WEIGHTS=0.25,0.75
    export FUSION_NUM_QUERIES=5
    export RECALL_TOP_K=50
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_EXPANSION_MAX_TERMS=6
    export QUERY_ROUTING_ENABLED=1
    
    # Rerank
    export RERANK_ENABLED=1
    export RERANK_BACKEND=qwen3reranker
    export RERANK_TOP_N=8
    
    NEED_REINGEST=true
}

apply_preset_hierarchical() {
    print_success "应用层次化方案"
    
    # 入库配置
    export NODE_PARSER_MODE=hierarchical
    export HIERARCHICAL_CHUNK_SIZES=1024,384,128
    
    # 检索配置
    export FUSION_MODE=DIST_BASED_SCORE
    export FUSION_WEIGHTS=0.35,0.65
    export FUSION_NUM_QUERIES=5
    export RECALL_TOP_K=40
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_ROUTING_ENABLED=1
    
    # Rerank
    export RERANK_ENABLED=1
    export RERANK_TOP_N=8
    
    NEED_REINGEST=true
}

apply_preset_recall_first() {
    print_success "应用召回优先方案（不需要重新入库）"
    
    # 检索配置 - 大召回
    export FUSION_MODE=DIST_BASED_SCORE
    export FUSION_WEIGHTS=0.3,0.7
    export FUSION_NUM_QUERIES=5
    export RECALL_TOP_K=100
    
    # 查询增强
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_ROUTING_ENABLED=1
    
    # Rerank - 从大量候选中精排
    export RERANK_ENABLED=1
    export RERANK_TOP_N=10
    
    NEED_REINGEST=false
}

apply_preset_precision_first() {
    print_success "应用精度优先方案（不需要重新入库）"
    
    # 检索配置
    export FUSION_MODE=DIST_BASED_SCORE
    export FUSION_WEIGHTS=0.35,0.65
    export FUSION_NUM_QUERIES=5
    export RECALL_TOP_K=50
    
    # 查询增强 - 全部开启
    export QUERY_EXPANSION_ENABLED=1
    export QUERY_EXPANSION_MAX_TERMS=6
    export QUERY_ROUTING_ENABLED=1
    
    # Rerank - 强精排
    export RERANK_ENABLED=1
    export RERANK_BACKEND=qwen3reranker
    export RERANK_TOP_N=8
    
    NEED_REINGEST=false
}

# ─────────────────────────────────────────────────────────────────────────────
# 高级配置
# ─────────────────────────────────────────────────────────────────────────────

advanced_config() {
    print_section "高级配置"
    
    # 入库配置
    echo ""
    echo -e "${BOLD}[1] 文档切块配置${NC}"
    echo "    当前模式: $NODE_PARSER_MODE"
    echo "    可选: structure | semantic | hierarchical | sentence"
    read -p "    新值 (回车保持当前): " new_val
    [[ -n "$new_val" ]] && export NODE_PARSER_MODE="$new_val"
    
    if [[ "$NODE_PARSER_MODE" == "structure" ]]; then
        echo ""
        echo "    结构感知切块参数:"
        read -p "    STRUCTURE_MAX_CHUNK_SIZE [$STRUCTURE_MAX_CHUNK_SIZE]: " new_val
        [[ -n "$new_val" ]] && export STRUCTURE_MAX_CHUNK_SIZE="$new_val"
        read -p "    STRUCTURE_MIN_CHUNK_SIZE [$STRUCTURE_MIN_CHUNK_SIZE]: " new_val
        [[ -n "$new_val" ]] && export STRUCTURE_MIN_CHUNK_SIZE="$new_val"
        read -p "    STRUCTURE_OVERLAP [$STRUCTURE_OVERLAP]: " new_val
        [[ -n "$new_val" ]] && export STRUCTURE_OVERLAP="$new_val"
    elif [[ "$NODE_PARSER_MODE" == "semantic" ]]; then
        echo ""
        echo "    语义切块参数:"
        read -p "    SEMANTIC_BREAKPOINT_THRESHOLD [$SEMANTIC_BREAKPOINT_THRESHOLD]: " new_val
        [[ -n "$new_val" ]] && export SEMANTIC_BREAKPOINT_THRESHOLD="$new_val"
        read -p "    SEMANTIC_CHUNK_SIZE [$SEMANTIC_CHUNK_SIZE]: " new_val
        [[ -n "$new_val" ]] && export SEMANTIC_CHUNK_SIZE="$new_val"
    elif [[ "$NODE_PARSER_MODE" == "sentence" ]]; then
        echo ""
        echo "    固定切块参数:"
        read -p "    CHUNK_SIZE [$CHUNK_SIZE]: " new_val
        [[ -n "$new_val" ]] && export CHUNK_SIZE="$new_val"
        read -p "    CHUNK_OVERLAP [$CHUNK_OVERLAP]: " new_val
        [[ -n "$new_val" ]] && export CHUNK_OVERLAP="$new_val"
    fi
    
    # 融合配置
    echo ""
    echo -e "${BOLD}[2] 融合检索配置${NC}"
    echo "    当前模式: $FUSION_MODE"
    echo "    可选: RECIPROCAL_RANK | DIST_BASED_SCORE | RELATIVE_SCORE"
    read -p "    新值 (回车保持当前): " new_val
    [[ -n "$new_val" ]] && export FUSION_MODE="$new_val"
    
    if [[ "$FUSION_MODE" == "DIST_BASED_SCORE" ]]; then
        echo "    当前权重: $FUSION_WEIGHTS (向量,BM25)"
        read -p "    新值 (回车保持当前): " new_val
        [[ -n "$new_val" ]] && export FUSION_WEIGHTS="$new_val"
    fi
    
    read -p "    FUSION_NUM_QUERIES [$FUSION_NUM_QUERIES]: " new_val
    [[ -n "$new_val" ]] && export FUSION_NUM_QUERIES="$new_val"
    
    read -p "    RECALL_TOP_K [$RECALL_TOP_K]: " new_val
    [[ -n "$new_val" ]] && export RECALL_TOP_K="$new_val"
    
    # 查询增强
    echo ""
    echo -e "${BOLD}[3] 查询增强配置${NC}"
    read -p "    QUERY_EXPANSION_ENABLED (0/1) [$QUERY_EXPANSION_ENABLED]: " new_val
    [[ -n "$new_val" ]] && export QUERY_EXPANSION_ENABLED="$new_val"
    
    read -p "    QUERY_EXPANSION_MAX_TERMS [$QUERY_EXPANSION_MAX_TERMS]: " new_val
    [[ -n "$new_val" ]] && export QUERY_EXPANSION_MAX_TERMS="$new_val"
    
    read -p "    QUERY_ROUTING_ENABLED (0/1) [$QUERY_ROUTING_ENABLED]: " new_val
    [[ -n "$new_val" ]] && export QUERY_ROUTING_ENABLED="$new_val"
    
    # Rerank 配置
    echo ""
    echo -e "${BOLD}[4] Rerank 配置${NC}"
    read -p "    RERANK_ENABLED (0/1) [$RERANK_ENABLED]: " new_val
    [[ -n "$new_val" ]] && export RERANK_ENABLED="$new_val"
    
    if [[ "$RERANK_ENABLED" == "1" ]]; then
        echo "    当前后端: $RERANK_BACKEND"
        echo "    可选: sbert | hf | qwen3reranker"
        read -p "    新值 (回车保持当前): " new_val
        [[ -n "$new_val" ]] && export RERANK_BACKEND="$new_val"
        
        read -p "    RERANK_MODEL [$RERANK_MODEL]: " new_val
        [[ -n "$new_val" ]] && export RERANK_MODEL="$new_val"
        
        read -p "    RERANK_TOP_N [$RERANK_TOP_N]: " new_val
        [[ -n "$new_val" ]] && export RERANK_TOP_N="$new_val"
    fi
    
    print_success "配置已更新"
}

# ─────────────────────────────────────────────────────────────────────────────
# 显示当前配置
# ─────────────────────────────────────────────────────────────────────────────

show_config() {
    print_section "当前配置"
    
    echo ""
    echo -e "${BOLD}📁 文档切块${NC}"
    echo "   NODE_PARSER_MODE      = $NODE_PARSER_MODE"
    if [[ "$NODE_PARSER_MODE" == "structure" ]]; then
        echo "   MAX_CHUNK_SIZE        = $STRUCTURE_MAX_CHUNK_SIZE"
        echo "   MIN_CHUNK_SIZE        = $STRUCTURE_MIN_CHUNK_SIZE"
        echo "   OVERLAP               = $STRUCTURE_OVERLAP"
    elif [[ "$NODE_PARSER_MODE" == "semantic" ]]; then
        echo "   SEMANTIC_THRESHOLD    = $SEMANTIC_BREAKPOINT_THRESHOLD"
        echo "   SEMANTIC_CHUNK_SIZE   = $SEMANTIC_CHUNK_SIZE"
    elif [[ "$NODE_PARSER_MODE" == "sentence" ]]; then
        echo "   CHUNK_SIZE            = $CHUNK_SIZE"
        echo "   CHUNK_OVERLAP         = $CHUNK_OVERLAP"
    elif [[ "$NODE_PARSER_MODE" == "hierarchical" ]]; then
        echo "   CHUNK_SIZES           = $HIERARCHICAL_CHUNK_SIZES"
    fi
    echo "   AUTO_FIX_ENCODING     = $AUTO_FIX_ENCODING"
    echo "   CLEAN_OCR_GARBAGE     = $CLEAN_OCR_GARBAGE"
    
    echo ""
    echo -e "${BOLD}🔍 混合检索${NC}"
    echo "   FUSION_MODE           = $FUSION_MODE"
    [[ "$FUSION_MODE" == "DIST_BASED_SCORE" ]] && echo "   FUSION_WEIGHTS        = $FUSION_WEIGHTS"
    echo "   FUSION_NUM_QUERIES    = $FUSION_NUM_QUERIES"
    echo "   RECALL_TOP_K          = $RECALL_TOP_K"
    echo "   LEXICAL_ENABLED       = $LEXICAL_ENABLED"
    
    echo ""
    echo -e "${BOLD}✨ 查询增强${NC}"
    echo "   QUERY_EXPANSION       = $QUERY_EXPANSION_ENABLED"
    echo "   EXPANSION_MAX_TERMS   = $QUERY_EXPANSION_MAX_TERMS"
    echo "   QUERY_ROUTING         = $QUERY_ROUTING_ENABLED"
    
    echo ""
    echo -e "${BOLD}🎯 Rerank${NC}"
    echo "   RERANK_ENABLED        = $RERANK_ENABLED"
    if [[ "$RERANK_ENABLED" == "1" ]]; then
        echo "   RERANK_BACKEND        = $RERANK_BACKEND"
        echo "   RERANK_MODEL          = $RERANK_MODEL"
        echo "   RERANK_TOP_N          = $RERANK_TOP_N"
    fi
    
    echo ""
    echo -e "${BOLD}🤖 LLM${NC}"
    echo "   OLLAMA_NUM_CTX        = $OLLAMA_NUM_CTX"
    echo "   OLLAMA_TIMEOUT        = $OLLAMA_TIMEOUT"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# 入库数据
# ─────────────────────────────────────────────────────────────────────────────

run_ingest() {
    print_section "数据入库"
    
    show_config
    
    echo ""
    read -p "确认开始入库？这将重建向量库 (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "已取消"
        return 1
    fi
    
    echo ""
    print_info "开始入库..."
    echo ""
    
    if python -m pcb_rag.ingest; then
        print_success "数据入库完成"
        return 0
    else
        print_error "数据入库失败"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 启动查询
# ─────────────────────────────────────────────────────────────────────────────

run_query() {
    print_section "启动查询服务"
    
    show_config
    
    echo ""
    print_info "启动查询服务..."
    echo ""
    
    python -m pcb_rag.query
}

# ─────────────────────────────────────────────────────────────────────────────
# 运行评估
# ─────────────────────────────────────────────────────────────────────────────

run_evaluation() {
    print_section "运行检索评估"
    
    echo ""
    echo "选择评估模式："
    echo ""
    echo -e "  ${BOLD}[1]${NC} vector      - 纯向量检索"
    echo -e "  ${BOLD}[2]${NC} bm25        - 纯BM25检索"
    echo -e "  ${BOLD}[3]${NC} fusion      - RRF融合检索"
    echo -e "  ${BOLD}[4]${NC} union_rrf   - 并集RRF融合（推荐）"
    echo -e "  ${BOLD}[5]${NC} 自定义"
    echo ""
    read -p "请选择 [1-5]: " eval_choice
    
    case $eval_choice in
        1) EVAL_MODE="vector" ;;
        2) EVAL_MODE="bm25" ;;
        3) EVAL_MODE="fusion" ;;
        4) EVAL_MODE="union_rrf" ;;
        5) 
            read -p "输入评估模式: " EVAL_MODE
            ;;
        *) print_error "无效选择"; return 1 ;;
    esac
    
    echo ""
    read -p "召回数量 RECALL_K [$RECALL_TOP_K]: " recall_k
    recall_k=${recall_k:-$RECALL_TOP_K}
    
    read -p "是否启用 Rerank? (y/N): " use_rerank
    rerank_flag=""
    rerank_n=""
    if [[ "$use_rerank" == "y" || "$use_rerank" == "Y" ]]; then
        rerank_flag="--rerank"
        read -p "Rerank Top N [$RERANK_TOP_N]: " rerank_n
        rerank_n=${rerank_n:-$RERANK_TOP_N}
        rerank_flag="$rerank_flag --rerank-top-n $rerank_n"
    fi
    
    read -p "评估K值列表 [1,3,5,10,20]: " ks
    ks=${ks:-"1,3,5,10,20"}
    
    # 生成输出文件名
    timestamp=$(date +%Y%m%d_%H%M%S)
    output_file="./eval/eval_report_${EVAL_MODE}_${timestamp}.json"
    
    echo ""
    print_info "运行评估..."
    echo ""
    echo "命令: python eval/evaluate_recall.py --mode $EVAL_MODE --recall-k $recall_k --ks $ks $rerank_flag --out $output_file"
    echo ""
    
    if python eval/evaluate_recall.py \
        --mode "$EVAL_MODE" \
        --recall-k "$recall_k" \
        --ks "$ks" \
        $rerank_flag \
        --out "$output_file"; then
        
        echo ""
        print_success "评估完成"
        echo ""
        echo -e "${BOLD}评估结果：${NC}"
        python3 -c "
import json
with open('$output_file') as f:
    r = json.load(f)
print(f\"  样本数: {r['n']}\")
print(f\"  MRR:    {r['mrr']:.4f}\")
print(f\"  Hit@1:  {r['hit_rate'].get('1', 0):.4f}\")
print(f\"  Hit@5:  {r['hit_rate'].get('5', 0):.4f}\")
print(f\"  Hit@10: {r['hit_rate'].get('10', 0):.4f}\")
print(f\"  NDCG@10:{r['ndcg'].get('10', 0):.4f}\")
"
        echo ""
        echo "详细报告: $output_file"
    else
        print_error "评估失败"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 帮助信息
# ─────────────────────────────────────────────────────────────────────────────

show_help() {
    print_header "帮助信息"
    
    echo -e "${BOLD}预设方案说明：${NC}"
    echo ""
    echo "  高精度方案："
    echo "    - 结构感知切块，按GB/T标准章节结构切分"
    echo "    - 自动修复文件编码（GBK→UTF-8）和清理OCR乱码"
    echo "    - 加权融合，BM25权重65%用于精确匹配"
    echo "    - 启用Rerank进行精排"
    echo "    - 启用查询路由，智能选择检索策略"
    echo "    - 预期效果：MRR 0.55+, Hit@10 0.88+"
    echo ""
    echo "  高速度方案："
    echo "    - 固定切块，入库速度快"
    echo "    - RRF融合，计算简单"
    echo "    - 关闭Rerank，降低延迟"
    echo "    - 预期效果：延迟 <500ms, Hit@10 0.75+"
    echo ""
    echo "  标准文档方案："
    echo "    - 专门针对GB/T标准文档优化"
    echo "    - BM25权重提升到75%，精确匹配术语"
    echo "    - 扩展词数增加，覆盖更多同义词"
    echo ""
    
    echo -e "${BOLD}关键参数说明：${NC}"
    echo ""
    echo "  FUSION_WEIGHTS:"
    echo "    - 格式: 向量权重,BM25权重"
    echo "    - 0.35,0.65 = BM25主导（适合专业术语）"
    echo "    - 0.60,0.40 = 向量主导（适合语义理解）"
    echo ""
    echo "  RECALL_TOP_K:"
    echo "    - 召回数量，越大召回率越高但延迟增加"
    echo "    - 推荐: 40-100"
    echo ""
    echo "  RERANK_TOP_N:"
    echo "    - Rerank后保留的文档数"
    echo "    - 推荐: 5-10"
    echo ""
    
    echo -e "${BOLD}评估指标说明：${NC}"
    echo ""
    echo "  MRR (Mean Reciprocal Rank):"
    echo "    - 相关文档排名倒数的平均值"
    echo "    - 目标: ≥0.60"
    echo ""
    echo "  Hit@K:"
    echo "    - 前K个结果中包含相关文档的比例"
    echo "    - Hit@1 目标: ≥0.45, Hit@10 目标: ≥0.85"
    echo ""
    echo "  NDCG@K:"
    echo "    - 考虑排名位置的综合指标"
    echo "    - 目标: ≥0.65"
    echo ""
    
    echo "详细文档请参考："
    echo "  - docs/CONFIGURATION_GUIDE.md (配置指南)"
    echo "  - docs/OPTIMIZATION_NOTES.md (优化记录与路线图)"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# 快速启动
# ─────────────────────────────────────────────────────────────────────────────

quick_start() {
    if ! select_preset; then
        return 1
    fi
    
    show_config
    
    if [[ "$NEED_REINGEST" == "true" ]]; then
        echo ""
        read -p "是否需要重新入库数据？(y/N): " reingest
        if [[ "$reingest" == "y" || "$reingest" == "Y" ]]; then
            run_ingest || return 1
        fi
    fi
    
    echo ""
    read -p "是否启动查询服务？(Y/n): " start_query
    if [[ "$start_query" != "n" && "$start_query" != "N" ]]; then
        run_query
    else
        echo ""
        print_info "配置已应用，可手动运行:"
        echo "  python -m pcb_rag.query"
        echo ""
        print_info "或运行评估:"
        echo "  python eval/evaluate_recall.py --mode union_rrf --recall-k $RECALL_TOP_K"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

main() {
    set_defaults
    
    # 支持命令行参数
    case "${1:-}" in
        --query|-q)
            run_query
            exit 0
            ;;
        --ingest|-i)
            run_ingest
            exit 0
            ;;
        --eval|-e)
            run_evaluation
            exit 0
            ;;
        --config|-c)
            show_config
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
    esac
    
    while true; do
        show_main_menu
        
        case $main_choice in
            1) quick_start ;;
            2) 
                advanced_config
                show_config
                echo ""
                read -p "是否保存并启动？(Y/n): " confirm
                if [[ "$confirm" != "n" && "$confirm" != "N" ]]; then
                    echo ""
                    read -p "需要重新入库吗？(y/N): " reingest
                    [[ "$reingest" == "y" || "$reingest" == "Y" ]] && run_ingest
                    run_query
                fi
                ;;
            3) run_evaluation ;;
            4) run_ingest ;;
            5) run_query ;;
            6) show_config; read -p "按回车继续..." ;;
            7) show_help; read -p "按回车继续..." ;;
            0) 
                echo ""
                print_info "再见！"
                exit 0
                ;;
            *)
                print_error "无效选择，请重试"
                ;;
        esac
    done
}

# 运行主程序
main "$@"
