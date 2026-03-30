#!/usr/bin/env bash
# 一键启动 vLLM + cpolar 穿透（使用固定二级子域名）
set -e

# ========== 用户配置区（请根据实际情况修改） ==========
# vLLM 配置
VLLM_PORT=8000
MODEL_PATH="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3___5-9B"  # 你的模型路径
CONDA_ENV="vllm_qwen"                   # conda 环境名

# cpolar 配置（固定子域名）
CPOLAR_SUBDOMAIN="zjlchat"               # 你在 cpolar 官网保留的二级子域名
CPOLAR_REGION="cn_vip"                    # 保留时选择的区域（一般为 cn_vip）
LOG_DIR="/root/autodl-tmp/vllm"           # 日志存放目录
# ===================================================

mkdir -p "$LOG_DIR"

# 清除代理（防止干扰内网穿透）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 1. 启动 vLLM 服务（若未运行）
if ! pgrep -f "vllm serve.*$MODEL_PATH" >/dev/null; then
    echo ">>> 启动 vLLM 服务 ..."
    # 加载 conda 环境（确保 vllm 命令可用）
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"

    # 后台启动 vLLM
    nohup vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --max-model-len 8192 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.85 \
        --swap-space 4 \
        --served-model-name qwen3.5-9b \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        >> "$LOG_DIR/vllm.log" 2>&1 &
    sleep 3
fi

# 2. 等待 vLLM 真正响应
echo ">>> 等待 vLLM 启动完成 ..."
until curl -s http://localhost:${VLLM_PORT}/v1/models >/dev/null 2>&1; do
    echo "   等待中..."
    sleep 1
done
echo ">>> vLLM 已在 ${VLLM_PORT} 就绪"

# 3. 启动 cpolar 穿透（使用固定子域名）
if ! pgrep -f "cpolar http.*-subdomain=${CPOLAR_SUBDOMAIN}" >/dev/null; then
    echo ">>> 启动 cpolar 穿透 (使用固定子域名: ${CPOLAR_SUBDOMAIN}) ..."
    # 关键改动：使用 ./cpolar 而不是 cpolar
    nohup ./cpolar http -subdomain="${CPOLAR_SUBDOMAIN}" -region="${CPOLAR_REGION}" "${VLLM_PORT}" \
        >> "$LOG_DIR/cpolar.log" 2>&1 &
    sleep 2

    # 构建预期的固定URL
    EXPECTED_URL="https://${CPOLAR_SUBDOMAIN}.vip.cpolar.cn"
    echo "=============================================================="
    echo "  预期公网地址（固定子域名）： $EXPECTED_URL"
    echo "  本地端口： ${VLLM_PORT}"
    echo "  注意：请确保已在 cpolar 官网保留此子域名"
    echo "=============================================================="

    # 从日志中提取实际分配的地址（用于确认）
    for i in {1..10}; do
        sleep 2
        URL=$(grep -Eo 'https://[^[:space:]]+\.cpolar\.(top|cn)' "$LOG_DIR/cpolar.log" | tail -n1)
        [[ -n "$URL" ]] && break
    done
    if [[ -n "$URL" ]]; then
        echo "  从日志中确认的实际地址： $URL"
    fi
else
    echo ">>> cpolar (子域名 ${CPOLAR_SUBDOMAIN}) 已运行，跳过"
fi

echo ">>> 一键启动完成！"
echo ">>> 查看日志：tail -f $LOG_DIR/vllm.log  |  tail -f $LOG_DIR/cpolar.log"