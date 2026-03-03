#!/usr/bin/env bash
# 一键启动 Ollama + cpolar 穿透（AutoDL 实测）- 使用固定二级子域名
set -e
LOG_DIR="/root/autodl-tmp/ollama"
OLLA_PORT=6006
CPOLAR_PORT=6006
# --- 新增部分：cpolar 配置变量 ---
CPOLAR_SUBDOMAIN="zjlchat"  # 请修改为你保留的二级子域名
CPOLAR_REGION="cn_vip"       # 地区，通常为 cn_vip (China VIP)
# --- 新增部分结束 ---
mkdir -p "$LOG_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 1. 启动 Ollama（若未运行）
if ! pgrep -f "ollama serve" >/dev/null; then
    echo ">>> 启动 Ollama 服务 ..."
    export OLLAMA_MODELS="/root/autodl-tmp/ollama"
    export OLLAMA_HOST="0.0.0.0:${OLLA_PORT}"
    nohup ollama serve >>"$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
fi

# 2. 等待 Ollama 真正响应
echo ">>> 等待 Ollama 启动完成 ..."
until curl -s http://localhost:${OLLA_PORT}/api/tags >/dev/null 2>&1; do
    echo "   等待中..."
    sleep 1
done
echo ">>> Ollama 已在 ${OLLA_PORT} 就绪"

# 3. 再启动 cpolar（端口一定已就绪）
# 修改判断条件，包含子域名参数
if ! pgrep -f "cpolar http -subdomain=${CPOLAR_SUBDOMAIN}" >/dev/null; then
    echo ">>> 启动 cpolar 穿透 (使用固定子域名: ${CPOLAR_SUBDOMAIN}) ..."
    nohup cpolar http -subdomain=${CPOLAR_SUBDOMAIN} -region=${CPOLAR_REGION} ${CPOLAR_PORT} >>"$LOG_DIR/cpolar.log" 2>&1 &
    sleep 2
    # 构建预期的固定URL
    EXPECTED_URL="https://${CPOLAR_SUBDOMAIN}.vip.cpolar.cn"
    echo "=============================================================="
    echo "  预期公网地址（固定子域名）： $EXPECTED_URL"
    echo "  本地端口： ${CPOLAR_PORT}"
    echo "  注意：请确保已在cpolar官网保留此子域名"
    echo "=============================================================="
    # 仍然尝试从日志获取，以防自动构建的URL与实际分配有细微差别（虽然通常不会）
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
echo ">>> 查看日志：tail -f $LOG_DIR/ollama.log  |  tail -f $LOG_DIR/cpolar.log"