#!/usr/bin/env bash
# =============================================================================
# build_ops_transformer.sh
# 编译 ops-transformer 三方依赖的构建脚本
#
# 主要功能:
#   1. 安装依赖包: 读取 third_party/ops-transformer/requirements.txt 并 pip 安装
#   2. 执行编译:   bash build.sh --pkg --soc={soc} --vendor_name={vendor_name}
#                  --ops={ops} -j{jobs} -O3
#
# 用法:
#   bash build_ops_transformer.sh            # 使用默认常量编译
#   bash build_ops_transformer.sh install    # 仅安装依赖
#   bash build_ops_transformer.sh build      # 仅执行编译
#   bash build_ops_transformer.sh --soc=ascend950 build   # 指定 SOC 编译
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 常量定义区 (按需修改)
# -----------------------------------------------------------------------------

# 待编译的算子列表 (数组格式, 使用 snake name)
OPS=(
    quant_lightning_indexer_v2
    quant_lightning_indexer_v2_metadata
    sparse_flash_mla
    sparse_flash_mla_metadata
    mega_moe
    key_pool
    pool_key_indexer
    sparse_flash_attention
)

# 目标 Ascend SoC 芯片型号 (可选: ascend910b / ascend910_93 / ascend950 / ascend960dt 等)
# 可通过 --soc=xxx 参数覆盖, 未指定时使用此默认值
SOC="ascend910_93"

# 自定义算子包 vendor 名称
VENDOR_NAME="ops_transformer"

# 并行编译线程数 (自动获取当前 CPU 核数)
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)"

# 编译优化级别 (默认 O3)
OPT_LEVEL="O3"

# -----------------------------------------------------------------------------
# 路径定义
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_TRANSFORMER_DIR="${SCRIPT_DIR}/third_party/ops-transformer"
REQUIREMENTS_FILE="${OPS_TRANSFORMER_DIR}/requirements.txt"

# -----------------------------------------------------------------------------
# 函数: 安装依赖包
# -----------------------------------------------------------------------------
install_deps() {
    echo "============================================================"
    echo "[Step 1/2] 安装依赖包"
    echo "============================================================"

    if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
        echo "错误: 未找到依赖文件 ${REQUIREMENTS_FILE}"
        exit 1
    fi

    echo "读取依赖文件: ${REQUIREMENTS_FILE}"
    echo "----------------------------------------"
    cat "${REQUIREMENTS_FILE}"
    echo "----------------------------------------"
    echo "开始安装依赖..."
    pip install -r "${REQUIREMENTS_FILE}"
    echo "依赖安装完成。"
    echo ""
}

# -----------------------------------------------------------------------------
# 函数: 执行编译
# -----------------------------------------------------------------------------
build_ops() {
    echo "============================================================"
    echo "[Step 2/2] 执行编译"
    echo "============================================================"

    if [[ ! -d "${OPS_TRANSFORMER_DIR}" ]]; then
        echo "错误: 未找到 ops-transformer 目录 ${OPS_TRANSFORMER_DIR}"
        echo "请先执行 git submodule update --init --recursive 拉取子模块。"
        exit 1
    fi

    # 将 OPS 数组转换为逗号分隔字符串
    local ops_str=""
    local op
    for op in "${OPS[@]}"; do
        if [[ -z "${ops_str}" ]]; then
            ops_str="${op}"
        else
            ops_str="${ops_str},${op}"
        fi
    done

    local build_cmd="bash build.sh --pkg --soc=${SOC} --vendor_name=${VENDOR_NAME} --ops=${ops_str} -j${JOBS} -${OPT_LEVEL}"

    echo "编译目录: ${OPS_TRANSFORMER_DIR}"
    echo "编译命令:"
    echo "  ${build_cmd}"
    echo ""

    cd "${OPS_TRANSFORMER_DIR}"
    eval "${build_cmd}"
    cd "${SCRIPT_DIR}"

    echo "编译完成。"
    echo ""
}

# -----------------------------------------------------------------------------
# 解析参数: 支持 --soc=xxx 覆盖默认 SOC, 其余位置参数作为 action
# -----------------------------------------------------------------------------
parse_args() {
    local action="all"

    for arg in "$@"; do
        case "${arg}" in
            --soc=*)
                SOC="${arg#*=}"
                ;;
            --soc)
                # 需要取下一个参数, 在第二轮中处理
                ;;
            install|build|all)
                action="${arg}"
                ;;
        esac
    done

    # 第二轮: 处理 --soc 后跟空格分隔值的形式
    local idx=1
    while [[ $idx -le $# ]]; do
        local arg="${!idx}"
        if [[ "${arg}" == "--soc" ]]; then
            local next_idx=$((idx + 1))
            if [[ $next_idx -le $# ]]; then
                SOC="${!next_idx}"
            fi
        fi
        idx=$((idx + 1))
    done

    echo "${action}"
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
main() {
    local action
    action="$(parse_args "$@")"

    # 将 OPS 数组转换为逗号分隔字符串用于显示
    local ops_str=""
    local op
    for op in "${OPS[@]}"; do
        if [[ -z "${ops_str}" ]]; then
            ops_str="${op}"
        else
            ops_str="${ops_str},${op}"
        fi
    done

    echo "============================================================"
    echo "ops-transformer 构建脚本"
    echo "  SOC          : ${SOC}"
    echo "  VENDOR_NAME  : ${VENDOR_NAME}"
    echo "  OPS          : ${ops_str}"
    echo "  JOBS         : ${JOBS}"
    echo "  OPT_LEVEL    : ${OPT_LEVEL}"
    echo "============================================================"
    echo ""

    case "${action}" in
        install)
            install_deps
            ;;
        build)
            build_ops
            ;;
        all)
            install_deps
            build_ops
            ;;
        *)
            echo "用法: bash build_ops_transformer.sh [install|build|all]"
            echo "  install  仅安装依赖包"
            echo "  build    仅执行编译"
            echo "  all      安装依赖并编译 (默认)"
            exit 1
            ;;
    esac

    echo "全部完成。"
}

main "$@"