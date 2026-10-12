#!/usr/bin/env bash
# =============================================================================
# build_ops_transformer.sh
# 编译 ops-transformer 三方依赖的构建脚本
#
# 主要功能:
#   1. 安装依赖包: 读取 third_party/ops-transformer/requirements.txt 并 pip 安装
#   2. 执行编译:   bash build.sh --pkg --soc={soc} --vendor_name={vendor_name}
#                  --ops={ops} -j{jobs} -O3
#   3. 安装产物:   将编译生成的 *.run 包安装到 ${ASCEND_HOME_PATH}/opp
#   4. 注册 vendor: 校验并确保 vendor(custom) 已写入 ${ASCEND_HOME_PATH}/opp/vendors/config.ini
#
# 用法:
#   bash build_ops_transformer.sh            # 安装依赖 + 编译 + 安装产物 + 注册 vendor (默认)
#   bash build_ops_transformer.sh install    # 安装依赖 + 安装产物 + 注册 vendor
#   bash build_ops_transformer.sh build      # 仅执行编译
#   bash build_ops_transformer.sh all        # 安装依赖 + 编译 + 安装产物 + 注册 vendor
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

# 自定义算子包 vendor 名称 (编译参数 --vendor_name 使用, CMake 会追加 _transformer 后缀)
VENDOR_NAME="custom"

# 实际安装到 ${ASCEND_HOME_PATH}/opp/vendors/ 下的目录名
# (ops-transformer 的 CMake 将 VENDOR_NAME 拼成 ${VENDOR_NAME}_transformer 作为安装目录)
VENDOR_DIR_NAME="${VENDOR_NAME}_transformer"

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
    echo "[Step 1/4] 安装依赖包"
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
    echo "[Step 2/4] 执行编译"
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
# 函数: 安装编译产物 (.run 包)
# 将 third_party/ops-transformer/build_out/*.run 安装到 ${ASCEND_HOME_PATH}/opp
# -----------------------------------------------------------------------------
install_package() {
    echo "============================================================"
    echo "[Step 3/4] 安装编译产物"
    echo "============================================================"

    if [ -z "${ASCEND_HOME_PATH}" ]; then
        echo "错误: 环境变量 ASCEND_HOME_PATH 未设置, 无法确定安装路径。"
        exit 1
    fi

    local build_out_dir="${OPS_TRANSFORMER_DIR}/build_out"
    if [[ ! -d "${build_out_dir}" ]]; then
        echo "错误: 未找到编译产物目录 ${build_out_dir}"
        exit 1
    fi

    local run_files=("${build_out_dir}"/*.run)
    if [[ ${#run_files[@]} -eq 0 ]]; then
        echo "错误: 目录 ${build_out_dir} 下未找到 .run 包"
        exit 1
    fi
    if [[ ${#run_files[@]} -gt 1 ]]; then
        echo "错误: 目录 ${build_out_dir} 下存在多个 .run 包:"
        printf '%s\n' "${run_files[@]}"
        exit 1
    fi

    local run_file="${run_files[0]}"
    local opp_install_path="${ASCEND_HOME_PATH}/opp"

    echo "安装包: ${run_file}"
    echo "安装路径: ${opp_install_path}"
    echo ""

    bash "${run_file}" --quiet --install-path="${opp_install_path}"

    echo "产物安装完成。"
    echo ""

    update_config_ini
}

# -----------------------------------------------------------------------------
# 函数: 校验并确保 vendor 已注册到 config.ini
# .run 包安装时通常会写入 ${OPP}/vendors/config.ini 的 load_priority,
# 此处作为兜底: 若缺失则显式补充, 保证自定义算子包被加载。
# -----------------------------------------------------------------------------
update_config_ini() {
    local opp_install_path="${ASCEND_HOME_PATH}/opp"
    local config_file="${opp_install_path}/vendors/config.ini"

    echo "============================================================"
    echo "[Step 4/4] 校验 config.ini 中的 vendor 注册"
    echo "============================================================"

    if [[ ! -f "${config_file}" ]]; then
        echo "警告: 未找到 ${config_file}, 尝试创建并写入 load_priority。"
        if ! mkdir -p "$(dirname "${config_file}")"; then
            echo "错误: 无法创建目录 $(dirname "${config_file}")"
            exit 1
        fi
        if ! echo "load_priority=${VENDOR_DIR_NAME}" > "${config_file}"; then
            echo "错误: 写入 ${config_file} 失败, 请检查写权限 (当前用户: $(whoami))"
            exit 1
        fi
        echo "已创建 ${config_file} 并写入 load_priority=${VENDOR_DIR_NAME}"
        echo ""
        return
    fi

    # 已存在 config.ini: 检查 load_priority 是否包含本 vendor
    local found_vendors
    found_vendors="$(grep -w "load_priority" "${config_file}" | cut --only-delimited -d"=" -f2-)"

    if [[ -z "${found_vendors}" ]]; then
        # 无 load_priority 行, 追加
        if ! echo "load_priority=${VENDOR_DIR_NAME}" >> "${config_file}"; then
            echo "错误: 追加 load_priority 到 ${config_file} 失败, 请检查写权限"
            exit 1
        fi
        echo "已向 ${config_file} 追加 load_priority=${VENDOR_DIR_NAME}"
    elif ! echo "${found_vendors}" | grep -qw "${VENDOR_DIR_NAME}"; then
        # 已有 load_priority 但不含本 vendor, 将本 vendor 置顶追加
        if ! sed -i "s/^load_priority=${found_vendors}/load_priority=${VENDOR_DIR_NAME},${found_vendors}/" "${config_file}"; then
            echo "错误: 更新 ${config_file} 的 load_priority 失败, 请检查写权限"
            exit 1
        fi
        echo "已更新 ${config_file}: load_priority=${VENDOR_DIR_NAME},${found_vendors}"
    else
        echo "${config_file} 已包含 vendor [${VENDOR_DIR_NAME}], 无需更新。"
    fi

    echo "校验结果: $(grep -w 'load_priority' "${config_file}")"
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
    echo "  VENDOR_DIR   : ${VENDOR_DIR_NAME}"
    echo "  OPS          : ${ops_str}"
    echo "  JOBS         : ${JOBS}"
    echo "  OPT_LEVEL    : ${OPT_LEVEL}"
    echo "============================================================"
    echo ""

    case "${action}" in
        install)
            install_deps
            install_package
            ;;
        build)
            build_ops
            ;;
        all)
            install_deps
            build_ops
            install_package
            ;;
        *)
            echo "用法: bash build_ops_transformer.sh [install|build|all]"
            echo "  install  安装依赖包并安装编译产物"
            echo "  build    仅执行编译"
            echo "  all      安装依赖 + 编译 + 安装产物 (默认)"
            exit 1
            ;;
    esac

    echo "全部完成。"
}

main "$@"