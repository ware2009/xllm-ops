# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        multi_latent_attention_def.cpp
    )
endif()

# Dynamically set CATLASS_ARCH based on the SOC being built.
# NOTE: In the CMake scope SOC_VERSION may be empty; the reliable variable is
# ASCEND_COMPUTE_UNIT (see CMakeCache, e.g. "ascend950"). We accept both and also
# any *950 / *310p5 spelling. The host tiling .cpp is compiled into a framework-
# aggregated OBJECT library defined in another directory, so neither
# add_compile_definitions nor set_source_files_properties from this subdir can
# reach it. Instead we generate a per-build config header (mla_arch_config.h)
# whose content depends on the SOC selected at CONFIGURE time: A3 gets an empty
# header (falls back to MLATiling) while A5 gets "#define CATLASS_ARCH 3510"
# (compiles the arch35 branch via MLATilingA5).
string(TOLOWER "${SOC_VERSION}" _MLA_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _MLA_UNIT_LOWER)
if(_MLA_SOC_LOWER MATCHES "ascend950" OR _MLA_SOC_LOWER MATCHES "ascend310p5"
   OR _MLA_UNIT_LOWER MATCHES "ascend950" OR _MLA_UNIT_LOWER MATCHES "ascend310p5")
    set(CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
    set(_MLA_IS_A5 TRUE)
else()
    set(CATLASS_ARCH_DEF "")
    set(_MLA_IS_A5 FALSE)
endif()

add_ops_compile_options(
    OP_NAME MultiLatentAttention
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            ${CATLASS_ARCH_DEF}
            -I${CMAKE_CURRENT_LIST_DIR}
)

# Generate per-build arch config header for host tiling sources.
# The tiling .cpp includes "mla_arch_config.h" which is generated here.
if(_MLA_IS_A5)
    set(MLA_ARCH_CONFIG_BODY "#define CATLASS_ARCH 3510")
else()
    set(MLA_ARCH_CONFIG_BODY "")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/mla_arch_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/mla_arch_config.h
    @ONLY
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE multi_latent_attention ACLNNTYPE aclnn)
endif()