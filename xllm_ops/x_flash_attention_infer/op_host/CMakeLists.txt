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
         x_flash_attention_infer_def.cpp
    )
endif()

string(TOLOWER "${SOC_VERSION}" _XFA_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _XFA_UNIT_LOWER)
if(_XFA_SOC_LOWER MATCHES "ascend950" OR _XFA_SOC_LOWER MATCHES "ascend310p5"
   OR _XFA_UNIT_LOWER MATCHES "ascend950" OR _XFA_UNIT_LOWER MATCHES "ascend310p5")
    set(XFA_CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
else()
    set(XFA_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")
endif()

add_ops_compile_options(
    OP_NAME XFlashAttentionInfer
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            ${XFA_CATLASS_ARCH_DEF}
            -I${CANN_3RD_LIB_PATH}/catlass/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE  x_flash_attention_infer ACLNNTYPE aclnn)
endif()
