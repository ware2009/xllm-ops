# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        mega_chunk_gdn_def.cpp
    )
endif()

set(MEGA_CHUNK_GDN_PROFILE_OPTIONS)
if(ASCEND_COMPUTE_UNIT MATCHES "(^|;)ascend950(;|$)" AND DEFINED ENV{MEGA_CHUNK_GDN_PROFILE_STOP})
    set(MEGA_CHUNK_GDN_PROFILE_STOP "$ENV{MEGA_CHUNK_GDN_PROFILE_STOP}")
    if(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "cumsum")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_CUMSUM)
    elseif(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "transpose")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_TRANSPOSE)
    elseif(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "kkt")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_KKT)
    elseif(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "solve")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_SOLVE)
    elseif(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "wy")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_WY)
    elseif(MEGA_CHUNK_GDN_PROFILE_STOP STREQUAL "h")
        list(APPEND MEGA_CHUNK_GDN_PROFILE_OPTIONS -DMEGA_STOP_AFTER_H)
    else()
        message(FATAL_ERROR
            "Unsupported MEGA_CHUNK_GDN_PROFILE_STOP=${MEGA_CHUNK_GDN_PROFILE_STOP}")
    endif()
endif()

add_ops_compile_options(
    OP_NAME MegaChunkGdn
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -DGDN_KERNEL_NAME=mega_chunk_gdn
            -DGDN_D=128
            -DGDN_C=128
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pto-isa/include
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    # Profiling stops must never leak into a simultaneous A2/A3 build.
    if(MEGA_CHUNK_GDN_PROFILE_OPTIONS)
        add_ops_compile_options(
            OP_NAME MegaChunkGdn
            COMPUTE_UNIT ascend950
            OPTIONS ${MEGA_CHUNK_GDN_PROFILE_OPTIONS})
    endif()
    add_modules_sources(OPTYPE mega_chunk_gdn ACLNNTYPE aclnn)
endif()
