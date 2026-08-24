#pragma once

#include <cstdio>

#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opname, param, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid argument for %s, reason: %s\n", opname, param, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opname, param, actual, expected) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid value for %s, actual: %s, expected: %s\n", opname, param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid value for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid values for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shape for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shapes for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opname, param, actual, expected) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shape dim for %s, actual: %s, expected: %s\n", opname, param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shape dim for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shape size for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid shape sizes for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid dtype for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opname, param, actual, reason) \
    fprintf(stderr, "[OP_LOGE] %s: Invalid dtypes for %s, actual: %s, reason: %s\n", opname, param, actual, reason)
#endif
