// Compatibility shims for ops_err.h enhanced macros (CANN 9.0.0)
// These macros exist in CANN 9.1.0+ but are missing in 9.0.0
// We map them to plain OP_LOGE calls using string conversion.
#pragma once
#include "err/ops_err.h"
#include <string>
#include <sstream>

namespace qli_err_compat {
inline std::string to_str() { return ""; }
template <typename T, typename... Args>
std::string to_str(const T& v, const Args&... args) {
    std::ostringstream oss;
    oss << v;
    return oss.str() + to_str(args...);
}
}

#define QLI_ERR_MSG(...) (qli_err_compat::to_str(__VA_ARGS__).c_str())

#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, argNames, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, argNames, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, argNames, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, argName, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, argNames, ...) OP_LOGE(opName, "%s", QLI_ERR_MSG(__VA_ARGS__))
#endif
