

#ifndef CROSS_ATTENTION_COMMON
#define CROSS_ATTENTION_COMMON


constexpr uint32_t SYNC_QK_READY_ID = 1;
constexpr uint32_t SYNC_SOFTMAX_READY_ID = 2;
constexpr uint32_t SYNC_PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 4; // row * col * blockStackNum


template <typename T>
CATLASS_DEVICE T AlignUp(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

struct CrossAttnKernelParams {
    GM_ADDR q;
    GM_ADDR kCache;
    GM_ADDR vCache;
    GM_ADDR blockTable;
    GM_ADDR actualKvLens;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR o;
    GM_ADDR tiling;

    CATLASS_DEVICE
    CrossAttnKernelParams() {
    }

    CATLASS_DEVICE
    CrossAttnKernelParams(GM_ADDR q_,
    GM_ADDR kCache_,
    GM_ADDR vCache_,
    GM_ADDR blockTable_,
    GM_ADDR actualKvLens_,
    GM_ADDR s_,
    GM_ADDR p_,
    GM_ADDR oTemp_,
    GM_ADDR o_, GM_ADDR tiling_) : q(q_), kCache(kCache_), vCache(vCache_),
    blockTable(blockTable_), actualKvLens(actualKvLens_), s(s_),
    p(p_), oTemp(oTemp_), o(o_), tiling(tiling_)
    {}
};


#endif