/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "arch35/beam_search_group_apt.h"

#include <cstdint>

#include "common/common.h"
#include "kernel_operator.h"

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearchGroup<TokenIdType, LogProbType>::Init(
     GM_ADDR log_probs, GM_ADDR top_tokens, GM_ADDR top_probs,
    GM_ADDR out_token_ids, GM_ADDR out_token_index, GM_ADDR out_log_probs,
    GM_ADDR out_beam_count_prefix_sums, GM_ADDR candidate_ws,
     int32_t num_sequences, int32_t sequence_length,
    int32_t beam_width, int32_t top_k, int32_t request_num, int32_t core_num,
    int32_t min_size, int32_t step_size, TopkTiling &topKTilingData, TopkTiling &topKTilingData1, TopkTiling &topKTilingDataTail) {
  log_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)log_probs);
  top_tokens_gm.SetGlobalBuffer((__gm__ TokenIdType *)top_tokens);
  top_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)top_probs);
  out_token_ids_gm.SetGlobalBuffer((__gm__ TokenIdType *)out_token_ids);
  out_token_index_gm.SetGlobalBuffer((__gm__ int32_t *)out_token_index);
  out_log_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)out_log_probs);
  out_beam_count_prefix_sums_gm.SetGlobalBuffer(
      (__gm__ int32_t *)out_beam_count_prefix_sums);
  // Candidate pool: probs segment [core_num][align_top_k] fp32 at offset 0,
  // index segment [core_num][align_top_k] int32 right behind it. The pool
  // base is the kernel `workspace` argument, which is the user workspace
  // pointer in the framework op development mode.
  cand_probs_ws.SetGlobalBuffer((__gm__ LogProbType *)candidate_ws);
  cand_index_ws.SetGlobalBuffer(
      (__gm__ int32_t *)((__gm__ LogProbType *)candidate_ws +
                         static_cast<int64_t>(core_num) * AlignUp(top_k, 32 / sizeof(LogProbType))));
  this->num_sequences = num_sequences;
  this->sequence_length = sequence_length;
  this->beam_width = beam_width;
  this->top_k = top_k;
  this->request_num = request_num;
  this->core_num = core_num;
  this->core_idx = static_cast<int32_t>(block_idx);
  this->topKTilingData = topKTilingData;
  this->topKTilingData1 = topKTilingData1;
  this->topKTilingDataTail = topKTilingDataTail;
  this->align_beam_width1 = AlignUp(this->beam_width, 32);
  this->align_beam_width2 = AlignUp(this->beam_width, 32 / sizeof(LogProbType));
  this->align_top_k = AlignUp(this->top_k, 32 / sizeof(LogProbType));
  int32_t align_buf = this->align_beam_width1 > this->align_top_k
                          ? this->align_beam_width1 : this->align_top_k;
  this->min_size = min_size;
  this->step_size = step_size;
  // Note: the UB buffer layout below is byte-identical to the baseline
  // kernel (no additional allocations); the lane-local consolidation runs
  // as a per-round chain, so no candidate staging buffer is required.
  pipe.InitBuffer(log_probs_in_que, 1, align_buf * sizeof(LogProbType));
  pipe.InitBuffer(top_tokens_in_que, 1, step_size * align_buf * sizeof(LogProbType));
  pipe.InitBuffer(out_token_ids_out_que, 1, align_buf * sizeof(TokenIdType));
  pipe.InitBuffer(out_token_index_out_que, 1, align_buf * sizeof(TokenIdType));
  pipe.InitBuffer(out_log_probs_out_que, 1, align_buf * sizeof(LogProbType));
  pipe.InitBuffer(out_beam_count_prefix_sums_out_que, 1, align_buf * sizeof(int32_t));
  pipe.InitBuffer(top_k_result_prob_buf, align_buf * sizeof(LogProbType));
  pipe.InitBuffer(top_k_result_index_buf, align_buf * sizeof(int32_t));
  pipe.InitBuffer(prefix_probs_buf, align_buf * sizeof(LogProbType));
  pipe.InitBuffer(prefix_index_buf, align_buf * sizeof(int32_t));
  pipe.InitBuffer(top_k_second_res_buf, align_buf * sizeof(LogProbType));
  pipe.InitBuffer(merge_probs_buf, AlignUp(2 * this->align_top_k, 32) * sizeof(LogProbType));
  pipe.InitBuffer(merge_index_buf, AlignUp(2 * this->align_top_k, 32) * sizeof(int32_t));
  pipe.InitBuffer(top_k_second_res_index_buf, align_buf * sizeof(int32_t));
  pipe.InitBuffer(top_k_tmp_buf, this->min_size);
  pipe.InitBuffer(beam_counts_buf, align_buf * sizeof(int32_t));
  pipe.InitBuffer(beam_write_pos_buf, align_buf * sizeof(int32_t));
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline int32_t
BeamSearchGroup<TokenIdType, LogProbType>::AlignUp(int32_t value,
                                              int32_t alignment) {
  if (value % alignment != 0) {
    value = (value / alignment + 1) * alignment;
  }
  return value;
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::PsumDispatch(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  // Round-level dispatch, executed by every core of the request's group:
  // R = ceil(beam_width / step_size) rounds are either processed serially by
  // group lane 0 (original per-round TopK#1 + TopK#2 chain, a2-08 fallback
  // for R < BSG_MIN_PARALLEL_ROUNDS or the natural lane_count <= 1
  // degeneration) or split contiguously across the active lanes of the
  // group with pool publication and a vectorized tree merge afterwards.
  // All SyncAll call sites of the kernel live in this function.
  int32_t groups = this->request_num < this->core_num ? this->request_num : this->core_num;
  if (groups < 1) {
    groups = 1;
  }
  int32_t lane_count = this->core_num / groups;
  if (lane_count < 1) {
    lane_count = 1;
  }
  int32_t group = this->core_idx % groups;
  int32_t lane = this->core_idx / groups;
  int32_t total_rounds = (this->beam_width + this->step_size - 1) / this->step_size;
  if (total_rounds < BSG_MIN_PARALLEL_ROUNDS || lane_count <= 1) {
    // Serial fallback (a2-08): only the threshold above and the natural
    // single-lane degeneration reach this path - there is no envelope on
    // the number of rounds or on align_top_k. The prefix init was already
    // done by the caller exactly like the baseline (with the a2-05 width).
    if (lane == 0) {
      PsumSerial(request_idx, prefix_top_probs, prefix_top_index);
    }
    return;
  }
  int32_t active_lanes = lane_count < total_rounds ? lane_count : total_rounds;
  if (active_lanes < 1) {
    active_lanes = 1;
  }
  int32_t round_begin = 0;
  int32_t round_end = 0;
  if (lane < active_lanes) {
    int32_t rounds_per_lane = total_rounds / active_lanes;
    int32_t remainder = total_rounds - rounds_per_lane * active_lanes;
    round_begin = lane * rounds_per_lane + (lane < remainder ? lane : remainder);
    round_end = round_begin + rounds_per_lane + (lane < remainder ? 1 : 0);
  }
  if (lane < active_lanes) {
    PsumParallel(request_idx, prefix_top_probs, prefix_top_index, round_begin, round_end);
    PublishCandidates(prefix_top_probs, prefix_top_index);
  }
  // Vectorized tree merge across the group lanes (a2-07): level s merges the
  // slot pair (lane, lane + s); the winner (lane % (2*s) == 0 and
  // lane + s < active_lanes) reads the peer pool slot, merges it with its
  // own locally maintained prefix through the baseline TopK#2 vector merge
  // (isInitIndex = true, value/index stay paired with -inf padding) and
  // republishes the result to its own slot. Every level is preceded by one
  // full-core SyncAll so the peer slot writes of the previous level are
  // visible; the loop is executed by every core (active, inactive and
  // non-winner alike), so all cores execute exactly the same number of
  // barriers: ceil(log2(active_lanes)) SyncAll calls. Only active slots
  // are ever read - inactive lanes never publish. After the last level the
  // group lane 0 prefix holds the group-global top-k (truncation law:
  // TopK(u Ai) = TopK(u TopK(Ai))) and is consumed by StackWithOutput.
  for (int32_t s = 1; s < active_lanes; s <<= 1) {
    AscendC::SyncAll();
    bool is_winner = (lane % (2 * s) == 0) && (lane + s < active_lanes);
    if (is_winner) {
      int32_t peer_core = group + (lane + s) * groups;
      MergePeerSlot(prefix_top_probs, prefix_top_index, peer_core);
      PublishCandidates(prefix_top_probs, prefix_top_index);
    }
  }
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearchGroup<TokenIdType, LogProbType>::Process() {
  // Original Process structure preserved (thin body): the request loop runs
  // on every core with request_idx = core_idx % groups + k * groups (stride
  // groups; request_num <= 48 < core_num gives exactly one request per
  // core); the prefix initialization is applied per request with the a2-05
  // width, and PsumDispatch handles serial/parallel rounds; group lane 0
  // emits the outputs after PsumDispatch returns.
  int32_t groups = this->request_num < this->core_num ? this->request_num : this->core_num;
  if (groups < 1) {
    groups = 1;
  }
  int32_t lane = this->core_idx / groups;
  int32_t total_rounds_ = (this->beam_width + this->step_size - 1) / this->step_size;
  int32_t lane_count_ = this->core_num / groups;
  bool serial_shape = total_rounds_ < BSG_MIN_PARALLEL_ROUNDS || lane_count_ <= 1;
  // a2-05 (baseline defect D2 fix): initialize both prefix buffers over the
  // full width max(align_beam_width1, align_top_k) (= align_buf, the very
  // width the two prefix buffers are allocated with) so that the round-0
  // TopK#2 merge, which reads align_top_k elements, never touches
  // uninitialized memory when align_top_k > align_beam_width1. The width is
  // identical on the serial fallback path, the parallel path and every
  // prefix consumer (lane publication reads align_top_k <= this width).
  int32_t prefix_init_width = this->align_beam_width1 > this->align_top_k
                                  ? this->align_beam_width1 : this->align_top_k;
  for (int32_t request_idx = this->core_idx % groups; request_idx < this->request_num;
       request_idx += groups) {
    if (serial_shape) {
      // Serial shape: only the handling core (group lane 0 == the baseline
      // request mapping) touches the prefix buffers, exactly like the
      // baseline kernel.
      if (lane == 0) {
        LocalTensor<LogProbType> prefix_top_probs = prefix_probs_buf.Get<LogProbType>();
        AscendC::Duplicate<LogProbType>(prefix_top_probs, static_cast<LogProbType>(-1.0f / 0.0f), prefix_init_width);
        LocalTensor<int32_t> prefix_top_index = prefix_index_buf.GetWithOffset<int32_t>(prefix_init_width, 0);
        AscendC::Duplicate<int32_t>(prefix_top_index, static_cast<int32_t>(0), prefix_init_width);
        PsumSerial(request_idx, prefix_top_probs, prefix_top_index);
        StackWithOutput(request_idx, prefix_top_probs, prefix_top_index);
      }
    } else {
      LocalTensor<LogProbType> prefix_top_probs = prefix_probs_buf.Get<LogProbType>();
      AscendC::Duplicate<LogProbType>(prefix_top_probs, static_cast<LogProbType>(-1.0f / 0.0f), prefix_init_width);
      LocalTensor<int32_t> prefix_top_index = prefix_index_buf.GetWithOffset<int32_t>(prefix_init_width, 0);
      AscendC::Duplicate<int32_t>(prefix_top_index, static_cast<int32_t>(0), prefix_init_width);
      PsumDispatch(request_idx, prefix_top_probs, prefix_top_index);
      if (lane == 0) {
        StackWithOutput(request_idx, prefix_top_probs, prefix_top_index);
      }
    }
  }
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::AlignUpDataCopyGm(
    AscendC::GlobalTensor<TokenIdType> dst,
    AscendC::LocalTensor<TokenIdType> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(TokenIdType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::AlignUpDataCopyGmInt(
    AscendC::GlobalTensor<int32_t> dst,
    AscendC::LocalTensor<int32_t> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(int32_t);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearchGroup<TokenIdType, LogProbType>::AlignUpDataCopyGmLog(
    AscendC::GlobalTensor<LogProbType> dst,
    AscendC::LocalTensor<LogProbType> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::AlignUpDataCopyProb(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length) {
  int32_t block_size = AlignUp(length, 32 / sizeof(LogProbType));
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::AlignUpDataCopyProbSlice(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length,
    int32_t slice_length) {
  int32_t block_size = AlignUp(length, 32 / sizeof(LogProbType)); // actully equals this->align_beam_width2
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = -1.0f / 0.0f;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = slice_length;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
  AscendC::PipeBarrier<PIPE_MTE2>();
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::PsumSerial(int32_t request_idx,
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  // Original serial Psum, verbatim semantics: the request's log_probs row is
  // loaded once, every round applies the Adds broadcast and TopK#1, and
  // TopKWithSorted merges the round candidates into the prefix chain.
  LocalTensor<LogProbType> log_probs_local = log_probs_in_que.AllocTensor<LogProbType>();
  AlignUpDataCopyProb(log_probs_local, log_probs_gm[request_idx * this->beam_width], this->beam_width);
  log_probs_in_que.EnQue(log_probs_local);
  int32_t beam_width_round = (this->beam_width + this->step_size - 1) / this->step_size;
  LocalTensor<LogProbType> log_probs_per = log_probs_in_que.DeQue<LogProbType>();
  int32_t tail_number = this->beam_width % this->step_size;
  for (int32_t i = 0; i < beam_width_round; i++) {
    LocalTensor<LogProbType> top_probs_local = top_tokens_in_que.AllocTensor<LogProbType>();

    // We need to make sure the TopK input are initialized
    // TopK block_size is aligned to 32
    bool is_last_round_with_tail = (i == beam_width_round - 1 && tail_number != 0);
    int32_t round_size = is_last_round_with_tail ? tail_number : this->step_size;
    int32_t each_round_aligned_probs_cnt = AlignUp(this->step_size * this->align_beam_width2, 32);
    if (is_last_round_with_tail) {
      // Note: This is an issue of AscendC:
      // The TopK op uses both topKInfo and topKTilingData as the input config
      // TopK will use the block_size from topKTilingData, which leads to data reuse from previous round
      // Here we simply reinit [top_probs_local, end)
      AscendC::Duplicate<LogProbType>(top_probs_local, static_cast<LogProbType>(-1.0f / 0.0f), each_round_aligned_probs_cnt);
    } else {
      LocalTensor top_probs_local_last_block = top_probs_local[each_round_aligned_probs_cnt - 32];
      AscendC::Duplicate<LogProbType>(top_probs_local_last_block, static_cast<LogProbType>(-1.0f / 0.0f), 32);
    }
    int32_t eventIDVToMTE2 = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);

    AlignUpDataCopyProbSlice(
      top_probs_local,
      top_probs_gm[request_idx * this->beam_width * this->beam_width + i * this->step_size * this->beam_width], this->beam_width,
      round_size);
    top_tokens_in_que.EnQue(top_probs_local);

    LocalTensor<LogProbType> top_probs_local_tmp2 = top_tokens_in_que.DeQue<LogProbType>();
    for (int beam_width_offset = 0; beam_width_offset < round_size;
         beam_width_offset++) {
      int32_t log_probs_beam_offset = i * this->step_size + beam_width_offset;
      LogProbType log_probs_value = log_probs_per.GetValue(log_probs_beam_offset);
      AscendC::Adds<LogProbType>(
          top_probs_local_tmp2[beam_width_offset * this->align_beam_width2],
          top_probs_local_tmp2[beam_width_offset * this->align_beam_width2],
          log_probs_value, this->beam_width);
    }
    TopKWithSorted(request_idx, top_probs_local_tmp2, prefix_top_probs,
                   prefix_top_index, round_size, i);
    top_tokens_in_que.FreeTensor(top_probs_local_tmp2);
  }

  log_probs_in_que.FreeTensor(log_probs_per);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::PsumParallel(int32_t request_idx,
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index,
    int32_t round_begin, int32_t round_end) {
  // Parallel Psum over the contiguous round slice [round_begin, round_end):
  // per-round dataflow is identical to the original Psum; the round's
  // TopK#1 output is merged straight into the lane-local prefix with the
  // baseline TopK#2 vector merge (per-round chain, no staging buffer, so
  // the UB layout stays byte-identical to the baseline kernel).
  LocalTensor<LogProbType> log_probs_local = log_probs_in_que.AllocTensor<LogProbType>();
  AlignUpDataCopyProb(log_probs_local, log_probs_gm[request_idx * this->beam_width], this->beam_width);
  log_probs_in_que.EnQue(log_probs_local);
  LocalTensor<LogProbType> log_probs_per = log_probs_in_que.DeQue<LogProbType>();

  LocalTensor<LogProbType> dst_local_value = top_k_result_prob_buf.Get<LogProbType>();
  LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>();

  int32_t total_rounds = (this->beam_width + this->step_size - 1) / this->step_size;
  int32_t tail_number = this->beam_width % this->step_size;
  for (int32_t i = round_begin; i < round_end; i++) {
    LocalTensor<LogProbType> top_probs_local = top_tokens_in_que.AllocTensor<LogProbType>();
    bool is_last_round_with_tail = (i == total_rounds - 1 && tail_number != 0);
    int32_t round_size = is_last_round_with_tail ? tail_number : this->step_size;
    int32_t each_round_aligned_probs_cnt = AlignUp(this->step_size * this->align_beam_width2, 32);
    if (is_last_round_with_tail) {
      AscendC::Duplicate<LogProbType>(top_probs_local, static_cast<LogProbType>(-1.0f / 0.0f), each_round_aligned_probs_cnt);
    } else {
      LocalTensor top_probs_local_last_block = top_probs_local[each_round_aligned_probs_cnt - 32];
      AscendC::Duplicate<LogProbType>(top_probs_local_last_block, static_cast<LogProbType>(-1.0f / 0.0f), 32);
    }
    int32_t eventIDVToMTE2 = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDVToMTE2);

    AlignUpDataCopyProbSlice(
      top_probs_local,
      top_probs_gm[request_idx * this->beam_width * this->beam_width + i * this->step_size * this->beam_width], this->beam_width,
      round_size);
    top_tokens_in_que.EnQue(top_probs_local);

    LocalTensor<LogProbType> top_probs_local_tmp2 = top_tokens_in_que.DeQue<LogProbType>();
    for (int beam_width_offset = 0; beam_width_offset < round_size;
         beam_width_offset++) {
      int32_t log_probs_beam_offset = i * this->step_size + beam_width_offset;
      LogProbType log_probs_value = log_probs_per.GetValue(log_probs_beam_offset);
      AscendC::Adds<LogProbType>(
          top_probs_local_tmp2[beam_width_offset * this->align_beam_width2],
          top_probs_local_tmp2[beam_width_offset * this->align_beam_width2],
          log_probs_value, this->beam_width);
    }
    RoundTopK(top_probs_local_tmp2, dst_local_value, dst_local_index, round_size, i);
    top_tokens_in_que.FreeTensor(top_probs_local_tmp2);
    // Merge the round candidates straight into the lane-local prefix
    // (baseline TopK#2 vector merge, isInitIndex = true).
    MergeCandidates(prefix_top_probs, prefix_top_index, dst_local_value, dst_local_index);
  }

  log_probs_in_que.FreeTensor(log_probs_per);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::RoundTopK(
    AscendC::LocalTensor<LogProbType> &top_probs_local,
    AscendC::LocalTensor<LogProbType> &dst_local_value,
    AscendC::LocalTensor<int32_t> &dst_local_index,
    int32_t round_length, int32_t round_idx) {
  // TopK#1 with exactly the original semantics: k = align_top_k over the
  // round block, indices offset by the global row base so that the merged
  // index keeps the row * align_beam_width2 + col encoding.
  LocalTensor<int32_t> src_local_index;
  LocalTensor<bool> src_local_finish;
  int32_t block_size = AlignUp(round_length * this->align_beam_width2, 32);
  TopKInfo topKInfo;
  topKInfo.outter = 1;
  topKInfo.inner = block_size;
  topKInfo.n = round_length * this->align_beam_width2;
  bool isLargest = true;
  LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();

  AscendC::TopK<LogProbType, false, false, false, AscendC::TopKMode::TOPK_NORMAL>(
      dst_local_value, dst_local_index, top_probs_local, src_local_index,
      src_local_finish, tmp_local, this->align_top_k,
      round_length < this->step_size ? this->topKTilingDataTail : this->topKTilingData,
      topKInfo, isLargest);
  AscendC::Adds<int32_t>(dst_local_index, dst_local_index,
                         static_cast<int32_t>((round_idx * this->step_size) * this->align_beam_width2),
                         this->align_top_k);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::MergeCandidates(
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index,
    const AscendC::LocalTensor<LogProbType> &cand_probs,
    const AscendC::LocalTensor<int32_t> &cand_index) {
  // Baseline TopK#2 merge pattern: [cand | prefix] -> top align_top_k with
  // isInitIndex = true, so the global indices carried by both inputs are
  // propagated to the output.
  LocalTensor<LogProbType> merge_top_probs = merge_probs_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(merge_top_probs, static_cast<LogProbType>(-1.0f / 0.0f), AlignUp(2 * this->align_top_k, 32));
  AscendC::DataCopy(merge_top_probs, cand_probs, this->align_top_k);
  AscendC::DataCopy(merge_top_probs[this->align_top_k], prefix_top_probs, this->align_top_k);
  LocalTensor<int32_t> merge_index = merge_index_buf.Get<int32_t>();
  AscendC::DataCopy(merge_index, cand_index, this->align_top_k);
  AscendC::DataCopy(merge_index[this->align_top_k], prefix_top_index, this->align_top_k);
  LocalTensor<LogProbType> dst_merge_probs = top_k_second_res_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(dst_merge_probs, static_cast<LogProbType>(-1.0f / 0.0f),
                              this->align_top_k);
  LocalTensor<int32_t> dst_merge_index = top_k_second_res_index_buf.Get<int32_t>();
  AscendC::Duplicate<int32_t>(dst_merge_index, static_cast<int32_t>(0),
                              this->align_top_k);
  int32_t block_size_merge = AlignUp(this->align_top_k * 2, 32);
  TopKInfo topKInfo2;
  topKInfo2.outter = 1;
  topKInfo2.inner = block_size_merge;
  topKInfo2.n = this->align_top_k * 2;
  LocalTensor<bool> src_local_finish;
  LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();

  AscendC::TopK<LogProbType, true, false, false, AscendC::TopKMode::TOPK_NORMAL>(
          dst_merge_probs,
          dst_merge_index,
          merge_top_probs,
          merge_index,
          src_local_finish,
          tmp_local,
          this->align_top_k,
          this->topKTilingData1,
          topKInfo2,
          true);

  AscendC::DataCopy(prefix_top_probs, dst_merge_probs, this->align_top_k);

  AscendC::DataCopy(prefix_top_index, dst_merge_index, this->align_top_k);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::TopKWithSorted(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &top_probs_local,
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index, int32_t round_length, int32_t round_idx) {
  // Original per-round merge helper (serial fallback path), verbatim.
  LocalTensor<LogProbType> dst_local_value =
      top_k_result_prob_buf.Get<LogProbType>();
  LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>(); // [this->align_top_k]
  LocalTensor<int32_t> src_local_index;
  LocalTensor<bool> src_local_finish;
  int32_t block_size = AlignUp(round_length * this->align_beam_width2, 32);
  TopKInfo topKInfo;
  topKInfo.outter = 1;
  topKInfo.inner = block_size;
  topKInfo.n = round_length * this->align_beam_width2;
  bool isLargest = true;
  LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();

  AscendC::TopK<LogProbType, false, false, false,AscendC::TopKMode::TOPK_NORMAL>(
      dst_local_value, dst_local_index, top_probs_local, src_local_index,
      src_local_finish, tmp_local, this->align_top_k, round_length < this->step_size ? this->topKTilingDataTail : this->topKTilingData, topKInfo,
      isLargest);
  AscendC::Adds<int32_t>(dst_local_index, dst_local_index,
                         static_cast<int32_t>((round_idx * this->step_size) * this->align_beam_width2),
                         this->align_top_k);
  LocalTensor<LogProbType> merge_top_probs = merge_probs_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(merge_top_probs, static_cast<LogProbType>(-1.0f / 0.0f), AlignUp(2 * this->align_top_k, 32));
  AscendC::DataCopy(merge_top_probs, dst_local_value, this->align_top_k);
  AscendC::DataCopy(merge_top_probs[this->align_top_k], prefix_top_probs, this->align_top_k);
  LocalTensor<int32_t> merge_index = merge_index_buf.Get<int32_t>();
  AscendC::DataCopy(merge_index, dst_local_index, this->align_top_k);
  AscendC::DataCopy(merge_index[this->align_top_k], prefix_top_index, this->align_top_k);
  LocalTensor<LogProbType> dst_merge_probs = top_k_second_res_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(dst_merge_probs, static_cast<LogProbType>(-1.0f / 0.0f),
                              this->align_top_k);
  LocalTensor<int32_t> dst_merge_index = top_k_second_res_index_buf.Get<int32_t>();
  AscendC::Duplicate<int32_t>(dst_merge_index, static_cast<int32_t>(0),
                              this->align_top_k);
  int32_t block_size_merge = AlignUp(this->align_top_k * 2, 32);
  TopKInfo topKInfo2;
  topKInfo2.outter = 1;
  topKInfo2.inner = block_size_merge;
  topKInfo2.n = this->align_top_k * 2;

  AscendC::TopK<LogProbType, true, false, false,AscendC::TopKMode::TOPK_NORMAL>(
          dst_merge_probs,
          dst_merge_index,
          merge_top_probs,
          merge_index,
          src_local_finish,
          tmp_local,
          this->align_top_k,
          this->topKTilingData1,
          topKInfo2,
          isLargest);

  AscendC::DataCopy(prefix_top_probs, dst_merge_probs, this->align_top_k);

  AscendC::DataCopy(prefix_top_index, dst_merge_index, this->align_top_k);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::MergePeerSlot(
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index, int32_t peer_core) {
  // Read the peer lane's pool slot (probs + index, exactly align_top_k pairs
  // published by the peer) and merge it into the local prefix. align_top_k is
  // always a multiple of 8, so the count-form DataCopy is safe here.
  LocalTensor<LogProbType> peer_probs = top_k_result_prob_buf.Get<LogProbType>();
  LocalTensor<int32_t> peer_index = top_k_result_index_buf.Get<int32_t>();
  int64_t peer_slot = static_cast<int64_t>(peer_core) * this->align_top_k;
  AscendC::DataCopy(peer_probs, cand_probs_ws[peer_slot], this->align_top_k);
  AscendC::DataCopy(peer_index, cand_index_ws[peer_slot], this->align_top_k);
  int32_t eventIDMTE2ToV = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
  MergeCandidates(prefix_top_probs, prefix_top_index, peer_probs, peer_index);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearchGroup<TokenIdType, LogProbType>::PublishCandidates(
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  // Publish exactly align_top_k (prob, index) pairs to this core's pool slot
  // (the prefix already carries -inf padding pairs beyond the real
  // candidates, so value/index stay paired).
  int32_t eventIDVToMTE3 = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
  int64_t slot = static_cast<int64_t>(this->core_idx) * this->align_top_k;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockCount = 1;
  copyParams.blockLen = this->align_top_k * sizeof(LogProbType);
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(cand_probs_ws[slot], prefix_top_probs, copyParams);
  AscendC::DataCopyPad(cand_index_ws[slot], prefix_top_index, copyParams);
}

// New version: Output topk token ids and log_probs
// out_token_ids_gm shape is [request_num * beam_width]
// out_token_index_gm shape is [request_num * beam_width]
// out_log_probs_gm shape is [request_num * beam_width]

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearchGroup<TokenIdType, LogProbType>::StackWithOutput(
     int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
     AscendC::LocalTensor<int32_t> &prefix_top_index) {
  // reuse some buffers from TopK
  LocalTensor<LogProbType> align_top_k_vector_float = merge_probs_buf.GetWithOffset<LogProbType>(this->align_top_k, 0);
  AscendC::Duplicate<float>(align_top_k_vector_float, static_cast<float>(this->align_beam_width2), this->align_top_k);
  LocalTensor<float> align_top_k_vector_cp = merge_probs_buf.GetWithOffset<float>(this->align_top_k, this->align_top_k * sizeof(LogProbType));
  LocalTensor<int32_t> beam_ids = top_k_second_res_index_buf.GetWithOffset<int32_t>(this->align_top_k, 0);
  LocalTensor<int32_t> remainders = top_k_result_index_buf.GetWithOffset<int32_t>(this->align_top_k, 0);
  AscendC::Duplicate<int32_t>(remainders, static_cast<int32_t>(this->align_beam_width2), this->align_top_k);
  LocalTensor<int32_t> beam_counts = beam_counts_buf.Get<int32_t>();
  AscendC::Duplicate<int32_t>(beam_counts, 0, this->align_top_k);
  LocalTensor<int32_t> beam_write_pos = beam_write_pos_buf.Get<int32_t>();

  AscendC::Cast(align_top_k_vector_cp, prefix_top_index, AscendC::RoundMode::CAST_RINT, this->top_k);
  AscendC::Div(align_top_k_vector_float, align_top_k_vector_cp, align_top_k_vector_float, this->top_k);
  AscendC::Cast(beam_ids, align_top_k_vector_float, AscendC::RoundMode::CAST_FLOOR, this->top_k);
  AscendC::Mul(remainders, beam_ids, remainders, this->top_k);
  AscendC::Sub(remainders, prefix_top_index, remainders, this->top_k);

  AscendC::Duplicate<int32_t>(beam_counts, 0, this->beam_width); // Prevent uninitialized values when this->beam_width > this->top_k
  for (int32_t i = 0; i < this->top_k; ++i) {
    int32_t beam_id = beam_ids.GetValue(i);
    int32_t count = beam_counts.GetValue(beam_id);
    beam_counts.SetValue(beam_id, count + 1);
  }
  LocalTensor<int32_t> beam_prefix = out_beam_count_prefix_sums_out_que.AllocTensor<int32_t>();
  int32_t running = 0;
  for (int32_t beam = 0; beam < this->beam_width; ++beam) {
    beam_write_pos.SetValue(beam, running);
    running += beam_counts.GetValue(beam);
    beam_prefix.SetValue(beam, running);
  }

  AscendC::Adds<int32_t>(beam_prefix, beam_prefix, static_cast<int32_t>(request_idx * this->beam_width), this->beam_width);
  out_beam_count_prefix_sums_out_que.EnQue(beam_prefix);
  LocalTensor<int32_t> prefix_tmp = out_beam_count_prefix_sums_out_que.DeQue<int32_t>();
  AlignUpDataCopyGmInt(
      out_beam_count_prefix_sums_gm[request_idx * this->beam_width], prefix_tmp,
      this->beam_width);
  out_beam_count_prefix_sums_out_que.FreeTensor(prefix_tmp);

  LocalTensor<TokenIdType> out_token_index_local = out_token_index_out_que.AllocTensor<TokenIdType>();
  LocalTensor<LogProbType> out_log_probs_local = out_log_probs_out_que.AllocTensor<LogProbType>();
  LocalTensor<TokenIdType> out_token_ids_local = out_token_ids_out_que.AllocTensor<TokenIdType>();

  int32_t top_token_index_offset = request_idx * this->beam_width * this->beam_width;
  for (int32_t i = 0; i < this->top_k; ++i) {
    int32_t beam_id = beam_ids.GetValue(i);
    int32_t write_pos = beam_write_pos.GetValue(beam_id);
    beam_write_pos.SetValue(beam_id, write_pos + 1);
    int32_t remainder = remainders.GetValue(i);
    out_token_index_local.SetValue(write_pos, static_cast<TokenIdType>(beam_id));
    out_log_probs_local.SetValue(write_pos, prefix_top_probs.GetValue(i));
    int32_t top_token_index = top_token_index_offset + beam_id * this->beam_width + remainder;
    // raw probs output without acc:
    // out_log_probs_local.SetValue(write_pos, top_probs_gm.GetValue(top_token_index));
    out_token_ids_local.SetValue(write_pos, top_tokens_gm.GetValue(top_token_index));
  }
  AscendC::Adds(out_token_index_local,
                out_token_index_local,
                static_cast<TokenIdType>(request_idx * this->beam_width),
                this->top_k);

  out_token_index_out_que.EnQue(out_token_index_local);
  LocalTensor<TokenIdType> out_token_index_tmp = out_token_index_out_que.DeQue<TokenIdType>();
  AlignUpDataCopyGm(out_token_index_gm[request_idx * this->top_k], out_token_index_tmp, this->top_k);
  // GetSequence(request_idx, out_token_index_tmp);
  out_token_index_out_que.FreeTensor(out_token_index_tmp);

  out_log_probs_out_que.EnQue(out_log_probs_local);
  LocalTensor<LogProbType> out_log_probs_tmp = out_log_probs_out_que.DeQue<LogProbType>();
  AlignUpDataCopyGmLog(out_log_probs_gm[request_idx * this->top_k], out_log_probs_tmp, this->top_k);
  out_log_probs_out_que.FreeTensor(out_log_probs_tmp);

  out_token_ids_out_que.EnQue(out_token_ids_local);
  LocalTensor<TokenIdType> out_token_ids_tmp = out_token_ids_out_que.DeQue<TokenIdType>();
  AlignUpDataCopyGm(out_token_ids_gm[request_idx * this->top_k], out_token_ids_tmp, this->top_k);
  out_token_ids_out_que.FreeTensor(out_token_ids_tmp);
}

__aicore__ inline void ProcessSequence::Init(GM_ADDR sequence,
                                            GM_ADDR token_index,
                                            GM_ADDR out_sequence,
                                            GM_ADDR out_token_ids,
                                            GM_ADDR top_tokens,
                                            int32_t beam_width,
                                            int32_t top_k,
                                            int32_t current_step,
                                            int32_t max_decode_step,
                                            int32_t request_num,
                                            int32_t core_num) {
  sequence_gm.SetGlobalBuffer((__gm__ int32_t *)sequence);
  token_index_gm.SetGlobalBuffer((__gm__ int32_t *)token_index);
  out_sequence_gm.SetGlobalBuffer((__gm__ int32_t *)out_sequence);
  out_token_ids_gm.SetGlobalBuffer((__gm__ int32_t *)out_token_ids);
  top_tokens_gm.SetGlobalBuffer((__gm__ int32_t *)top_tokens);
  this->beam_width = beam_width;
  this->current_step = current_step;
  this->top_k = this->current_step == 0 ? 1 : top_k;
  this->max_decode_step = max_decode_step;
  this->align_beam_width =(beam_width / 32 + 1) * 32;
  this->align_current_step = (current_step / 8 + 1) * 8;
  this->request_num = request_num;
  this->core_num = core_num;
  this->sequence_buf_alignsize = this->align_current_step * this->top_k;
  this->align_top_k = AlignUp(this->top_k, 8);
  pipe.InitBuffer(sequence_in_que, 1, this->align_current_step * this->beam_width * sizeof(int32_t));
  pipe.InitBuffer(sequence_out_que, 1, this->align_current_step * this->top_k * sizeof(int32_t));
  pipe.InitBuffer(sequence_buf, this->sequence_buf_alignsize * sizeof(int32_t) * 2); // store sequence update and token_index_offset
  pipe.InitBuffer(token_index_buf, this->top_k * sizeof(int32_t));
  pipe.InitBuffer(in_token_ids_que, 1,this->top_k * sizeof(int32_t));
  pipe.InitBuffer(out_token_ids_buf, this->top_k * sizeof(int32_t) * FLOAT_BLOCK_SIZE);
  // a2-10 (baseline defect D3 fix): the only consumer of top_tokens_buf,
  // SubProcessSeqPrefill, reads exactly beam_width int32 elements, so the
  // buffer is sized beam_width * sizeof(int32_t). The baseline
  // top_k * top_k * sizeof(int32_t) sizing was wrong in both directions:
  // decode (top_k = 1024) requested 4MB, above the physical UB of 253952B,
  // while prefill (top_k = 1) allocated only 4B for a beam_width-element
  // read.
  pipe.InitBuffer(top_tokens_buf, this->beam_width * sizeof(int32_t));
}

__aicore__ inline void ProcessSequence::SubProcessSeqPrefill(int32_t request_idx) {
    LocalTensor<int32_t> top_tokens_ids_local = top_tokens_buf.Get<int32_t>();
    AscendC::DataCopy(top_tokens_ids_local, top_tokens_gm[request_idx * this->beam_width * this->top_k],
        this->beam_width);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<int32_t> out_token_ids_local = out_token_ids_buf.Get<int32_t>();
    AscendC::Brcb(out_token_ids_local, top_tokens_ids_local,
        (this->top_k + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    tl::ascend::copy_ub_to_gm_beam_token<int32_t>(
      out_sequence_gm[request_idx * this->top_k * this->max_decode_step + this->current_step],
                out_token_ids_local,
                0,
                (this->max_decode_step - 1),
                1,
                this->top_k
    );
}

__aicore__ inline void ProcessSequence::Process() {
  // request_idx % core_num == block_idx with core_num = 56 AIV cores covers
  // request_num <= 48 (MAX_SUPPORT_REQUEST_NUM) exactly like the original
  // request_idx % 48 == block_idx mapping: every request is handled by
  // precisely one core.
  for (int32_t request_idx = 0; request_idx < this->request_num; request_idx++) {
    if (request_idx % this->core_num == block_idx) {
      if (this->current_step == 0) {
          SubProcessSeqPrefill(request_idx);
          return;
      }
      LocalTensor<int32_t> in_token_ids_local = in_token_ids_que.AllocTensor<int32_t>();
      AscendC::DataCopy(in_token_ids_local, out_token_ids_gm[request_idx * this->top_k],
        this->align_top_k);
      in_token_ids_que.EnQue(in_token_ids_local);
      LocalTensor<int32_t> in_token_ids_tmp = in_token_ids_que.DeQue<int32_t>();
      LocalTensor<int32_t> token_index_local = token_index_buf.Get<int32_t>();
      tl::ascend::copy_gm_to_ub_beam<int32_t>
                                (token_index_local, token_index_gm[request_idx * this->top_k],this->top_k,
                                  this->top_k,this->top_k,1
                                );

      in_sequence_local_origin = sequence_in_que.AllocTensor<int32_t>();
      int32_t sequence_offset = request_idx * this->beam_width * this->max_decode_step;

      tl::ascend::copy_gm_to_ub_beam_align<int32_t>
                                (in_sequence_local_origin, sequence_gm[sequence_offset],
                                  this->max_decode_step-this->current_step,
                                  0,
                                  this->current_step,
                                  this->beam_width
                                );

      sequence_in_que.EnQue(in_sequence_local_origin);
      in_sequence_local_origin = sequence_in_que.DeQue<int32_t>();
      LocalTensor<int32_t> sequence_local_update = sequence_buf.Get<int32_t>();
      gatherb_offset_local = sequence_buf.Get<int32_t>()[this->sequence_buf_alignsize];

      uint32_t repeat_times = (this->top_k + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE;
      if (this->top_k % FLOAT_BLOCK_SIZE != 0) {
        AscendC::Duplicate<int32_t>(gatherb_offset_local[(repeat_times - 1) * FLOAT_BLOCK_SIZE], 0, FLOAT_BLOCK_SIZE);
      }
      // call origin offset for gatherb

      // get relative index offset in per request
      int32_t token_sub_value = 0 - request_idx * this->beam_width;
      AscendC::Adds(token_index_local, token_index_local,
      static_cast<int32_t>(token_sub_value), this->top_k);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Muls(gatherb_offset_local, token_index_local,
      static_cast<int32_t>(sizeof(float) * FLOAT_BLOCK_SIZE), this->top_k);

      AscendC::PipeBarrier<PIPE_V>();
      // update sequence
      tl::ascend::copy_ub_to_ub_beam<int32_t, int32_t>
                (sequence_local_update, in_sequence_local_origin, this->sequence_buf_alignsize);
      AscendC::PipeBarrier<PIPE_V>();
      // TODO Need to consider beam_width not aligned to 8
      for (uint32_t i = 0; i < repeat_times; ++i) {
          // per loop update 8 datablock
        AscendC::Gatherb(sequence_local_update.template ReinterpretCast<uint32_t>()[i * FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE],
            in_sequence_local_origin.template ReinterpretCast<uint32_t>(),
            gatherb_offset_local.template ReinterpretCast<uint32_t>()[i * FLOAT_BLOCK_SIZE],
          1, AscendC::GatherRepeatParams(1, 8));
      }
      AscendC::PipeBarrier<PIPE_V>();
      out_sequence_local = sequence_out_que.AllocTensor<int32_t>();
      tl::ascend::copy_ub_to_ub_beam<int32_t, int32_t>
                      (out_sequence_local,
                        sequence_local_update,
                        this->align_current_step * this->top_k);
      sequence_out_que.EnQue(out_sequence_local);
      out_sequence_local = sequence_out_que.DeQue<int32_t>();

      tl::ascend::copy_ub_to_gm_beam<int32_t>
                      (out_sequence_gm[request_idx * this->top_k * this->max_decode_step],
                         out_sequence_local,
                         0,                                           // srcStride
                         this->max_decode_step - this->current_step,  // dstStride
                         this->current_step,                          // srcN
                         this->top_k);                           // srcM
      LocalTensor<int32_t> out_token_ids_local = out_token_ids_buf.Get<int32_t>();
      AscendC::Brcb(out_token_ids_local, in_token_ids_tmp,
        (this->top_k + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));

      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      tl::ascend::copy_ub_to_gm_beam_token<int32_t>(
        out_sequence_gm[request_idx * this->top_k * this->max_decode_step + this->current_step],
                  out_token_ids_local,
                  0,
                  (this->max_decode_step - 1),
                  1,
                  this->top_k
      );
      sequence_in_que.FreeTensor(in_sequence_local_origin);
      sequence_out_que.FreeTensor(out_sequence_local);
      in_token_ids_que.FreeTensor(in_token_ids_tmp);
    }
  }
}
extern "C" __global__ __aicore__ void
beam_search_group(GM_ADDR log_probs, GM_ADDR top_tokens, GM_ADDR top_probs, GM_ADDR sequence,
            GM_ADDR out_token_ids, GM_ADDR out_token_index, GM_ADDR out_log_probs, GM_ADDR out_beam_count_prefix_sums,
            GM_ADDR out_sequence,
            GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
  GET_TILING_DATA(tiling_data, tiling);
  if (tiling_data.current_step != 0) {
      BeamSearchGroup<int32_t, float> op;
      op.Init(log_probs, top_tokens, top_probs, out_token_ids, out_token_index,
              out_log_probs, out_beam_count_prefix_sums,
              workspace,
              tiling_data.num_sequences, tiling_data.sequence_length,
              tiling_data.beam_width, tiling_data.top_k, tiling_data.request_num,
              tiling_data.core_num, tiling_data.min_size, tiling_data.step_size,
              tiling_data.topKTilingData, tiling_data.topKTilingData1, tiling_data.topKTilingDataTail);
      op.Process();
      AscendC::SyncAll();
      op.pipe.Destroy();
  }
  ProcessSequence process_sequence;
  process_sequence.Init(sequence, out_token_index, out_sequence, out_token_ids, top_tokens,
                        tiling_data.beam_width, tiling_data.top_k, tiling_data.current_step,
                        tiling_data.max_decode_step,tiling_data.request_num,
                        tiling_data.core_num);
  process_sequence.Process();
}
