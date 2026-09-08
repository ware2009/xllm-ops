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

// Ascend950 (arch35) adaptation of beam_search_group:
// - The per-request serial Psum rounds are parallelized across the AI Vector
//   cores: groups = min(request_num, core_num), lane_count = core_num/groups,
//   core (group, lane) handles request base+group (stride groups) and the
//   contiguous round slice [round_begin, round_end) of that request, where
//   the round count is R = ceil(beam_width / step_size) with exactly the
//   original per-round semantics (step_size rows x beam_width cols slice of
//   top_probs, log_probs row loaded once per lane per request, scalar
//   GetValue + Adds broadcast, TopK#1 with k = align_top_k and the global
//   index offset round_idx * step_size * align_beam_width2).
// - Lane-local consolidation is a per-round chain (a2-06): each round's
//   TopK#1 candidates are merged straight into the lane-local prefix with
//   the baseline TopK#2 vector merge, so the UB buffer layout stays
//   byte-identical to the baseline kernel (no candidate staging buffer).
// - Each active lane publishes exactly align_top_k (prob, index) pairs into
//   the GM user-workspace candidate pool: [core_num][align_top_k] fp32
//   probs followed by [core_num][align_top_k] int32 indices, slot offset =
//   core * align_top_k in both segments; the pool base is the kernel
//   `workspace` argument (user workspace pointer in the framework op
//   development mode).
// - The group lanes tree-merge the pool slots with the baseline TopK#2
//   vector merge (TopK<float, true, false, false, TOPK_NORMAL>,
//   isInitIndex = true) as required by a2-07: at level s = 1, 2, 4, ...
//   < active_lanes the winner (lane % (2*s) == 0 && lane + s < active_lanes)
//   merges its peer's slot into its own slot; every level is preceded by
//   one full-core SyncAll and every core (active, inactive, non-winner)
//   executes exactly the same number of barriers.
// - R < BSG_MIN_PARALLEL_ROUNDS (or the natural lane_count <= 1
//   degeneration) falls back to the original serial per-round Psum
//   (TopK#1 + TopK#2 prefix merge chain) on group lane 0; the other cores
//   only participate in the uniform barrier sequence. Per a2-08 this is
//   the only fallback condition: no stability envelope on the number of
//   rounds or on align_top_k.
// - Round-1 confirmed baseline defects fixed inside this adaptation:
//   D2 (a2-05) - the prefix Duplicate initialization width is
//   max(AlignUp(bw, 32), align_top_k) on every path (serial fallback and
//   parallel), so the round-0 TopK#2 merge never reads uninitialized tail
//   elements; D3 (a2-10) - ProcessSequence's top_tokens_buf is sized
//   beam_width * sizeof(int32_t) because its only consumer reads
//   beam_width elements (the baseline top_k*top_k sizing over-allocates
//   4MB for decode top_k=1024, above the physical UB).
// - ascend910b / ascend910_93 keep the original shared kernel
//   (op_kernel/beam_search_group.cpp + beam_search_group.h), which stays
//   byte-identical; this file is only compiled for ascend950 through the
//   opFile.value = "beam_search_group_apt" routing in op_host/beam_search_group_def.cpp.

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
// Minimum number of serial rounds for the parallel lane decomposition to be
// used (a2-08); below this threshold - or when a group naturally has a
// single lane - group lane 0 runs the original serial Psum. This is the
// only fallback envelope: no upper bound on rounds or align_top_k.
constexpr int32_t BSG_MIN_PARALLEL_ROUNDS = 8;

using namespace AscendC;
template <typename TokenIdType, typename LogProbType> class BeamSearchGroup {
public:
  __aicore__ inline BeamSearchGroup() {}
  __aicore__ inline void
  Init(GM_ADDR log_probs, GM_ADDR top_tokens,
       GM_ADDR top_probs, GM_ADDR out_token_ids, GM_ADDR out_token_index,
       GM_ADDR out_log_probs, GM_ADDR out_beam_count_prefix_sums,
       GM_ADDR candidate_ws,
       int32_t num_sequences, int32_t sequence_length,
       int32_t beam_width, int32_t top_k, int32_t request_num, int32_t core_num,
       int32_t min_size, int32_t step_size, TopkTiling &topKTilingData,
       TopkTiling &topKTilingData1, TopkTiling &topKTilingDataTail);
  __aicore__ inline void Process();

  __aicore__ inline int32_t AlignUp(int32_t value, int32_t alignment);
  __aicore__ inline void
  AlignUpDataCopyGm(AscendC::GlobalTensor<TokenIdType> dst,
                    AscendC::LocalTensor<TokenIdType> src, int32_t length);
  __aicore__ inline void
  AlignUpDataCopyGmInt(AscendC::GlobalTensor<int32_t> dst,
                       AscendC::LocalTensor<int32_t> src, int32_t length);
  __aicore__ inline void
  AlignUpDataCopyGmLog(AscendC::GlobalTensor<LogProbType> dst,
                       AscendC::LocalTensor<LogProbType> src, int32_t length);

  __aicore__ inline void
  AlignUpDataCopyProb(AscendC::LocalTensor<LogProbType> dst,
                      AscendC::GlobalTensor<LogProbType> src, int32_t length);
  __aicore__ inline void
  AlignUpDataCopyProbSlice(AscendC::LocalTensor<LogProbType> dst,
                           AscendC::GlobalTensor<LogProbType> src,
                           int32_t length, int32_t slice_length);

  // Original serial per-round Psum: TopK#1 over each round plus TopK#2
  // prefix merge chain (semantics identical to the baseline kernel).
  __aicore__ inline void
  PsumSerial(int32_t request_idx,
             AscendC::LocalTensor<LogProbType> &prefix_top_probs,
             AscendC::LocalTensor<int32_t> &prefix_top_index);
  // Round-level dispatch executed by every core of the request's group:
  // serial fallback or parallel lane slicing + pool publication +
  // per-level vectorized tree merge (all barrier call sites live here;
  // Process() keeps the original thin shape).
  __aicore__ inline void
  PsumDispatch(int32_t request_idx,
               AscendC::LocalTensor<LogProbType> &prefix_top_probs,
               AscendC::LocalTensor<int32_t> &prefix_top_index);
  // Parallel Psum over the contiguous round slice [round_begin, round_end):
  // each round's TopK#1 candidates are merged straight into the lane-local
  // prefix with the baseline TopK#2 vector merge (per-round chain, no
  // staging buffer; a2-06).
  __aicore__ inline void
  PsumParallel(int32_t request_idx,
               AscendC::LocalTensor<LogProbType> &prefix_top_probs,
               AscendC::LocalTensor<int32_t> &prefix_top_index,
               int32_t round_begin, int32_t round_end);
  // TopK#1 over one round's (round_size x align_beam_width2) probabilities
  // with the global index offset round_idx * step_size * align_beam_width2.
  __aicore__ inline void
  RoundTopK(AscendC::LocalTensor<LogProbType> &top_probs_local,
            AscendC::LocalTensor<LogProbType> &dst_local_value,
            AscendC::LocalTensor<int32_t> &dst_local_index,
            int32_t round_length, int32_t round_idx);
  // Baseline TopK#2 merge: top align_top_k of [cand | prefix] with
  // isInitIndex = true, written back into the lane-local prefix.
  __aicore__ inline void
  MergeCandidates(AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                  AscendC::LocalTensor<int32_t> &prefix_top_index,
                  const AscendC::LocalTensor<LogProbType> &cand_probs,
                  const AscendC::LocalTensor<int32_t> &cand_index);
  // Original per-round merge helper used by the serial fallback path.
  __aicore__ inline void
  TopKWithSorted(int32_t request_idx,
                 AscendC::LocalTensor<LogProbType> &top_probs_local,
                 AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                 AscendC::LocalTensor<int32_t> &prefix_top_index,
                 int32_t step_size, int32_t round_idx);
  // Read the peer pool slot (probs + index) and merge it into the prefix.
  __aicore__ inline void
  MergePeerSlot(AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                AscendC::LocalTensor<int32_t> &prefix_top_index,
                int32_t peer_core);
  // Publish exactly align_top_k (prob, index) pairs of the lane-local prefix
  // to this core's pool slot.
  __aicore__ inline void
  PublishCandidates(AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                    AscendC::LocalTensor<int32_t> &prefix_top_index);
  __aicore__ inline void
  StackWithOutput(int32_t request_idx,
                  AscendC::LocalTensor<LogProbType> &prefix_top_probs,
                  AscendC::LocalTensor<int32_t> &prefix_top_index);
  AscendC::TPipe pipe;

private:
  AscendC::GlobalTensor<LogProbType> log_probs_gm;
  AscendC::GlobalTensor<TokenIdType> top_tokens_gm;
  AscendC::GlobalTensor<LogProbType> top_probs_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_ids_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_index_gm;
  AscendC::GlobalTensor<LogProbType> out_log_probs_gm;
  AscendC::GlobalTensor<int32_t> out_beam_count_prefix_sums_gm;
  // Candidate pool in the GM user workspace:
  // [core_num][align_top_k] fp32 probs at offset 0, then
  // [core_num][align_top_k] int32 indices; slot offset = core * align_top_k.
  AscendC::GlobalTensor<LogProbType> cand_probs_ws;
  AscendC::GlobalTensor<int32_t> cand_index_ws;
  int32_t num_sequences;
  int32_t sequence_length;
  int32_t beam_width;
  int32_t top_k;
  int32_t request_num;
  int32_t core_num;
  int32_t core_idx;
  int32_t align_beam_width1;
  int32_t align_beam_width2;
  int32_t align_top_k;
  int32_t min_size;
  int32_t step_size;
  TopkTiling topKTilingData;
  TopkTiling topKTilingData1;
  TopkTiling topKTilingDataTail;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> log_probs_in_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> top_tokens_in_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_ids_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_index_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_log_probs_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_beam_count_prefix_sums_out_que;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_prob_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> beam_counts_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> beam_write_pos_buf;
};

class ProcessSequence{
 public:
  __aicore__ inline ProcessSequence() {}
  __aicore__ inline void
  Init(GM_ADDR sequence, GM_ADDR token_index, GM_ADDR out_sequence, GM_ADDR out_token_ids, GM_ADDR top_tokens,
     int32_t beam_width, int32_t top_k, int32_t current_step, int32_t max_decode_step, int32_t request_num,
     int32_t core_num);
  __aicore__ inline void
  Process();
  __aicore__ inline void
  SubProcessSeqPrefill(int32_t request_idx);

 private:
  int32_t beam_width;
  int32_t current_step;
  int32_t max_decode_step;
  int32_t request_num;
  int32_t core_num;
  int32_t align_beam_width;
  int32_t align_current_step;
  int32_t sequence_buf_alignsize;
  int32_t top_k;
  int32_t align_top_k;
  AscendC::GlobalTensor<int32_t> sequence_gm;
  AscendC::GlobalTensor<int32_t> out_sequence_gm;
  AscendC::GlobalTensor<int32_t> token_index_gm;
  AscendC::GlobalTensor<int32_t> out_token_ids_gm;
  AscendC::GlobalTensor<int32_t> top_tokens_gm;
  AscendC::LocalTensor<int32_t> in_sequence_local_origin;
  AscendC::LocalTensor<int32_t> gatherb_offset_local;
  AscendC::LocalTensor<int32_t> out_sequence_local;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> sequence_in_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> sequence_out_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_token_ids_que;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sequence_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> token_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_tokens_buf;
  AscendC::TBuf<AscendC::TPosition::VECOUT> out_token_ids_buf;
};
