import math
import random
import numpy as np
import pytest
import pandas as pd
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

torch.manual_seed(1)
random.seed(1)

device_id = 7
torch_npu.npu.set_device(device_id)


def test_x_attention_with_cpu(
    dtype,
    q_head_num,
    kv_head_num,
    head_dim,
    batch,
    beam_size,
    max_decode_step,
    prompt_length,
    unshared_kv_len
):
    shared_block_tokens = batch * prompt_length
    unshared_block_num = batch + 209
    total_tokens = batch * beam_size

    query = torch.empty(total_tokens, q_head_num, head_dim, dtype=dtype).uniform_(-1, 1)
    shared_key_block = torch.empty(
        shared_block_tokens, kv_head_num, head_dim, dtype=dtype
    ).uniform_(-1, 1)
    shared_value_block = torch.empty_like(shared_key_block).uniform_(-1, 1)
    shared_kv_lens = torch.full((batch,), prompt_length, dtype=torch.int32)

    unshared_block_tables = torch.randperm(unshared_block_num)[:batch].to(torch.int32)
    unshared_key_block = torch.zeros(
        unshared_block_num,
        beam_size,
        kv_head_num,
        max_decode_step,
        head_dim,
        dtype=dtype,
    )
    unshared_value_block = torch.zeros_like(unshared_key_block)
    for block_idx in unshared_block_tables.tolist():
        unshared_key_block[block_idx, :, :, :unshared_kv_len, :].uniform_(-500, 500)
        unshared_value_block[block_idx, :, :, :unshared_kv_len, :].uniform_(-3, 3)

    decode_step = torch.tensor([unshared_kv_len], dtype=torch.int32)

    npu_out = custom_ops.x_attention_npu(
        query.npu(),
        shared_key_block.npu(),
        shared_value_block.npu(),
        unshared_key_block.npu(),
        unshared_value_block.npu(),
        shared_kv_lens.npu(),
        decode_step.npu(),
        None,
        unshared_block_tables.npu(),
    )
    torch.npu.synchronize()

    npu_out = npu_out.cpu().float()


if __name__ == '__main__':
    head_dim = 128
    q_head_num = 16
    kv_head_num = 8
    dtype = torch.bfloat16
    data_dict = []
    for batch in [1, 6]:
        for beam_width in [128, 256, 512]:
            for max_decode_step in [2, 4]:
                for prompt_length in [128, 256, 512, 1024]:
                    unshared_kv_len = random.randint(1, max_decode_step)
                    data_dict.append({
                        "batch": batch,
                        "q_head_num": q_head_num,
                        "kv_head_num": kv_head_num,
                        "beam_width": beam_width,
                        "max_decode_step": max_decode_step,
                        "unshared_kv_len": unshared_kv_len,
                        "prompt_len": prompt_length
                    })
                    test_x_attention_with_cpu(dtype, q_head_num, kv_head_num, head_dim, batch, beam_width, max_decode_step, prompt_length, unshared_kv_len)
    data = pd.DataFrame(data_dict)
    data.to_csv("x_attention_perf_a5.csv", index=False)
