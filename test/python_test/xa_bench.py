import argparse
import ctypes as ct
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/export/home/ext.wangguangcai1/code2/xllm-ops/test/python_test' if False else __file__.rsplit('/', 1)[0])
import x_attention_test as xt

SIZES = {
    'tiny':      dict(beam_width=256, shared_length=3,    decode_steps=1,  max_decode_steps=2),
    'dec_s':     dict(beam_width=256, shared_length=512,  decode_steps=1,  max_decode_steps=1),
    'dec_m':     dict(beam_width=256, shared_length=4096, decode_steps=1,  max_decode_steps=1),
    'dec_l':     dict(beam_width=256, shared_length=4096, decode_steps=32, max_decode_steps=32),
    'beam1024':  dict(beam_width=1024, shared_length=4096, decode_steps=1,  max_decode_steps=1),
}


class A:
    pass


def build(args, name, cfg):
    a = A()
    a.batch = 1
    a.beam_width = cfg['beam_width']
    a.q_heads = 16
    a.kv_heads = 4
    a.head_dim = 128
    a.shared_length = cfg['shared_length']
    a.max_decode_steps = cfg['max_decode_steps']
    a.decode_steps = cfg['decode_steps']
    a.dtype = 'bf16'
    a.kv_format = 'ncdhw'
    a.pattern = 'random'
    a.amplitude = 1.0
    a.seed = 20261026
    a.scale = 0.08838834764831845
    a.device = args.device
    a.op_api_lib = args.op_api_lib
    a.op_api_header = None
    return a


def run_case(args, name, cfg, library, libs, sync):
    a = build(args, name, cfg)
    arrays = xt.make_inputs(a)
    import torch
    import torch_npu
    ptr, dims = ct.c_void_p, ct.POINTER(ct.c_int64)
    create = xt.bind(libs, 'aclCreateTensor', [dims, ct.c_uint64, ct.c_int, dims,
                     ct.c_int64, ct.c_int, dims, ct.c_uint64, ptr], ptr)
    tensors = []
    for slot, raw in enumerate(arrays):
        host = torch.from_numpy(raw.copy())
        if slot < 5 and a.dtype == 'bf16':
            host = host.view(torch.bfloat16)
        tensor = host.to(f'npu:{a.device}')
        fmt = 30 if slot in (3, 4) and a.kv_format == 'ncdhw' else 2
        tensors.append(torch_npu.npu_format_cast(tensor, fmt).contiguous())
    out = torch_npu.npu_format_cast(torch.empty(
        (a.batch * a.beam_width, a.q_heads, a.head_dim), dtype=tensors[0].dtype,
        device=f'npu:{a.device}'), 2)
    tensors.append(out)
    handles = []
    for slot, tensor in enumerate(tensors):
        shape = (ct.c_int64 * tensor.dim())(*tensor.shape)
        stride = (ct.c_int64 * tensor.dim())(*tensor.stride())
        dtype = 3 if slot in (5, 6, 7) else 27
        fmt = 30 if slot in (3, 4) and a.kv_format == 'ncdhw' else 2
        handles.append(create(shape, tensor.dim(), dtype, stride, 0, fmt, shape, tensor.dim(), tensor.data_ptr()))
    workspace_fn = xt.bind(libs, 'aclnnXAttentionGetWorkspaceSize',
                           [ptr] * 9 + [ct.c_double, ptr, ct.POINTER(ct.c_uint64), ct.POINTER(ptr)])
    execute = xt.bind(libs, 'aclnnXAttention', [ptr, ct.c_uint64, ptr, ptr])
    malloc = xt.bind(libs, 'aclrtMalloc', [ct.POINTER(ptr), ct.c_uint64, ct.c_int])
    free = xt.bind(libs, 'aclrtFree', [ptr])
    ws = ct.c_uint64(0)
    executor = ct.c_void_p()
    xt.check(workspace_fn(*handles[:8], None, a.scale, handles[-1], ct.byref(ws), ct.byref(executor)), 'GetWorkspaceSize')
    wsp = ct.c_void_p()
    xt.check(malloc(ct.byref(wsp), ws, 0), 'malloc')
    stream = torch.npu.current_stream().npu_stream

    def once():
        size, executor = ct.c_uint64(), ptr()
        xt.check(workspace_fn(*handles[:8], None, a.scale, handles[-1], ct.byref(size), ct.byref(executor)), 'GetWorkspaceSize')
        xt.check(execute(wsp, size, executor, stream), 'execute')

    q_bytes = tensors[0].numel() * 2
    kv_bytes = tensors[3].numel() * 2
    for _ in range(args.warmup):
        once()
    sync(stream)
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        once()
        sync(stream)
        lat.append(time.perf_counter() - t0)
    thr = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        for _ in range(10):
            once()
        sync(stream)
        thr.append((time.perf_counter() - t0) / 10)
    for h in handles:
        pass
    row = {
        'case': name,
        'beam': a.beam_width, 'shared': a.shared_length,
        'decode': a.decode_steps, 'max_decode': a.max_decode_steps,
        'workspace_bytes': ws.value,
        'q_mb': q_bytes / 1e6, 'unshared_kv_mb': kv_bytes / 1e6,
        'latency_ms': sorted(lat)[len(lat) // 2] * 1e3,
        'latency_min_ms': min(lat) * 1e3,
        'throughput_ms': sorted(thr)[len(thr) // 2] * 1e3,
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--op-api-lib')
    p.add_argument('--cases', default='tiny,dec_s,dec_m,dec_l,beam1024')
    args = p.parse_args()

    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    library = xt.op_library(args.op_api_lib)
    libs = [ct.CDLL(str(library), mode=ct.RTLD_GLOBAL),
            ct.CDLL('libnnopbase.so', mode=ct.RTLD_GLOBAL),
            ct.CDLL('libascendcl.so', mode=ct.RTLD_GLOBAL)]
    sync = xt.bind(libs, 'aclrtSynchronizeStream', [ct.c_void_p])
    rows = []
    for name in args.cases.split(','):
        rows.append(run_case(args, name, SIZES[name], library, libs, sync))
    print('== summary ==')
    for r in rows:
        print(f"{r['case']:9s} beam={r['beam']:5d} shared={r['shared']:5d} dec={r['decode']:3d}"
              f" ws={r['workspace_bytes']/1e6:8.2f}MB lat={r['latency_ms']:9.4f}ms"
              f" min={r['latency_min_ms']:9.4f}ms thr={r['throughput_ms']:9.4f}ms")


if __name__ == '__main__':
    main()
