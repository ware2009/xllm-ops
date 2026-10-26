#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Generate OneRec XAttention inputs and compare ACLNN NPU output with CPU FP32.

Standalone: no model, dump, msProbe, custom_ops extension or sibling script.
CPU mode needs NumPy; run mode additionally needs torch, torch_npu and CANN.
See test_xattention_cpu_npu.md for commands and the input contract.
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


NAMES = ('query', 'shared_key', 'shared_value', 'unshared_key',
         'unshared_value', 'block_table', 'shared_lengths', 'decode_step')


def quantize(values, dtype):
    values = np.asarray(values, dtype='<f4')
    if dtype == 'float16':
        return values.astype('<f2')
    # Round-to-nearest-even BF16. Generated inputs are finite.
    bits = values.view('<u4')
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')


def fp32(raw, dtype):
    if dtype == 'bf16':
        return (raw.astype('<u4') << 16).view('<f4')
    return raw.astype(np.float32)


def make_inputs(args):
    b, w, h, kh, d = args.batch, args.beam_width, args.q_heads, args.kv_heads, args.head_dim
    shapes = [(b * w, h, d), (b * args.shared_length, kh, d),
              (b * args.shared_length, kh, d),
              (b, w, kh, args.max_decode_steps, d),
              (b, w, kh, args.max_decode_steps, d)]
    rng = np.random.Generator(np.random.PCG64(args.seed))
    arrays = []
    for slot, shape in enumerate(shapes):
        # Integer grid makes seed-generated payloads reproducible across machines.
        values = rng.integers(-1024, 1025, size=shape, dtype=np.int32).astype(np.float32) / 1024
        values *= args.amplitude
        if args.pattern == 'zero-query' and slot == 0:
            values.fill(0)
        if args.pattern == 'constant-value' and slot in (2, 4):
            values.fill(.5)
        arrays.append(quantize(values, args.dtype))
    # Reverse mapping exercises a nonzero block index when batch > 1.
    arrays.extend([np.arange(b - 1, -1, -1, dtype=np.int32),
                   np.full(b, args.shared_length, dtype=np.int32),
                   np.array([args.decode_steps], dtype=np.int32)])
    return arrays


def cpu_reference(arrays, dtype, scale):
    q, sk, sv, uk, uv, table, lengths, step = arrays
    beam, kv_heads = uk.shape[1:3]
    heads, dim = q.shape[1:]
    group = heads // kv_heads
    result = np.empty(q.shape, dtype=np.float32)
    offset = 0
    for request, length in enumerate(lengths):
        length = int(length)
        shared_k = fp32(sk[offset:offset + length], dtype).transpose(1, 0, 2)
        shared_v = fp32(sv[offset:offset + length], dtype).transpose(1, 0, 2)
        block = int(table[request])
        for row in range(beam):
            keys = np.concatenate([shared_k, fp32(uk[block, row, :, :int(step[0])], dtype)], axis=1)
            values = np.concatenate([shared_v, fp32(uv[block, row, :, :int(step[0])], dtype)], axis=1)
            query = fp32(q[request * beam + row], dtype).reshape(kv_heads, group, dim)
            scores = np.einsum('hgd,hld->hgl', query, keys) * np.float32(scale)
            scores -= scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            result[request * beam + row] = np.einsum('hgl,hld->hgd', weights, values).reshape(heads, dim)
        offset += length
    if not np.isfinite(result).all():
        raise ValueError('CPU reference contains NaN/Inf; reduce --amplitude')
    return result


def metrics(reference, actual, atol, rtol):
    x, y = reference.astype(np.float64), actual.astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    difference = np.full(x.shape, np.inf)
    np.subtract(x, y, out=difference, where=finite)
    difference = np.abs(difference)
    mismatch = ~finite | (difference > atol + rtol * np.abs(x))
    xf, yf = x[finite], y[finite]
    delta = difference[finite]
    norm_x, norm_y = np.linalg.norm(xf), np.linalg.norm(yf)
    cosine = float(np.clip(np.dot(xf, yf) / norm_x / norm_y, -1, 1)) if norm_x and norm_y else None
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(difference), x.shape))
    def number(value):
        return float(value) if np.isfinite(value) else None
    return {'status': 'FAIL' if mismatch.any() else 'PASS',
            'max_abs': float(delta.max()) if delta.size else None,
            'mean_abs': float(delta.mean()) if delta.size else None,
            'relative_l2': float(np.linalg.norm(xf - yf) / norm_x) if norm_x else None,
            'cosine_sim': cosine, 'mismatch_ratio': float(mismatch.mean()),
            'nonfinite_count': int((~finite).sum()),
            'worst_index': list(worst), 'worst_reference': number(x[worst]),
            'worst_actual': number(y[worst])}


def op_library(explicit):
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    roots = list(filter(None, os.getenv('ASCEND_CUSTOM_OPP_PATH', '').split(':')))
    if os.getenv('ASCEND_OPP_PATH'):
        roots.append(str(Path(os.environ['ASCEND_OPP_PATH']) / 'vendors/custom_xllm_math'))
    candidates = {p.resolve() for root in roots for p in
                  (Path(root) / 'op_api/lib').glob('libcust_opapi*.so')}
    if len(candidates) != 1:
        raise ValueError(f'Pass --op-api-lib: expected one vendor library, found {sorted(map(str, candidates))}')
    return candidates.pop()


def verify_header(library, explicit):
    include = library.parent.parent / 'include'
    paths = [Path(explicit).expanduser()] if explicit else [
        include / 'aclnnop/aclnn_x_attention.h', include / 'aclnn_x_attention.h']
    for path in paths:
        if not path.is_file():
            continue
        source = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
        match = re.search(r'\baclnnXAttentionGetWorkspaceSize\s*\((.*?)\)\s*;', source, re.S)
        parameters = match[1].split(',') if match else []
        patterns = ([r'const\s+aclTensor\s*\*\s*\w+'] * 9 +
                    [r'double\s+\w+', r'const\s+aclTensor\s*\*\s*\w+',
                     r'uint64_t\s*\*\s*\w+', r'aclOpExecutor\s*\*\s*\*\s*\w+'])
        if len(parameters) != len(patterns) or any(
                not re.fullmatch(p, arg.strip()) for p, arg in zip(patterns, parameters)):
            raise ValueError(f'Installed XAttention ABI differs from this script: {path}')
        print(f'op_api_header={path.resolve()} (ABI verified)', flush=True)
        return str(path.resolve())
    raise ValueError('Cannot verify installed XAttention ABI; pass --op-api-header')


def bind(libraries, name, arguments, restype=ct.c_int):
    for library in libraries:
        function = getattr(library, name, None)
        if function is not None:
            function.argtypes, function.restype = arguments, restype
            return function
    raise RuntimeError(f'Cannot resolve {name}')


def check(status, stage):
    if status:
        raise RuntimeError(f'{stage}: ACL status={status}; stop and inspect plog')


def write_report(output, report):
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def run_npu(args, arrays, reference, report, output):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    library = op_library(args.op_api_lib)
    header = verify_header(library, args.op_api_header)
    report['runtime'] = {'pid': os.getpid(), 'torch': torch.__version__,
                         'torch_npu': torch_npu.__version__,
                         'device': torch.npu.get_device_name(args.device),
                         'op_api': str(library), 'op_api_header': header}
    print(json.dumps(report['runtime']), flush=True)
    libraries = [ct.CDLL(str(library), mode=ct.RTLD_GLOBAL),
                 ct.CDLL('libnnopbase.so', mode=ct.RTLD_GLOBAL),
                 ct.CDLL('libascendcl.so', mode=ct.RTLD_GLOBAL)]
    ptr, dims = ct.c_void_p, ct.POINTER(ct.c_int64)
    create = bind(libraries, 'aclCreateTensor', [dims, ct.c_uint64, ct.c_int, dims,
                  ct.c_int64, ct.c_int, dims, ct.c_uint64, ptr], ptr)
    destroy = bind(libraries, 'aclDestroyTensor', [ptr])
    workspace_fn = bind(libraries, 'aclnnXAttentionGetWorkspaceSize',
                        [ptr] * 9 + [ct.c_double, ptr, ct.POINTER(ct.c_uint64), ct.POINTER(ptr)])
    execute = bind(libraries, 'aclnnXAttention', [ptr, ct.c_uint64, ptr, ptr])
    malloc = bind(libraries, 'aclrtMalloc', [ct.POINTER(ptr), ct.c_uint64, ct.c_int])
    free = bind(libraries, 'aclrtFree', [ptr])
    sync = bind(libraries, 'aclrtSynchronizeStream', [ptr])
    tensors = []
    for slot, raw in enumerate(arrays):
        host = torch.from_numpy(raw.copy())
        if slot < 5 and args.dtype == 'bf16':
            host = host.view(torch.bfloat16)
        tensor = host.to(f'npu:{args.device}')
        # Explicit base formats avoid implicit padded/blocked torch_npu storage.
        fmt = 30 if slot in (3, 4) and args.kv_format == 'ncdhw' else 2
        tensors.append(torch_npu.npu_format_cast(tensor, fmt).contiguous())
    tensors.append(torch_npu.npu_format_cast(torch.empty(
        reference.shape, dtype=tensors[0].dtype, device=f'npu:{args.device}'), 2))
    handles = []
    for slot, tensor in enumerate(tensors):
        shape = (ct.c_int64 * tensor.dim())(*tensor.shape)
        stride = (ct.c_int64 * tensor.dim())(*tensor.stride())
        dtype = 3 if slot in (5, 6, 7) else (27 if args.dtype == 'bf16' else 1)
        fmt = 30 if slot in (3, 4) and args.kv_format == 'ncdhw' else 2
        handle = create(shape, tensor.dim(), dtype, stride, 0, fmt,
                        shape, tensor.dim(), tensor.data_ptr())
        if not handle:
            raise RuntimeError(f'aclCreateTensor returned null: slot={slot}')
        handles.append(handle)
    first = None
    for iteration in range(1, args.repeat + 1):
        tensors[-1].fill_(float('nan'))
        torch.npu.synchronize()
        stream = torch.npu.current_stream().npu_stream
        size, executor, workspace = ct.c_uint64(), ptr(), ptr()
        check(workspace_fn(*handles[:8], None, args.scale, handles[-1],
                           ct.byref(size), ct.byref(executor)), 'GetWorkspaceSize')
        if not executor.value:
            raise RuntimeError('GetWorkspaceSize returned null executor')
        print(f'BEGIN iteration={iteration} workspace_bytes={size.value}', flush=True)
        if size.value:
            check(malloc(ct.byref(workspace), size.value, 0), 'aclrtMalloc')
        # After an ACL failure exit the process; do not reuse a failed device context.
        check(execute(workspace, size.value, executor, stream), 'aclnnXAttention')
        check(sync(stream), 'aclrtSynchronizeStream')
        actual = tensors[-1].float().cpu().numpy()
        if workspace.value:
            check(free(workspace), 'aclrtFree')
        result = {'iteration': iteration, 'workspace_bytes': size.value,
                  'cpu_vs_npu': metrics(reference, actual, args.atol, args.rtol),
                  'per_head_max_abs': [metrics(reference[:, head], actual[:, head],
                                              args.atol, args.rtol)['max_abs']
                                       for head in range(args.q_heads)]}
        if first is None:
            first = actual.copy()
        else:
            result['first_vs_repeat'] = metrics(first, actual, 0, 0)
        np.save(output / f'npu_output_{iteration}.npy', actual)
        report['iterations'].append(result)
        write_report(output, report)
        print(json.dumps(result, allow_nan=False), flush=True)
    for handle in handles:
        check(destroy(handle), 'aclDestroyTensor')
    return any(row['cpu_vs_npu']['status'] == 'FAIL' or
               row.get('first_vs_repeat', {}).get('status') == 'FAIL'
               for row in report['iterations'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'cpu'), default='run')
    parser.add_argument('--dtype', choices=('bf16', 'float16'), default='bf16')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--beam-width', type=int, default=256)
    parser.add_argument('--q-heads', type=int, default=16)
    parser.add_argument('--kv-heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, choices=(128,), default=128)
    parser.add_argument('--shared-length', type=int, default=3)
    parser.add_argument('--max-decode-steps', type=int, default=2)
    parser.add_argument('--decode-steps', type=int, default=1, help='Effective KV count, no extra +1')
    parser.add_argument('--scale', type=float, default=0.08838834764831845)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--amplitude', type=float, default=1.)
    parser.add_argument('--pattern', choices=('random', 'zero-query', 'constant-value'), default='random')
    parser.add_argument('--kv-format', choices=('ncdhw', 'nd'), default='ncdhw')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--op-api-lib')
    parser.add_argument('--op-api-header')
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-2)
    parser.add_argument('--output', default='xattention_synthetic')
    args = parser.parse_args(argv)
    positive = ('batch', 'beam_width', 'q_heads', 'kv_heads', 'shared_length',
                'max_decode_steps', 'decode_steps', 'repeat')
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error('Dimensions, KV lengths and repeat must be positive')
    if args.q_heads % args.kv_heads or args.q_heads // args.kv_heads > 128:
        parser.error('q-heads must be divisible by kv-heads; group size must be <= 128')
    if not args.decode_steps <= args.max_decode_steps <= 256:
        parser.error('Require 1 <= decode-steps <= max-decode-steps <= 256')
    if args.device < 0 or args.seed < 0:
        parser.error('device and seed must be nonnegative')
    if any(not math.isfinite(getattr(args, name)) for name in ('scale', 'amplitude', 'atol', 'rtol')):
        parser.error('scale, amplitude and tolerances must be finite')
    if args.scale <= 0 or not 0 < args.amplitude <= 100 or min(args.atol, args.rtol) < 0:
        parser.error('Require scale > 0, 0 < amplitude <= 100 and nonnegative tolerances')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    report = None
    try:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output must be absent or an empty directory')
        arrays = make_inputs(args)
        report = {'config': vars(args), 'numpy': np.__version__, 'inputs': [], 'iterations': [],
                  'environment': {name: os.getenv(name, '<unset>') for name in (
                      'LD_PRELOAD', 'ASCEND_CUSTOM_OPP_PATH', 'ASCEND_RT_VISIBLE_DEVICES',
                      'ASCEND_LAUNCH_BLOCKING', 'TASK_QUEUE_ENABLE', 'PER_STREAM_QUEUE')}}
        for slot, (name, raw) in enumerate(zip(NAMES, arrays)):
            item = {'slot': slot, 'name': name, 'shape': list(raw.shape),
                    'dtype': args.dtype if slot < 5 else 'int32',
                    'format': args.kv_format if slot in (3, 4) else 'nd',
                    'sha256': hashlib.sha256(raw.tobytes()).hexdigest()}
            if slot >= 5:
                item['values'] = raw.tolist()
            report['inputs'].append(item)
            print(json.dumps(item), flush=True)
        print(f'scale={args.scale} pattern={args.pattern} mode={args.mode}', flush=True)
        reference = cpu_reference(arrays, args.dtype, args.scale)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / 'cpu_reference.npy', reference)
        write_report(output, report)
        failed = run_npu(args, arrays, reference, report, output) if args.mode == 'run' else False
        print(f'Report: {output / "report.json"}', flush=True)
        return int(failed)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if report is not None and output.is_dir():
            report['error'] = str(exc)
            write_report(output, report)
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())