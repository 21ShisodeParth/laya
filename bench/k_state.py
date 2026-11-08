"""Synthetic CUDA profiler for the encoder work implied by k-state variants.

This isolates attention and feed-forward work with controlled token counts; it
does not load Laya weights or model ModernBERT RoPE/local-window details. Use it
to compare packing mechanics and kernel eligibility, then confirm findings with
``Agent.predict_k_state`` on a real checkpoint.
"""
import argparse
import contextlib
import csv
import json
import os
import statistics

import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x, position_ids, theta=10000.0):
    dim = x.shape[-1]
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = position_ids.float()[..., None] * freq
    phase = torch.cat([phase, phase], dim=-1)[:, None]
    return x * phase.cos().to(x.dtype) + _rotate_half(x) * phase.sin().to(x.dtype)


class Block(nn.Module):
    def __init__(self, width=256, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def project(self, x, position_ids):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.dim)
        q, k, v = (value.transpose(1, 2) for value in qkv.unbind(dim=2))
        return _apply_rope(q, position_ids), _apply_rope(k, position_ids), v

    def finish(self, x, attended):
        attended = attended.transpose(1, 2).reshape_as(x)
        x = x + self.out(attended)
        return x + self.ffn(self.norm2(x))

    def forward(self, x, mask, position_ids, local=True):
        q, k, v = self.project(x, position_ids)
        mask = self.add_local_window(mask, position_ids, position_ids, local=local)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.finish(x, attended), (k, v)

    @staticmethod
    def add_local_window(mask, query_positions, key_positions, radius=64, local=True):
        if not local:
            return mask
        near = (query_positions[:, :, None] - key_positions[:, None, :]).abs() <= radius
        near = near[:, None]
        return near if mask is None else mask & near


class SyntheticEncoder(nn.Module):
    def __init__(self, layers=4, width=256, heads=4, device="cuda", dtype=torch.float16):
        super().__init__()
        self.layers = nn.ModuleList(Block(width, heads) for _ in range(layers)).to(device=device, dtype=dtype)

    def forward(self, x, mask, position_ids):
        cache = []
        for index, layer in enumerate(self.layers):
            x, kv = layer(x, mask, position_ids, local=bool(index % 2))
            cache.append(kv)
        return x, cache


@contextlib.contextmanager
def stage(name, device):
    with torch.autograd.profiler.record_function(name):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()


def _make_hidden(batch, length, width, device, dtype):
    return torch.randn(batch, length, width, device=device, dtype=dtype)


def _group_sizes(n, k):
    return [n // k + (i < n % k) for i in range(k)]


def _run_baseline(model, n, state_len, head_len, device, dtype):
    x = _make_hidden(n, head_len + state_len, 256, device, dtype)
    positions = torch.arange(head_len + state_len, device=device).expand(n, -1)
    with stage("question_encode", device):
        output, _ = model(x, None, positions)
    with stage("head", device):
        return output[:, :head_len].mean()


def _run_grouped(model, n, state_len, head_len, k, device, dtype, block_diagonal):
    sizes = _group_sizes(n, k)
    max_heads = max(sizes) * head_len
    length = max_heads + state_len
    x = _make_hidden(k, length, 256, device, dtype)
    mask = torch.ones((k, 1, length, length), dtype=torch.bool, device=device)
    if block_diagonal:
        for row, size in enumerate(sizes):
            count = size * head_len
            mask[row, 0, :count, :count] = False
            cursor = 0
            for _ in range(size):
                mask[row, 0, cursor:cursor + head_len, cursor:cursor + head_len] = True
                mask[row, 0, cursor:cursor + head_len, count:] = True
                cursor += head_len
    for row, size in enumerate(sizes):
        count = size * head_len
        if count < max_heads:
            mask[row, :, :, count:max_heads] = False
            diagonal = torch.arange(count, max_heads, device=device)
            mask[row, 0, diagonal, diagonal] = True
    positions = torch.arange(length, device=device).expand(k, -1)
    with stage("question_encode", device):
        output, _ = model(x, mask, positions)
    with stage("head", device):
        return output.mean()


def _run_mask_only(model, n, state_len, head_len, device, dtype):
    length = state_len + head_len
    x = _make_hidden(n, length, 256, device, dtype)
    mask = torch.ones((n, 1, length, length), dtype=torch.bool, device=device)
    mask[:, :, :state_len, state_len:] = False
    positions = torch.arange(length, device=device).expand(n, -1)
    with stage("question_encode", device):
        output, _ = model(x, mask, positions)
    with stage("head", device):
        return output[:, state_len:].mean()


def _run_cached(model, n, state_len, head_len, device, dtype):
    state = _make_hidden(1, state_len, 256, device, dtype)
    state_positions = torch.arange(state_len, device=device).unsqueeze(0)
    cache = []
    with stage("state_encode", device):
        for index, layer in enumerate(model.layers):
            q, key, value = layer.project(state, state_positions)
            mask = layer.add_local_window(None, state_positions, state_positions, local=bool(index % 2))
            attended = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, dropout_p=0.0)
            state = layer.finish(state, attended)
            cache.append((key, value))
    questions = _make_hidden(n, head_len, 256, device, dtype)
    query_positions = torch.arange(state_len, state_len + head_len, device=device).expand(n, -1)
    key_positions = torch.arange(state_len + head_len, device=device).expand(n, -1)
    key_valid = torch.ones((n, 1, 1, state_len + head_len), dtype=torch.bool, device=device)
    with stage("question_encode", device):
        for index, (layer, (state_key, state_value)) in enumerate(zip(model.layers, cache)):
            q, head_key, head_value = layer.project(questions, query_positions)
            key = torch.cat([state_key.expand(n, -1, -1, -1), head_key], dim=2)
            value = torch.cat([state_value.expand(n, -1, -1, -1), head_value], dim=2)
            mask = layer.add_local_window(key_valid, query_positions, key_positions, local=bool(index % 2))
            attended = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, dropout_p=0.0)
            questions = layer.finish(questions, attended)
    with stage("head", device):
        result = questions.mean()
    cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)
    return result, cache_bytes


def _call_variant(name, model, args, block_diagonal=False):
    n, state_len, head_len, k, device, dtype = args
    if name == "baseline":
        return _run_baseline(model, n, state_len, head_len, device, dtype), 0
    if name == "grouped":
        return _run_grouped(model, n, state_len, head_len, k, device, dtype, block_diagonal), 0
    if name == "mask_only_k0":
        return _run_mask_only(model, n, state_len, head_len, device, dtype), 0
    value, cache_bytes = _run_cached(model, n, state_len, head_len, device, dtype)
    return value, cache_bytes


def _time_variant(name, model, args, warmup, iterations, block_diagonal):
    for _ in range(warmup):
        _call_variant(name, model, args, block_diagonal)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations, cache_bytes = [], 0
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _, cache_bytes = _call_variant(name, model, args, block_diagonal)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end))
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "p50_ms": statistics.median(durations),
        "p95_ms": p95,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "cache_kv_bytes": cache_bytes,
    }


def _trace_variant(name, model, args, trace_dir, block_diagonal):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        _call_variant(name, model, args, block_diagonal)
        torch.cuda.synchronize()
    path = os.path.join(trace_dir, "%s.json" % name)
    prof.export_chrome_trace(path)
    ops = sorted({event.key for event in prof.key_averages()
                  if any(word in event.key.lower() for word in ("flash", "efficient_attention", "sdpa"))})
    return path, ops


def _sweep():
    configs = []
    for n in (1, 2, 5, 10, 20):
        configs.append((n, 128, 32, min(n, 4), "n"))
    for state_len in (64, 128, 192, 256, 384, 512):
        configs.append((10, state_len, 32, 4, "S"))
    for k in (1, 2, 5, 10):
        configs.append((10, 256, 32, k, "k"))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--block-diagonal", action="store_true")
    parser.add_argument("--output", default="bench/k_state_results.csv")
    parser.add_argument("--traces", default="bench/traces/k_state")
    parser.add_argument("--quick", action="store_true", help="one n=5, S=128, h=32 configuration")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.device == "cpu":
        raise SystemExit("This harness uses CUDA events and requires a CUDA device (the target is T4).")
    device, dtype = torch.device(args.device), torch.float16
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(device), "torch:", torch.__version__, "dtype: fp16")
    model = SyntheticEncoder(layers=args.layers, device=device, dtype=dtype).eval()
    configs = [(5, 128, 32, 2, "quick")] if args.quick else _sweep()
    variants = ("baseline", "grouped", "mask_only_k0", "cached_k0")
    rows, traced = [], set()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)
    for n, state_len, head_len, k, sweep in configs:
        for variant in variants:
            variant_k = min(k, n) if variant == "grouped" else (n if variant == "baseline" else 0)
            call_args = (n, state_len, head_len, variant_k, device, dtype)
            metrics = _time_variant(variant, model, call_args, args.warmup, args.iterations,
                                    args.block_diagonal)
            row = {"sweep": sweep, "variant": variant, "n": n, "S": state_len,
                   "h": head_len, "k": variant_k, "dtype": "fp16", **metrics}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
            if variant not in traced:
                path, kernels = _trace_variant(variant, model, call_args, args.traces,
                                               args.block_diagonal)
                traced.add(variant)
                print(json.dumps({"trace": path, "attention_ops": kernels}, sort_keys=True))
    with open(args.output, "w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("results:", args.output)


if __name__ == "__main__":
    main()
