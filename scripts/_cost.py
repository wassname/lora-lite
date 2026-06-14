"""Measure the cost of an attached adapter: params, FLOPs/MACs, time, GPU mem.

Which metric is "best" for comparing adapters? They answer different questions:

- trainable_params  -- deterministic "size" number. The headline.
- macs_per_token     -- deterministic, hardware-INDEPENDENT compute. Best for an
                        apples-to-apples comparison: wall-time is noisy and the old
                        rotation adapter paid a per-forward Cayley solve the new ones
                        do not. "adds" (additions) ~= MACs; FLOPs ~= 2 * MACs.
- fwd_ms / bwd_ms    -- felt cost, but noisy: warmup + median over `iters`, never one run.
- peak_gpu_mb        -- resident + activation peak around fwd(+bwd).

FLOPs come from torch.utils.flop_counter.FlopCounterMode (built in, no new dep). Its
convention is MACs (a (m,k)@(k,n) matmul counts as m*n*k); we expose both `flops`
(as returned) and `macs_per_token = flops / n_tokens` -- calibrate once on a known
matmul if you need to be sure of the factor of 2.
"""
from __future__ import annotations

import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def _time_call(fn, warmup: int, iters: int, cuda: bool) -> float:
    """Median wall-time of fn() in milliseconds (warmup excluded)."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def measure_cost(
    model: torch.nn.Module,
    fwd_fn,
    *,
    bwd_step_fn=None,
    n_tokens: int | None = None,
    adapter_filter: str = "lora_",
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Cost of the currently-attached adapter.

    fwd_fn():        run one forward (no grad). Used for FLOPs + fwd timing.
    bwd_step_fn():   zero_grad + forward + loss.backward(). Used for bwd timing.
    n_tokens:        tokens in the fwd_fn batch, for macs_per_token.
    adapter_filter:  substring marking adapter params/buffers (default 'lora_').
    """
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    named = list(model.named_parameters()) + list(model.named_buffers())
    adapter_bytes = sum(t.numel() * t.element_size() for n, t in named if adapter_filter in n)

    # FLOPs: one forward under the counter (no grad so we count inference cost).
    # FlopCounterMode can assert on some fused attention shapes; degrade to None.
    try:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            fwd_fn()
        flops = fc.get_total_flops()
    except Exception as e:
        print(f"  [warn] FLOP count failed ({type(e).__name__}: {e}); flops=None")
        flops = None

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    fwd_ms = _time_call(lambda: _no_grad(fwd_fn), warmup, iters, cuda)
    bwd_ms = _time_call(bwd_step_fn, warmup, iters, cuda) if bwd_step_fn is not None else None
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if cuda else None

    return dict(
        trainable_params=trainable_params,
        adapter_resident_mb=adapter_bytes / 1e6,
        flops=flops,
        macs_per_token=(flops / n_tokens) if (flops and n_tokens) else None,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        peak_gpu_mb=peak_gpu_mb,
    )


def _no_grad(fn):
    with torch.no_grad():
        return fn()


class group_init_meter:
    """Context manager: wall-time + peak CPU RAM of a group_init / attach-with-calib.

    CorDA accumulates C = E[xx^T] on CPU and runs eigh(d_in^3) -- the expensive corner.
    Use around ll.attach(model, cfg, calibration_data=...) to log that asymmetry.
    """

    def __init__(self):
        self.ms = None
        self.peak_cpu_mb = None

    def __enter__(self):
        import tracemalloc
        self._tm = tracemalloc
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1e3
        _, peak = self._tm.get_traced_memory()
        self._tm.stop()
        self.peak_cpu_mb = peak / 1e6
        return False
