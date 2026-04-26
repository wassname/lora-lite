"""Smoke test: current variants on a tiny synthetic transformer-like model.

Verifies:
    1. Identity at t=0 (delta ~ 0, output close to base).
    2. Save/load round-trip preserves outputs.
    3. A few SGD steps reduce a random loss (gradients flow).

Run:
    cd lora-lite
    python -m pip install -e .
    python tests/smoke.py

BLUF format:
    SHOULD: max|y_adapter - y_base| < tol_init for all variants. ELSE init or hook bug.
    SHOULD: loss decreases > 5% over 20 SGD steps for all variants. ELSE grad/wiring bug.
"""
from __future__ import annotations
import argparse
import os, sys, math
from pathlib import Path
import torch
from torch import nn

# allow running as `python tests/smoke.py` without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lora_lite as ll  # noqa: E402


ARTIFACT_DIR = Path(__file__).parent / "_artifacts"


def assert_no_base_grads(model: nn.Module) -> None:
    leaked = [name for name, p in model.named_parameters() if "lora_" not in name and p.grad is not None]
    assert leaked == [], f"base params received grads: {leaked}"


# ---- a tiny transformer-like stack: 4 blocks of (q,k,v,o, gate,up,down) Linears ----
class TinyBlock(nn.Module):
    def __init__(self, d=64, ff=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        h = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        m = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return x + h + m


class TinyModel(nn.Module):
    def __init__(self, n_layers=4, d=64, ff=128, vocab=100):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyBlock(d, ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

        class Cfg:  # mimic HF .config.hidden_size
            hidden_size = d
        self.config = Cfg()

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for blk in self.layers:
            x = blk(x)
        return self.lm_head(x)


class FakeLinearLike(nn.Module):
    """Not nn.Linear, but structurally bnb-like enough for target discovery."""

    def __init__(self, d_in=8, d_out=8):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class FakeBnbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.layers = nn.ModuleList([FakeLinearLike(8, 8)])

    def forward(self, x):
        return self.layers[0](x)


def variant_test(variant: str, dtype=torch.float32):
    print(f"\n=== variant={variant} dtype={dtype} ===")
    torch.manual_seed(0)
    model = TinyModel().to(dtype)
    ids = torch.randint(0, 100, (2, 16))

    with torch.no_grad():
        y_base = model(ids).clone()

    cfg = ll.LoraLiteConfig(
        variant=variant,
        r=4,
        alpha=4 if variant == "pissa" else 8,  # PiSSA needs scale==1 for clean recon
        dtype=dtype,
        # delora identity-at-init demands lambda0==0 (then delta * scale = 0)
        variant_kwargs={"lambda0": 0.0} if variant == "delora" else {},
    )
    handles = ll.attach(model, cfg)
    n_targets = len(handles)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  attached {n_targets} targets, trainable params={n_trainable}")
    assert n_targets == 28, f"expected 28 TinyModel targets, got {n_targets}"

    with torch.no_grad():
        y_adapt = model(ids)
    err = (y_adapt - y_base).abs().max().item()
    base_scale = y_base.abs().max().item()
    print(f"  t=0 identity: max|y_adapt - y_base| = {err:.3e}  (base scale {base_scale:.3e})")

    # variant-specific identity tolerance
    tol = {
        "lora": 1e-6,
        "pissa": 5e-4,    # SVD recon in fp32 is tight; bf16 would be ~1e-2
        "delora": 1e-6,   # lambda0=0
        "ia3": 1e-6,
        "dora": 5e-5,     # m * V/||V|| with V=W -> rounding in norm/divide
        "hra": 1e-6,      # gate=0 -> exact identity
    }[variant] * max(1.0, base_scale)
    assert err < tol, f"  FAIL identity: err {err} > tol {tol}"
    print(f"  SHOULD: err<{tol:.1e}. PASS.")

    # save/load round-trip
    ARTIFACT_DIR.mkdir(exist_ok=True)
    p = ARTIFACT_DIR / f"{variant}_smoke_adapter.pt"
    ll.save(model, str(p))
    # detach + fresh model + load
    ll.detach(model)
    torch.manual_seed(0)
    model2 = TinyModel().to(dtype)
    # for PiSSA, base weights got mutated; load() re-runs PiSSA init on the fresh
    # same-seed base, then overwrites lora_A/B with saved values.
    ll.load(model2, str(p))
    with torch.no_grad():
        y_loaded = model2(ids)
    err2 = (y_loaded - y_adapt).abs().max().item()
    print(f"  save/load: max|y_loaded - y_adapt| = {err2:.3e}")
    assert err2 < tol, f"  FAIL save/load: {err2} > {tol}"
    print(f"  SHOULD: err2<{tol:.1e}. PASS.")
    ll.detach(model2)

    # gradient flow: 20 SGD steps on random target.
    # For delora, lambda0==0 makes A,B grads zero (scale=0); use lambda0=0.1 for training.
    torch.manual_seed(0)
    model = TinyModel().to(dtype)
    train_cfg = cfg
    if variant == "delora":
        train_cfg = ll.LoraLiteConfig(
            variant=cfg.variant, r=cfg.r, alpha=cfg.alpha, dtype=cfg.dtype,
            variant_kwargs={"lambda0": 0.1},
        )
    ll.attach(model, train_cfg)
    target = torch.randn(2, 16, 100, dtype=dtype) * 0.1
    trainable = [p for p in model.parameters() if p.requires_grad]
    # delora has tightly-normalised updates; use Adam with higher lr to see signal in 20 steps
    if variant in ("delora", "ia3", "hra"):
        opt = torch.optim.Adam(trainable, lr=1e-1)
    elif variant == "dora":
        opt = torch.optim.Adam(trainable, lr=1e-3)  # m near ||W||_c, bigger lr blows up
    else:
        opt = torch.optim.SGD(trainable, lr=1e-2)
    losses = []
    for step in range(20):
        opt.zero_grad()
        loss = (model(ids) - target).pow(2).mean()
        loss.backward()
        assert_no_base_grads(model)
        opt.step()
        losses.append(loss.item())
    drop = (losses[0] - losses[-1]) / max(losses[0], 1e-12)
    print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  drop={100*drop:.1f}%")
    assert drop > 0.05, f"  FAIL: loss drop only {drop:.2%}, expected >5%"
    print(f"  SHOULD: drop>5%. PASS.")


def structural_linear_like_test():
    print("\n=== structural linear-like target test (bnb-style, not nn.Linear) ===")
    torch.manual_seed(0)
    model = FakeBnbModel()
    x = torch.randn(2, 3, 8)
    y_base = model(x).detach()
    ll.attach(model, ll.LoraLiteConfig(variant="lora", r=2, alpha=4, dtype=torch.float32, target_roles=()))
    layer = model.layers[0]
    assert hasattr(layer, "lora_A") and hasattr(layer, "lora_B")
    y = model(x)
    err = (y.detach() - y_base).abs().max().item()
    loss = y.pow(2).mean()
    loss.backward()
    grad_nonzero = layer.lora_B.grad.abs().sum().item() > 0
    print(f"  attached lora_A={tuple(layer.lora_A.shape)} lora_B={tuple(layer.lora_B.shape)}")
    print(f"  identity_err={err:.3e}  grad_nonzero={grad_nonzero}")
    assert err == 0.0
    assert grad_nonzero
    print("  SHOULD: structural target attaches and lora_B receives grad. PASS.")


def bitsandbytes_cuda_smoke(require_bnb: bool):
    label = "required" if require_bnb else "optional"
    print(f"\n=== {label} bitsandbytes CUDA smoke ===")
    if not torch.cuda.is_available():
        if require_bnb:
            raise RuntimeError("CUDA unavailable; required real bnb 4/8-bit smoke cannot run.")
        print("  SKIP: CUDA unavailable; real bnb 4/8-bit forward needs GPU on this machine.")
        return
    try:
        import bitsandbytes as bnb
    except ImportError:
        if require_bnb:
            raise RuntimeError("bitsandbytes unavailable; install the bnb-test extra.")
        print("  SKIP: bitsandbytes unavailable.")
        return

    class BnbModel(nn.Module):
        def __init__(self, Layer):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 8})()
            self.layers = nn.ModuleList([Layer(8, 8, bias=False)]).cuda()

        def forward(self, x):
            return self.layers[0](x)

    for layer_cls in (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit):
        torch.manual_seed(0)
        model = BnbModel(layer_cls)
        x = torch.randn(2, 3, 8, device="cuda")
        y_base = model(x).detach()
        ll.attach(model, ll.LoraLiteConfig(variant="lora", r=2, alpha=4, dtype=torch.float16, target_roles=()))
        y = model(x)
        err = (y.detach() - y_base).abs().max().item()
        y.pow(2).mean().backward()
        grad_nonzero = model.layers[0].lora_B.grad.abs().sum().item() > 0
        print(f"  {layer_cls.__name__}: identity_err={err:.3e} grad_nonzero={grad_nonzero}")
        assert err == 0.0
        assert grad_nonzero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-bnb", action="store_true")
    args = parser.parse_args()

    for v in ("lora", "pissa", "delora", "ia3", "dora", "hra"):
        variant_test(v, dtype=torch.float32)
    structural_linear_like_test()
    bitsandbytes_cuda_smoke(args.require_bnb)
    print("\nALL PASS.")


if __name__ == "__main__":
    main()
