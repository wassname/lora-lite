"""Per-variant attach + train + save + load round-trip, plus surgical regressions.

The big invariant is the parametrized train_save_load test: identity at t=0,
gradient flow on a real loss, then save -> reload onto a fresh model and
confirm the trained outputs survive the round-trip. Cheap on CPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import lora_lite as ll


CFG_BY_VARIANT = {
    "lora": ll.LoRAConfig,
    "pissa": ll.PiSSAConfig,
    "delora": ll.DeLoRAConfig,
    "ia3": ll.IA3Config,
    "ia3_ff": ll.IA3FFConfig,
    "dora": ll.DoRAConfig,
    "hra": ll.HRAConfig,
    "eva": ll.EVAConfig,
    "antipasto": ll.AntiPaSTOConfig,
}

# Per-variant identity tolerance at t=0 (after attach, before any step).
# fp32 SVD round-trip + per-row norm = looser tolerance for pissa/dora/antipasto.
IDENTITY_TOL = {
    "lora": 1e-6,
    "pissa": 5e-4,
    "delora": 1e-6,
    "ia3": 1e-6,
    "ia3_ff": 1e-6,
    "dora": 5e-5,
    "hra": 5e-6,
    "eva": 1e-6,
    "antipasto": 5e-4,
}


class TinyBlock(nn.Module):
    def __init__(self, d: int = 64, ff: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        m = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return x + h + m


class TinyModel(nn.Module):
    def __init__(self, n_layers: int = 4, d: int = 64, ff: int = 128, vocab: int = 100):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyBlock(d, ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = type("Cfg", (), {"hidden_size": d})()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(ids)
        for block in self.layers:
            x = block(x)
        return self.lm_head(x)


class FakeLinearLike(nn.Module):
    """linear-like, but not nn.Linear: stand-in for bnb 4/8-bit modules."""

    def __init__(self, d_in: int = 8, d_out: int = 8):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class FakeBnbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.layers = nn.ModuleList([FakeLinearLike(8, 8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


def cfg_for(variant: str) -> ll.AdapterConfig:
    extra = {"lambda0": 0.1} if variant == "delora" else {}
    return CFG_BY_VARIANT[variant](
        r=4,
        alpha=4 if variant == "pissa" else 8,
        dtype=torch.float32,
        **extra,
    )


def attach_with_calib(model: nn.Module, cfg: ll.AdapterConfig, ids: torch.Tensor) -> None:
    if cfg.variant == "eva":
        calib = [ids for _ in range(2)]
        ll.attach(model, cfg, calibration_data=calib)
    else:
        ll.attach(model, cfg)


def trainable_grad_norm(model: nn.Module) -> float:
    return sum(
        p.grad.detach().float().norm().item()
        for n, p in model.named_parameters()
        if "lora_" in n and p.grad is not None
    )


@pytest.mark.parametrize("variant", list(CFG_BY_VARIANT))
def test_train_save_load(variant: str, tmp_path: Path):
    """Identity at t=0, one SGD step, save, reload onto fresh model, outputs match."""
    torch.manual_seed(0)
    model = TinyModel()
    ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        y_base = model(ids).clone()

    cfg = cfg_for(variant)
    attach_with_calib(model, cfg, ids)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)

    with torch.no_grad():
        y_init = model(ids).clone()
    assert (y_init - y_base).abs().max().item() < IDENTITY_TOL[variant]

    target = torch.randn_like(y_init) * 0.1
    opt = torch.optim.SGD(trainable, lr=1e-2)
    opt.zero_grad()
    loss = (model(ids) - target).pow(2).mean()
    loss.backward()
    leaked = [n for n, p in model.named_parameters() if "lora_" not in n and p.grad is not None]
    assert leaked == []
    assert trainable_grad_norm(model) > 0
    opt.step()

    with torch.no_grad():
        y_trained = model(ids).clone()

    path = tmp_path / "adapter.pt"
    ll.save(model, str(path))

    torch.manual_seed(0)
    model_loaded = TinyModel()
    ll.load(model_loaded, str(path))  # EVA load skips group_init; calibration_data not needed
    with torch.no_grad():
        y_loaded = model_loaded(ids)
    assert (y_loaded - y_trained).abs().max().item() < max(IDENTITY_TOL[variant], 1e-5)


@pytest.mark.parametrize("variant", ["lora", "delora", "ia3", "hra"])
def test_hook_only_variants_attach_to_non_linear_target(variant: str):
    """bnb-style targets are linear-like but not nn.Linear; hook-only variants must accept them."""
    extra = {"lambda0": 0.1} if variant == "delora" else {}
    cfg = CFG_BY_VARIANT[variant](r=2, alpha=4, dtype=torch.float32, target_roles=(), **extra)
    model = FakeBnbModel()
    ll.attach(model, cfg)
    x = torch.randn(2, 3, 8)
    model(x).pow(2).mean().backward()
    assert trainable_grad_norm(model) > 0


@pytest.mark.parametrize("variant", ["pissa", "dora", "antipasto"])
def test_weight_reading_variants_reject_non_linear(variant: str):
    r = 4 if variant == "antipasto" else 2  # antipasto needs r % block_size==0
    cfg = CFG_BY_VARIANT[variant](r=r, alpha=r, dtype=torch.float32, target_roles=())
    with pytest.raises(TypeError, match="plain nn.Linear"):
        ll.attach(FakeBnbModel(), cfg)


def test_save_load_strict_keys(tmp_path: Path):
    torch.manual_seed(0)
    model = TinyModel()
    ll.attach(model, ll.LoRAConfig(r=4, alpha=8, dtype=torch.float32))
    p = tmp_path / "lora.pt"
    ll.save(model, str(p))
    blob = torch.load(p, weights_only=True, map_location="cpu")

    missing = {"cfg": blob["cfg"], "state": dict(blob["state"]), "base_fp": blob.get("base_fp", {})}
    missing["state"].pop(next(iter(missing["state"])))
    torch.save(missing, p)
    with pytest.raises(RuntimeError, match="missing lora keys"):
        ll.load(TinyModel(), str(p))

    bad = {"cfg": blob["cfg"], "state": dict(blob["state"]), "base_fp": blob.get("base_fp", {})}
    bad["state"]["layers.0.q_proj.lora_extra"] = torch.zeros(1)
    torch.save(bad, p)
    with pytest.raises(RuntimeError, match="unexpected lora keys"):
        ll.load(TinyModel(), str(p))


def test_no_target_layers_is_loud():
    cfg = ll.LoRAConfig(target_names=("definitely_missing",))
    with pytest.raises(RuntimeError, match="no target layers"):
        ll.attach(TinyModel(), cfg)


def test_eva_requires_calibration():
    """EVA's group_init must error loudly if calibration_data is missing."""
    with pytest.raises(ValueError, match="calibration_data"):
        ll.attach(TinyModel(), ll.EVAConfig(r=4, alpha=8, dtype=torch.float32))


def test_dora_bias_passthrough():
    """Regression: DoRA must NOT scale bias; identity holds with bias=True at t=0."""
    torch.manual_seed(0)
    d = 16
    layer = nn.Linear(d, d, bias=True)
    x = torch.randn(2, d)
    y_base = layer(x).detach()

    class Wrap(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": d})()
            self.layers = nn.ModuleList([lin])

        def forward(self, x):
            return self.layers[0](x)

    model = Wrap(layer)
    ll.attach(model, ll.DoRAConfig(r=2, alpha=4, dtype=torch.float32, target_roles=()))
    with torch.no_grad():
        y = model(x)
    assert (y - y_base).abs().max().item() < 1e-5


def test_hra_forward_is_x_R_T():
    """HRA must apply x @ R^T (loop i = r-1 down to 0). Asymmetric U makes order observable."""
    torch.manual_seed(0)
    d = 8
    layer = nn.Linear(d, d, bias=False)
    x = torch.randn(2, 3, d)

    class Wrap(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": d})()
            self.layers = nn.ModuleList([lin])

        def forward(self, x):
            return self.layers[0](x)

    model = Wrap(layer)
    ll.attach(model, ll.HRAConfig(r=4, alpha=4, dtype=torch.float32, target_roles=()))
    # break paired symmetry so order matters
    with torch.no_grad():
        layer.lora_U.add_(0.1 * torch.randn_like(layer.lora_U))

    U = layer.lora_U
    R = torch.eye(d)
    for i in range(U.shape[0]):
        u = U[i]
        sq = (u * u).sum().clamp_min(1e-12)
        R = R - (2.0 / sq) * torch.outer(R @ u, u)
    with torch.no_grad():
        y_adapt = model(x)
        y_ref = torch.nn.functional.linear(x, layer.weight @ R)
    assert (y_adapt - y_ref).abs().max().item() < 1e-5
