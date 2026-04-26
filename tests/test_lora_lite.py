from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

import lora_lite as ll


ARTIFACT_DIR = Path(__file__).parent / "_artifacts"


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
    def __init__(self, d_in: int = 8, d_out: int = 8):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class FakeBnbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.layers = nn.ModuleList([FakeLinearLike(8, 8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


def cfg_for_variant(variant: str, *, training: bool = False) -> ll.LoraLiteConfig:
    return ll.LoraLiteConfig(
        variant=variant,
        r=4,
        alpha=4 if variant == "pissa" else 8,
        dtype=torch.float32,
        variant_kwargs={"lambda0": 0.1 if training else 0.0} if variant == "delora" else {},
    )


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def assert_only_lora_trainable(model: nn.Module) -> None:
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


def assert_no_base_grads(model: nn.Module) -> None:
    leaked = [name for name, p in model.named_parameters() if "lora_" not in name and p.grad is not None]
    assert leaked == []


def perturb_first_adapter(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "lora_lambda" in name:
            with torch.no_grad():
                p.add_(0.25)
            return
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.flatten()[0].add_(0.25)
            return
    for name, p in model.named_parameters():
        if "lora_g" in name:
            with torch.no_grad():
                p.flatten()[0].add_(0.25)
            return
    raise AssertionError("no perturbable adapter parameter found")


@pytest.mark.parametrize("variant", ["lora", "pissa", "delora", "ia3", "dora"])
def test_variant_identity_hook_save_load_and_training(variant: str):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model = TinyModel()
    ids = torch.randint(0, 100, (2, 16))

    with torch.no_grad():
        y_base = model(ids).clone()

    cfg = cfg_for_variant(variant)
    handles = ll.attach(model, cfg)
    assert len(handles) == 28
    assert_only_lora_trainable(model)

    with torch.no_grad():
        y_init = model(ids).clone()
    identity_err = (y_init - y_base).abs().max().item()
    identity_tol = {"lora": 1e-6, "pissa": 5e-4, "delora": 1e-6, "ia3": 1e-6, "dora": 5e-5}[variant]
    assert identity_err < identity_tol

    before_perturb = adapter_state(model)
    perturb_first_adapter(model)
    with torch.no_grad():
        perturb_delta = (model(ids) - y_init).abs().max().item()
    assert perturb_delta > 1e-7
    for name, value in before_perturb.items():
        model.state_dict()[name].copy_(value)

    path = ARTIFACT_DIR / f"{variant}_adapter.pt"
    ll.save(model, str(path))
    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert set(saved["state"]) == set(adapter_state(model))
    assert any(k.startswith("layers.0.q_proj.lora_") for k in saved["state"])

    torch.manual_seed(0)
    model_loaded = TinyModel()
    ll.load(model_loaded, str(path))
    loaded_state = adapter_state(model_loaded)
    for name, value in saved["state"].items():
        assert torch.equal(loaded_state[name].cpu(), value)
    with torch.no_grad():
        y_loaded = model_loaded(ids)
    assert (y_loaded - y_init).abs().max().item() < identity_tol

    torch.manual_seed(0)
    train_model = TinyModel()
    ll.attach(train_model, cfg_for_variant(variant, training=True))
    assert_only_lora_trainable(train_model)
    target = torch.randn(2, 16, 100) * 0.1
    trainable = [p for p in train_model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=0.1) if variant in ("delora", "ia3") else (
        torch.optim.Adam(trainable, lr=1e-3) if variant == "dora" else torch.optim.SGD(trainable, lr=1e-2)
    )
    losses = []
    first_grad_norm = math.nan
    before_train = adapter_state(train_model)
    for step in range(20):
        opt.zero_grad()
        loss = (train_model(ids) - target).pow(2).mean()
        loss.backward()
        assert_no_base_grads(train_model)
        grad_norm = sum(
            p.grad.detach().float().norm().item()
            for name, p in train_model.named_parameters()
            if "lora_" in name and p.grad is not None
        )
        assert math.isfinite(grad_norm)
        if step == 0:
            first_grad_norm = grad_norm
        opt.step()
        losses.append(loss.item())
    after_train = adapter_state(train_model)
    adapter_delta = sum((after_train[k] - before_train[k]).float().norm().item() for k in before_train)
    drop = (losses[0] - losses[-1]) / losses[0]
    assert first_grad_norm > 0
    assert adapter_delta > 0
    assert drop > 0.05


def test_load_fails_on_missing_and_unexpected_lora_keys():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model = TinyModel()
    ll.attach(model, cfg_for_variant("lora"))
    good_path = ARTIFACT_DIR / "lora_good.pt"
    ll.save(model, str(good_path))
    blob = torch.load(good_path, weights_only=True, map_location="cpu")

    missing_blob = {"cfg": blob["cfg"], "state": dict(blob["state"])}
    missing_blob["state"].pop(next(iter(missing_blob["state"])))
    missing_path = ARTIFACT_DIR / "lora_missing.pt"
    torch.save(missing_blob, missing_path)
    with pytest.raises(RuntimeError, match="missing lora keys"):
        ll.load(TinyModel(), str(missing_path))

    unexpected_blob = {"cfg": blob["cfg"], "state": dict(blob["state"])}
    unexpected_blob["state"]["layers.0.q_proj.lora_extra"] = torch.zeros(1)
    unexpected_path = ARTIFACT_DIR / "lora_unexpected.pt"
    torch.save(unexpected_blob, unexpected_path)
    with pytest.raises(RuntimeError, match="unexpected lora keys"):
        ll.load(TinyModel(), str(unexpected_path))


def test_no_target_layers_is_loud_failure():
    cfg = ll.LoraLiteConfig(variant="lora", target_names=("definitely_missing",))
    with pytest.raises(RuntimeError, match="no target layers"):
        ll.attach(TinyModel(), cfg)


@pytest.mark.parametrize("variant", ["lora", "delora", "ia3"])
def test_structural_non_linear_target_trains_for_forward_only_variants(variant: str):
    torch.manual_seed(0)
    model = FakeBnbModel()
    x = torch.randn(2, 3, 8)
    y_base = model(x).detach()
    cfg = ll.LoraLiteConfig(
        variant=variant,
        r=2,
        alpha=4,
        dtype=torch.float32,
        target_roles=(),
        variant_kwargs={"lambda0": 0.0} if variant == "delora" else {},
    )
    ll.attach(model, cfg)
    y_init = model(x)
    assert (y_init.detach() - y_base).abs().max().item() < 1e-6
    loss = y_init.pow(2).mean()
    loss.backward()
    assert_no_base_grads(model)
    adapter_grad_norm = sum(
        p.grad.detach().float().norm().item()
        for name, p in model.named_parameters()
        if "lora_" in name and p.grad is not None
    )
    assert adapter_grad_norm > 0


@pytest.mark.parametrize("variant", ["pissa", "dora"])
def test_weight_reading_variants_reject_structural_non_linear_target(variant: str):
    cfg = ll.LoraLiteConfig(variant=variant, r=2, alpha=2, dtype=torch.float32, target_roles=())
    with pytest.raises(TypeError, match="plain nn.Linear"):
        ll.attach(FakeBnbModel(), cfg)