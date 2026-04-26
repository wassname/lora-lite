"""bnb 4bit/8bit CUDA smoke. Skipped without CUDA + bitsandbytes installed."""
from __future__ import annotations

import pytest
import torch
from torch import nn

import lora_lite as ll


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
bnb = pytest.importorskip("bitsandbytes")


CFG_BY_VARIANT = {
    "lora": ll.LoRAConfig,
    "ia3": ll.IA3Config,
    "hra": ll.HRAConfig,
    "pissa": ll.PiSSAConfig,
    "dora": ll.DoRAConfig,
}


class BnbModel(nn.Module):
    def __init__(self, layer_cls):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.layers = nn.ModuleList([layer_cls(8, 8, bias=False)]).cuda()

    def forward(self, x):
        return self.layers[0](x)


@pytest.mark.parametrize("layer_cls", [bnb.nn.Linear8bitLt, bnb.nn.Linear4bit])
@pytest.mark.parametrize("variant", ["lora", "ia3", "hra"])
def test_hook_only_variants_attach_to_bnb(layer_cls, variant):
    """LoRA / IA3 / HRA only hook outputs; bnb dequantization is the layer's job."""
    torch.manual_seed(0)
    model = BnbModel(layer_cls)
    x = torch.randn(2, 3, 8, device="cuda")
    y_base = model(x).detach()

    cfg = CFG_BY_VARIANT[variant](r=2, alpha=4, dtype=torch.float16, target_roles=())
    ll.attach(model, cfg)
    y = model(x)
    assert (y.detach() - y_base).abs().max().item() < 1e-2

    y.pow(2).mean().backward()
    grad_total = sum(
        g.abs().sum().item()
        for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad and (g := p.grad) is not None
    )
    assert grad_total > 0


@pytest.mark.parametrize("layer_cls", [bnb.nn.Linear8bitLt, bnb.nn.Linear4bit])
@pytest.mark.parametrize("variant", ["pissa", "dora"])
def test_weight_reading_variants_reject_bnb(layer_cls, variant):
    model = BnbModel(layer_cls)
    cfg = CFG_BY_VARIANT[variant](r=2, alpha=2, dtype=torch.float16, target_roles=())
    with pytest.raises((TypeError, RuntimeError, AttributeError, ValueError)):
        ll.attach(model, cfg)
