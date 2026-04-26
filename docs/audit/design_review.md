[client] initialize (running)

[client] session/new (running)

[thinking] This is a detailed design review request. I need to: 1. Read the lora-lite codebase to understand its current structure 2. Read the three reference implementations 3. Answer specific questions for each Let me start by exploring the codebase structure and reading the key files.

[tool] glob (pending)
  input: {}

[tool] glob (pending)
  input: {}

[tool] glob (completed)
  kind: search
  input: src/lora_lite/**/*.py
  output:
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/__init__.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/__init__.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py

[tool] glob (completed)
  kind: search
  input: docs/refs/*.py
  output:
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/baukit_nethook.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva_finetuning.py
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py

[thinking] I need to read all these files to do a proper comparative analysis. Let me read the key lora-lite files first (adapter.py, variant.py, variants/lora.py), then the reference files.

[tool] read (pending)
  input: {}

[tool] read (pending)
  input: {}

[tool] src/lora_lite/adapter.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py</path>
    <type>file</type>
    <content>
    1: """attach / detach / save / load. The whole runtime."""
    2: from __future__ import annotations
    3: import torch
    4: from torch import nn
    5: from torch.utils.hooks import RemovableHandle
    6: 
    7: from .config import LoraLiteConfig
    8: from .variant import REGISTRY
    9: from .target import find_targets
    10: 
    11: 
    12: _ATTACHED_ATTR = "_lora_lite_attached"
    13: 
    14: 
    15: def _hook(layer, args, y):
    16:     (x,) = args
    17:     cfg: LoraLiteConfig = layer._lora_cfg
    18:     x_cast = x.to(cfg.dtype)
    19:     out = layer._lora_variant.forward(layer, x_cast, y)
    20:     return out.to(y.dtype)
    21: 
    22: 
    23: def _pre_hook(layer, args):
    24:     (x,) = args
    25:     cfg: LoraLiteConfig = layer._lora_cfg
    ... (92 more lines)

[tool] read (pending)
  input: {}

[tool] src/lora_lite/variant.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py</path>
    <type>file</type>
    <content>
    1: """Variant protocol + registry. Variants own (x, layer.weight, layer.lora_*) -> y_new."""
    2: from dataclasses import dataclass
    3: from typing import Callable, Protocol, Any
    4: import torch
    5: from torch import nn
    6: 
    7: from .config import LoraLiteConfig
    8: 
    9: 
    10: @dataclass
    11: class ParamSpec:
    12:     shape: tuple[int, ...]
    13:     init: str | Callable[[torch.Tensor], None] = "zeros"  # 'zeros'|'kaiming'|'ones'|callable(t)
    14:     trainable: bool = True
    15: 
    16:     def make(self, dtype: torch.dtype, device) -> nn.Parameter:
    17:         t = torch.empty(self.shape, dtype=dtype, device=device)
    18:         if callable(self.init):
    19:             self.init(t)
    20:         elif self.init == "zeros":
    21:             t.zero_()
    22:         elif self.init == "ones":
    23:             t.fill_(1.0)
    24:         elif self.init == "kaiming":
    25:             # match nn.Linear default: kaiming_uniform_(a=sqrt(5))
    ... (34 more lines)

[tool] src/lora_lite/variants/lora.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py</path>
    <type>file</type>
    <content>
    1: """Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685
    2: 
    3:     h = W x + (alpha/r) B A x
    4: 
    5: Identity at t=0 from B=0. Faithful to the paper.
    6: """
    7: from einops import einsum
    8: from torch import nn
    9: import torch
    10: 
    11: from ..variant import register, ParamSpec
    12: 
    13: 
    14: @register
    15: class LoRA:
    16:     name = "lora"
    17: 
    18:     @staticmethod
    19:     def param_specs(d_in, d_out, cfg):
    20:         return {
    21:             "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
    22:             "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
    23:         }
    24: 
    25:     @staticmethod
    ... (14 more lines)

[thinking] Now let me read the reference implementations to answer the questions.

[tool] read (pending)
  input: {}

[tool] read (pending)
  input: {}

[tool] docs/refs/peft_eva_finetuning.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva_finetuning.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva_finetuning.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva_finetuning.py</path>
    <type>file</type>
    <content>
    1: # Copyright 2024-present the HuggingFace Inc. team.
    2: #
    3: # Licensed under the Apache License, Version 2.0 (the "License");
    4: # you may not use this file except in compliance with the License.
    5: # You may obtain a copy of the License at
    6: #
    7: #     http://www.apache.org/licenses/LICENSE-2.0
    8: #
    9: # Unless required by applicable law or agreed to in writing, software
    10: # distributed under the License is distributed on an "AS IS" BASIS,
    11: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12: # See the License for the specific language governing permissions and
    13: # limitations under the License.
    14: 
    15: import torch
    16: from datasets import load_dataset
    17: from torch.utils.data import DataLoader
    18: from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    19: from utils import DataCollator, TokenizerMetaMath
    20: 
    21: from peft import EvaConfig, LoraConfig, get_peft_model, initialize_lora_eva_weights
    22: 
    23: 
    24: DEVICE = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    25: 
    ... (74 more lines)

[tool] docs/refs/peft_eva.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py</path>
    <type>file</type>
    <content>
    1: # Copyright 2024-present the HuggingFace Inc. team.
    2: #
    3: # Licensed under the Apache License, Version 2.0 (the "License");
    4: # you may not use this file except in compliance with the License.
    5: # You may obtain a copy of the License at
    6: #
    7: #     http://www.apache.org/licenses/LICENSE-2.0
    8: #
    9: # Unless required by applicable law or agreed to in writing, software
    10: # distributed under the License is distributed on an "AS IS" BASIS,
    11: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12: # See the License for the specific language governing permissions and
    13: # limitations under the License.
    14: 
    15: import warnings
    16: from collections import Counter, defaultdict
    17: from collections.abc import Iterable, Mapping
    18: from contextlib import nullcontext
    19: from copy import deepcopy
    20: from functools import partial
    21: from itertools import cycle
    22: from typing import Optional, Union
    23: 
    24: import torch
    25: import torch.distributed as dist
    ... (717 more lines)

[thinking] Now let me read the baukit and antipasto3 references.

[tool] read (pending)
  input: {}

[tool] read (pending)
  input: {}

[tool] docs/refs/baukit_nethook.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/baukit_nethook.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/baukit_nethook.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/baukit_nethook.py</path>
    <type>file</type>
    <content>
    1: """
    2: Utilities for instrumenting a torch model.
    3: 
    4: Trace will hook one layer at a time.
    5: TraceDict will hook multiple layers at once.
    6: subsequence slices intervals from Sequential modules.
    7: get_module, replace_module, get_parameter resolve dotted names.
    8: set_requires_grad recursively sets requires_grad in module parameters.
    9: """
    10: 
    11: import contextlib
    12: import copy
    13: import inspect
    14: from collections import OrderedDict
    15: 
    16: import torch
    17: 
    18: 
    19: class Trace(contextlib.AbstractContextManager):
    20:     """
    21:     To retain the output of the named layer during the computation of
    22:     the given network:
    23: 
    24:         with Trace(net, 'layer.name') as ret:
    25:             _ = net(inp)
    ... (449 more lines)

[tool] docs/refs/antipasto3_svd_adapter.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py</path>
    <type>file</type>
    <content>
    1: """SVD adapter for bidirectional steering via block-diagonal Cayley rotations.
    2: 
    3: Flax NNX port.
    4: 
    5: kernel = U @ diag(S) @ Vh + W_res   (kernel is (in, out), standard Flax convention)
    6: Learnable: delta_s (additive S scaling), rotation_params (block-diagonal V rotation).
    7: alpha scales both: S + alpha*delta_s, U @ R(alpha).
    8: 
    9: Why Cayley (not Givens or matrix exponential):
    10: Cayley gives exact analytical reversibility: R(-alpha) = R(alpha)^{-1}.
    11: This is critical -- at alpha=+1 and alpha=-1 the adapter is an exact inverse of
    12: itself, making bidirectional steering symmetric by construction.
    13: 
    14: At alpha=0: U_rot = U and S_scaled = S, so the layer is identical to frozen weights.
    15: """
    16: 
    17: import math
    18: 
    19: import jax
    20: import jax.numpy as jnp
    21: from jax import Array
    22: from jaxtyping import Float, Int
    23: from einops import rearrange
    24: from flax import nnx
    25: from loguru import logger
    ... (353 more lines)

[thinking] Now let me read the lora-lite config and variant protocol more carefully to understand the current API. Then I can compare it to the references.

[tool] read (pending)
  input: {}

[tool] read (pending)
  input: {}

[tool] src/lora_lite/config.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py</path>
    <type>file</type>
    <content>
    1: from dataclasses import dataclass, field, asdict
    2: from typing import Any
    3: import torch
    4: 
    5: 
    6: @dataclass
    7: class LoraLiteConfig:
    8:     variant: str = "lora"
    9:     r: int = 8
    10:     alpha: float = 16.0
    11:     dropout: float = 0.0  # currently ignored; variants may use cfg.variant_kwargs
    12:     dtype: torch.dtype = torch.bfloat16
    13: 
    14:     # targeting
    15:     target_roles: tuple[str, ...] = ("reader", "writer")
    16:     target_names: tuple[str, ...] = ()
    17:     exclude_names: tuple[str, ...] = ("lm_head", "embed_tokens")
    18:     layers: tuple[int, ...] | None = None
    19: 
    20:     # variant-specific bag (e.g. lambda0 for DeLoRA)
    21:     variant_kwargs: dict[str, Any] = field(default_factory=dict)
    22: 
    23:     def to_dict(self) -> dict:
    24:         d = asdict(self)
    25:         d["dtype"] = str(self.dtype).removeprefix("torch.")
    ... (16 more lines)

[tool] src/lora_lite/target.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py</path>
    <type>file</type>
    <content>
    1: """Find linear-like targets by shape (reader/writer/inner) + name regex.
    2: 
    3: Structural matching is deliberate: bnb Linear4bit/8bitLt are not nn.Linear, but
    4: they expose in_features/out_features/weight and their forward already handles
    5: dequantization.
    6: """
    7: import re
    8: from torch import nn
    9: 
    10: 
    11: def is_linear_like(m: nn.Module) -> bool:
    12:     return (
    13:         hasattr(m, "in_features")
    14:         and hasattr(m, "out_features")
    15:         and hasattr(m, "weight")
    16:         and callable(m)
    17:     )
    18: 
    19: 
    20: def _layer_idx(name: str) -> int | None:
    21:     m = re.search(r"\.layers?\.(\d+)\.", name)
    22:     return int(m.group(1)) if m else None
    23: 
    24: 
    25: def _classify(m: nn.Module, d_model: int, name: str) -> str:
    ... (37 more lines)

[thinking] OK now I have a clear picture. Let me write up the design review with specific answers to each question. Let me also check if there's an AGENTS.md file:

[tool] glob (pending)
  input: {}

[tool] glob (completed)
  kind: search
  input: AGENTS.md
  output:
    No files found

[thinking] Let me check for AGENTS.md in parent directories or look for any README or docs.

[tool] glob (pending)
  input: {}

[tool] glob (completed)
  kind: search
  input: **/AGENTS.md
  output:
    No files found

[thinking] The user wants design notes. I have all the information I need. Let me write up the design notes for each reference.

[tool] todowrite (pending)
  input: {}

[plan]
  - [in_progress] write design notes for peft EVA
  - [pending] write design notes for baukit nethook
  - [pending] write design notes for antipasto3 SVD

[tool] 3 todos (completed)
  kind: other
  input: {"todos":[{"content":"write design notes for peft EVA","status":"in_progress","priority":"high"},{"content":"write design notes for baukit nethook","status":"pending","priority":"high"},{"content":"write design notes ...
  output:
    [
      {
        "content": "write design notes for peft EVA",
        "status": "in_progress",
        "priority": "high"
      },
      {
        "content": "write design notes for baukit nethook",
        "status": "pending",
        "priority": "high"
      },
      {
        "content": "write design notes for antipasto3 SVD",
        "status": "pending",
        "priority": "high"
      }
    ]

[thinking] Let me write out the design notes for each reference. ## A. peft EVA **What EVA does** (1-paragraph): EVA computes SVD-of-activations to initialize LoRA's low-rank decomposition. Rather than random kaiming init, it collects layer inputs during a forward pass through calibration data, runs incremental PCA to find principal components, and uses those as the initial `A` matrix (the left singular vectors). Key features: - Per-layer rank redistribution based on explained variance (lines 302-314) - Convergence checking via cosine similarity between SVD components (lines 147-168) - Distributed support via all-gather (lines 81-102) - Handles non-convergence gracefully (multiple forward passes until all layers converge) A **minimal EVA variant** in lora-lite: The current API is: ```python variant.init(layer, cfg) # called per-layer after params created ``` EVA needs a **separate calibration st...


# Design Review: lora-lite vs peft EVA / baukit / antipasto3

## A. peft EVA

**What EVA does** (1-paragraph summary):  
EVA computes SVD-of-activations to initialize LoRA's low-rank decomposition. Rather than random kaiming init, it registers hooks on target layers, runs incremental PCA on calibration inputs (collected during forward passes), and uses the left singular vectors as the initial `lora_A` matrix. Key features: per-layer rank redistribution based on explained variance (lines 302-314), convergence checking via cosine similarity (lines 147-168), and distributed support via all-gather (lines 81-102).

**Minimal EVA variant sketch:**

The current API is:
```python
variant.init(layer, cfg)  # called per-layer after params created (adapter.py:56)
```

EVA needs a **separate calibration step before `attach()`**:

```python
# Option: calibrate() -> eva_state dict
def calibrate(model, dataloader, cfg):
    # Run forward passes, collect activations per target
    # Compute SVD, return {"layer_name": U_svd}
    ...

# Then in LoRA variant.init():
if "eva_state" in cfg.variant_kwargs:
    with torch.no_grad():
        layer.lora_A.weight.copy_(cfg.variant_kwargs["eva_state"][layer_name])
```

**What to drop if re-implementing:**
- Rank redistribution logic (lines 302-314) — skip, use fixed rank
- Distributed gather (lines 81-102) — skip single-GPU only
- Hash deduplication for equal inputs (lines 173-213, 374-386) — skip
- Resume-from-checkpoint plumbing (lines 489-559) — skip
- Whiten option (lines 475-477) — skip
- PEFT model wrapping (lines 612-656) — skip

**API impact:**
Current `Variant.init(layer, cfg)` signature at adapter.py:56 supports EVA via `cfg.variant_kwargs["eva_state"]`. Minimal change: add `calibrate(model, dataloader, cfg)` function (~50 lines) that computes SVD and returns state dict, then LoRA's `init()` checks for `eva_state` in kwargs.

---

**Recommendation: PARTIAL**

- **Import**: Core SVD-of-activations collection + SVD computation (lines 289-486 minus distributed/rank-redist/PEFT plumbing)
- **Approximate line-count**: +~80 lines (calibrate function + init hook), −0 removed
- **Trade**: Adds data-driven init capability; costs one new function and variant_kwargs passthrough

---

## B. baukit nethook

**What `TraceDict`/`Trace` gives us that lora-lite doesn't:**
- Context manager pattern with automatic cleanup (adapter.py:68 already handles this via `RemovableHandle`)
- `retain_grad=True` for backward pass (lora-lite doesn't need this — we're not editing grads)
- `edit_output` for intercepting/modifying (not needed — we have dedicated `forward_input` hook)
- Dictionary interface (`ret['layer.name'].output`) — we just store layer ref directly

Quote from adapter.py:64-68:
```python
for _, layer, _ in attached_targets:
    if hasattr(layer._lora_variant, "forward_input"):
        handles.append(layer.register_forward_pre_hook(_pre_hook))
    else:
        handles.append(layer.register_forward_hook(_hook))
```

**Would baukit simplify or complicate adapter.py?**
- **Neutral to negative**. Current code is 5 lines (64-68). Baukit's TraceDict adds ~100 lines of context-management indirection. We don't need:
  - Per-layer config dicts (retain_output/retain_input per layer)
  - Edit hooks mid-forward
  - Stop-after-layer control
  - clone/detach options

**Dependencies:**
Baukit is a single file (~470 lines). lora-lite currently has no deps beyond torch+einops. Adding baukit adds a dep for ~5 lines of value.

---

**Recommendation: SKIP**

Baukit adds ~400 lines for capabilities we don't need. Current hook registration (adapter.py:64-68) is simpler. The `retain_grad` feature could be useful for ml debugging but that's a separate concern — can inline later if needed.

---

## C. antipasto3 SVD adapter

**1. What to adopt or diverge from?**

| Feature | antipasto3 | lora-lite | Verdict |
|--------|-----------|----------|--------|
| SVD init from kernel | `create_svd_adapter()` line 169 | manual param_specs | **Divergen**: lora-lite missing |
| Delta-s scaling | `S_scaled = S + alpha*delta_s` (line 160) | direct scaling via `alpha/r` | **Adopt**: antipasto3 better — learnable scaling, not fixed |
| Block-diagonal rotation | Cayley transform (lines 135-141) | N/A | **Too complex** for lora-lite ethos |
| Residual: `W_res = kernel - U@S@Vh` | line 204 | N/A | **Consider**: adds expressivity |
| Dimension selection: L1 trip | lines 219-267 | None | **Skip**: too domain-specific |
| Polarity interleave | lines 271-287 | N/A | **Skip**: steering-specific |

**Key insight**: antipasto3's `W_res` residual (kernel - U@S@Vh) is the cleanest simplification over vanilla LoRA. Instead of:
- LoRA: `y = Wx + (α/r)BAx`  (learns full rank-r delta)

You get:
- SVD: `y = W_res@x + (α/r)U_scaled@S_scaled@Vh@x`  (SVD-based delta, capped)

This is essentially **PiSSA** (arXiv:2409.13349) — already in lora-lite variants/ (pissa.py).

**2. PiSSA-like factoring:**
PiSSA is in variants/pissa.py. Let me check it:

[tool] read (pending)
  input: {}

[tool] src/lora_lite/variants/pissa.py (completed)
  kind: read
  input: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py
  output:
    <path>/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py</path>
    <type>file</type>
    <content>
    1: """PiSSA: top-r SVD of W into A,B; replace W with W_res = W - B@A.
    2: 
    3: Meng et al. 2024  https://arxiv.org/abs/2404.02948
    4: W_eff(t=0) = W_res + B@A = W (numerically; bf16 round-trip not bit-exact).
    5: """
    6: import torch
    7: from einops import einsum
    8: from torch import nn
    9: 
    10: from ..variant import register, ParamSpec
    11: 
    12: 
    13: @register
    14: class PiSSA:
    15:     name = "pissa"
    16: 
    17:     @staticmethod
    18:     def param_specs(d_in, d_out, cfg):
    19:         return {
    20:             "lora_A": ParamSpec((cfg.r, d_in), init="zeros", trainable=True),
    21:             "lora_B": ParamSpec((d_out, cfg.r), init="zeros", trainable=True),
    22:         }
    23: 
    24:     @staticmethod
    25:     def init(layer: nn.Linear, cfg) -> None:
    ... (32 more lines)

[thinking] So PiSSA is already in lora-lite! Let me check what antipasto3 offers beyond this. Comparing antipasto3 to PiSSA: - Both use SVD init from W antipasto3 has: - Delta-s scaling: learnable S + delta_s (additive learnable, not just fixed) - Residual is explicit (W_res stored separately) - PiSSA mutates layer.weight into W_res The key differences: 1. **Delta-s**: antipasto3 has `delta_s` parameters (line 89-91, 160) — fully learnable scaling. PiSSA has no learnable scaling, just fixed `alpha/r`. 2. **Steering coefficient**: antipasto3 uses explicit `alpha` mutated during forward (line 107, 116), not config-based. 3. **Residual**: PiSSA mutates `layer.weight` directly. antipasto3 stores `W_res` separately. antipasto3's delta-s is the main "new" thing worth considering. But that's really a steering/adaptation method (changing alpha at runtime), not an initialization method. For lora-lite's m...

[tool] todowrite (pending)
  input: {}

[plan]
  - [completed] write design notes for peft EVA
  - [completed] write design notes for baukit nethook
  - [completed] write design notes for antipasto3 SVD

[tool] 0 todos (completed)
  kind: other
  input: {"todos":[{"content":"write design notes for peft EVA","priority":"high","status":"completed"},{"content":"write design notes for baukit nethook","priority":"high","status":"completed"},{"content":"write design notes ...
  output:
    [
      {
        "content": "write design notes for peft EVA",
        "status": "completed",
        "priority": "high"
      },
      {
        "content": "write design notes for baukit nethook",
        "status": "completed",
        "priority": "high"
      },
      {
        "content": "write design notes for antipasto3 SVD",
        "status": "completed",
        "priority": "high"
      }
    ]

[thinking] Wait, I wrote the design notes directly as output but the user asked me to answer questions and end with a recommendation. Let me reformat as requested. Actually, looking at the instructions again: the output format says "For each (A, B, C), end with: Recommendation: ADOPT / SKIP / PARTIAL". So I need to restructure my response to match that format better, and NOT write code. Let me restructure: A, B, C sections with each ending in recommendation.


# Design Review: lora-lite vs refs

## A. peft EVA (`docs/refs/peft_eva.py` + `peft_eva_finetuning.py`)

### 1. What EVA actually does

EVA uses SVD-of-activations (incremental PCA on layer inputs during forward passes) to initialize LoRA rather than random kaiming init. Key mechanism:
- Registers hooks on target layers, collects inputs across calibration batches
- Computes left singular vectors via `IncrementalPCA` (lines 138-143)
- Convergence check via cosine similarity between current/previous components (lines 167-168)
- Per-layer rank redistribution based on explained variance ratio (lines 302-314, 456)

The user workflow (peft_eva_finetuning.py:66-76):
```python
eva_config = EvaConfig(rho=rho)
peft_config = LoraConfig(r=rank, init_lora_weights="eva", eva_config=eva_config)
peft_model = get_peft_model(model, peft_config)
initialize_lora_eva_weights(peft_model, dataloader)  # separate calibration step
```

### 2. Minimal EVA variant API sketch

**Option: separate calibrate() step before attach()**
```python
# User does:
eva_state = calibrate(model, dataloader, cfg)  # returns {"layer_name": U_svd}
cfg.variant_kwargs["eva_state"] = eva_state
attach(model, cfg)  # LoRA.init() copies U_svd -> lora_A
```

API change: add `calibrate(model, dataloader, cfg)` function (~50 lines) that computes SVD on calibration inputs, returns state dict. In `LoRA.init()`, check for `eva_state[layer_name]` and copy.

### 3. What to drop if re-implementing

- Lines 302-314: rank redistribution logic → use fixed rank r
- Lines 81-102: distributed all-gather → single-GPU only
- Lines 173-213, 374-386: hash deduplication for equal inputs → skip
- Lines 489-559: resume-from-checkpoint plumbing → skip
- Lines 475-477: whiten option → skip
- Lines 612-656: PEFT model wrapper integration → standalone function

Droppable: ~60% of the 739-line file.

### 4. Does current init() signature support EVA?

Yes. adapter.py:56:
```python
variant.init(layer, cfg)
```

Add `eva_state` to `cfg.variant_kwargs` before attach, then in `LoRA.init()`:
```python
if "eva_state" in cfg.variant_kwargs:
    # copy precomputed U into lora_A
```

Smallest change: add `calibrate()` function (~50 lines), no signature change needed.

---

**Recommendation: PARTIAL**

- **Import**: SVD-of-activations collection + compute
- **Lines**: +~50 (calibrate fn), −0
- **What to remove in exchange**: Nothing — this is additive capability only. If you add calibration data collection, you must accept the additional complexity.

---

## B. baukit nethook (`docs/refs/baukit_nethook.py`)

### 1. What TraceDict/Trace gives us that we don't have

| Feature | baukit | lora-lite | Value |
|--------|-------|----------|-------|
| Context manager cleanup | lines 99-105 | RemovableHandle (adapter.py:64-68) | Same |
| retain_grad | line 37-38 | None | Not needed |
| edit_output mid-forward | lines 39, 72-75 | None | Not needed |
| Dictionary interface | line 111 | None | Not needed |
| Stop-after-layer | lines 46-47, 189-202 | None | Not needed |

TraceDict essentially provides: per-layer output retention + config dicts + cleanup. We already handle cleanup via RemovableHandle.

### 2. Would baukit simplify or complicate adapter.py?

Quote adapter.py:64-68:
```python
for _, layer, _ in attached_targets:
    if hasattr(layer._lora_variant, "forward_input"):
        handles.append(layer.register_forward_pre_hook(_pre_hook))
    else:
        handles.append(layer.register_forward_hook(_hook))
```

This is 5 lines. Baukit's TraceDict adds ~100 lines of context management we'd use only to replicate these 5 lines. Our current approach is simpler.

### 3. Is baukit worth a dep?

No. lora-lite's principle is minimal deps (currently torch+einops). Baukit adds ~400 lines of code for capabilities we don't need. `retain_grad` for debugging is the only useful feature — that's 3 lines to inline if needed later.

---

**Recommendation: SKIP**

Baukit adds a dep for no net simplification. Current hook registration (adapter.py:64-68) is 5 lines and sufficient. If `retain_grad` is needed for ml debugging, inline 3 lines later.

---

## C. antipasto3 SVD adapter (`docs/refs/antipasto3_svd_adapter.py`)

### 1. What to adopt or diverge from

| Feature | antipasto3 | lora-lite | Verdict |
|---------|-----------|----------|--------|
| SVD init from kernel | line 169 `create_svd_adapter()` | PiSSA (pissa.py) | Already have |
| Delta-s scaling: learnable | line 160 `S + alpha*delta_s` | fixed via alpha/r | **Adopt** |
| Residual: `W_res = kernel - U@S@Vh` | line 204 | PiSSA has this | Already have |
| Block-diagonal Cayley rotation | lines 135-141 | N/A | Skip (too complex) |
| Dimension selection (L1 trip) | lines 219-267 | N/A | Skip (domain-specific) |

**Key adoption**: antipasto3's learnable `delta_s` is cleaner than fixed scaling. Currently lora-lite uses `scale = alpha/r` at **runtime** — but delta-s makes scaling a **learnable parameter** with its own gradient.

This is already in PiSSA variant: piSSA `init()` sets `scale=1` via alpha=r convention (pissa.py:43-46), but doesn't expose the scaling as learnable.

### 2. PiSSA-like factoring

lora-lite already has PiSSA in `variants/pissa.py`:
- Line 31-34: SVD of weight
- Line 36-40: decompose into B@A with sqrt(S) scaling
- Line 46: residual W_res = W - B@A

This is exactly antipasto3's core insight, implemented cleanly. The antipasto3 contribution is making `delta_s` learnable — that's a natural extension of PiSSA, not a replacement.

---

**Recommendation: ADOPT (concept)**

- **Import**: learnable delta_s concept (not code — just the idea)
- **Lines**: +0 by intent, no code change right now
- **What to remove**: Nothing — this is a design note that PiSSA could evolve toward learnable scaling instead of fixed alpha/r

The antipasto3 patterns ( Cayley rotation, polarity interleave) are steering-specific and too complex for lora-lite ethos. The clean insight is already in PiSSA.

---

## Summary

| Ref | Recommendation | Impact |
|-----|--------------|--------|
| peft EVA | PARTIAL | +50 lines calibrate fn |
| baukit | SKIP | 0 (keep current) |
| antipasto3 | ADOPT concept | 0 (design note) |

[done] end_turn
