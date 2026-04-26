[client] initialize (running)

[client] session/new (running)
I’ll use the `gsd` skill here because this is a multi-part audit: first I’ll read the local implementation and references, then verify URLs/snapshots, then review the two harnesses and collapse it into the requested verdict tables.

[tool] Read SKILL.md (running)
  input: {"call_id":"call_hMhlWCMXk8cPaOEh4mCYl3wE","process_id":"55868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
  files: /home/wassname/.claude/skills/gsd/SKILL.md

[tool] Read SKILL.md (completed)
  kind: read
  input: {"call_id":"call_hMhlWCMXk8cPaOEh4mCYl3wE","process_id":"55868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
  files: /home/wassname/.claude/skills/gsd/SKILL.md
  output:
    ---
    name: gsd
    description: "Get Shit Done workflow for non-trivial tasks: spec -> implement -> test -> review -> wrap. Mandatory for anything with more than one step."
    ---
    
    <!--
    Design rationale and sources — for human readers, not the agent.
    
    - GSD spec-first: spec-driven development, scope fencing, verify-before-done
    - Manus file-as-memory: one file is working memory, checklist is state, append-only log
    - Boomerang: return concrete artifacts not summaries
    - DCP: invisible dedup, superseded writes, error purging
    
    Core principles:
    - Spec file is the only persistent artifact (single-file invariant).
    - Agent shouldn't know about its own resource limits.
    - Review scales with input size: frontier for plans, cheap for bulk.
    - Observable outputs: show evidence, not narration.
    -->
    
    # GSD (Get Shit Done)
    
    Mandatory for all non-trivial tasks (more than one step). Skip only for single-line fixes.
    
    ## Part 1: Core workflow
    
    ### 1.1 Spec-first planning
    
    ... (192 more lines)

[plan]
  - [in_progress] Inventory source, refs, tests
  - [pending] Verify external links and snapshot drift
  - [pending] Audit each variant against refs
  - [pending] Assess smoke and Qwen probe validity
  - [pending] Assemble concise findings with line citations
I’m starting with a source inventory: variant implementations, runtime attach/load code, local reference snapshots, and the two harnesses. The key thing I’m extracting is line-numbered evidence so the final review can stay specific rather than hand-wavy.

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/refs (running)
  input: {"call_id":"call_WTlnC2ndhh62jn4E6rDNB6nY","process_id":"48507","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","find docs/refs -maxdepth 2 -type f"],"cwd":"/media/wassname/SGIronWolf/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/refs

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (running)
  input: {"call_id":"call_wXC8wsjCbDji5rdKyPNA6LrT","process_id":"22303","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite docs/refs tests scripts"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite

[tool] Run git status --short (running)
  input: {"call_id":"call_FFOxlyCNApreJiKkClbo9Hxl","process_id":"78605","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","git status --short"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/l...

[tool] Run git status --short (completed)
  kind: execute
  input: {"call_id":"call_FFOxlyCNApreJiKkClbo9Hxl","process_id":"78605","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","git status --short"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/l...
  output:
    M pyproject.toml
     M src/lora_lite/__init__.py
     M src/lora_lite/config.py
     M src/lora_lite/variants/antipasto.py
     M src/lora_lite/variants/delora.py
     M src/lora_lite/variants/dora.py
     M src/lora_lite/variants/eva.py
     M src/lora_lite/variants/hra.py
     M src/lora_lite/variants/ia3.py
     M src/lora_lite/variants/lora.py
     M src/lora_lite/variants/pissa.py
     M uv.lock
    ?? docs/audit/REVIEW_PROMPT_VARIANTS_V4.md
    ?? docs/audit/variants_review_v4.md

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/refs (completed)
  kind: search
  input: {"call_id":"call_WTlnC2ndhh62jn4E6rDNB6nY","process_id":"48507","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","find docs/refs -maxdepth 2 -type f"],"cwd":"/media/wassname/SGIronWolf/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/refs
  output:
    find: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by find)
    docs/refs/peft_delora_layer.py
    docs/refs/peft_lora_layer.py
    docs/refs/peft_eva.py
    docs/refs/baukit_nethook.py
    docs/refs/orig_pissa_init.py
    docs/refs/peft_lora_dora.py
    docs/refs/peft_hra_layer.py
    docs/refs/antipasto3_svd_adapter.py
    docs/refs/orig_delora.py
    docs/refs/peft_lora_variants.py
    docs/refs/peft_ia3_layer.py
    docs/refs/orig_hra_layer.py
    docs/refs/peft_eva_finetuning.py

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (completed)
  kind: search
  input: {"call_id":"call_wXC8wsjCbDji5rdKyPNA6LrT","process_id":"22303","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite docs/refs tests scripts"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite
  output:
    tests/test_lora_lite.py
    tests/smoke.py
    tests/_artifacts/hra_adapter.pt
    tests/_artifacts/delora_smoke_adapter.pt
    tests/_artifacts/lora_unexpected.pt
    tests/_artifacts/ia3_smoke_adapter.pt
    tests/_artifacts/lora_smoke_adapter.pt
    tests/_artifacts/lora_missing.pt
    tests/_artifacts/dora_smoke_adapter.pt
    tests/_artifacts/eva_smoke_adapter.pt
    tests/_artifacts/hra_smoke_adapter.pt
    tests/_artifacts/pissa_smoke_adapter.pt
    tests/_artifacts/dora_adapter.pt
    tests/_artifacts/lora_good.pt
    tests/_artifacts/delora_adapter.pt
    tests/_artifacts/lora_adapter.pt
    tests/_artifacts/pissa_adapter.pt
    tests/_artifacts/antipasto_smoke_adapter.pt
    tests/_artifacts/ia3_adapter.pt
    scripts/qwen_train_probe.py
    src/lora_lite/__init__.py
    src/lora_lite/variant.py
    src/lora_lite/config.py
    src/lora_lite/adapter.py
    docs/refs/peft_eva_finetuning.py
    docs/refs/orig_hra_layer.py
    docs/refs/peft_ia3_layer.py
    docs/refs/peft_lora_variants.py
    ... (19 more lines)

[tool] Read variant.py (running)
  input: {"call_id":"call_oHYOvEKikwSmKCy4dhjfCInI","process_id":"73582","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py

[tool] Read config.py (running)
  input: {"call_id":"call_6WRPNCijW4RyEvz1cHydfmnj","process_id":"3191","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/config.py"],"cwd":"/media/wassname/SGIronWolf/proje...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py

[tool] Read adapter.py (running)
  input: {"call_id":"call_6B9nl0I4kqlNBxHAeqADPA71","process_id":"94068","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py

[tool] Read target.py (running)
  input: {"call_id":"call_KrDjqr1FquYfne814Qxkvot0","process_id":"55044","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/target.py"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py

[tool] Read variant.py (completed)
  kind: read
  input: {"call_id":"call_oHYOvEKikwSmKCy4dhjfCInI","process_id":"73582","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py
  output:
    1	"""Variant protocol + registry. Variants own (x, layer.weight, layer.lora_*) -> y_new."""
         2	from dataclasses import dataclass
         3	from typing import Callable, Protocol, Any
         4	import torch
         5	from torch import nn
         6	
         7	from .config import LoraLiteConfig
         8	
         9	
        10	@dataclass
        11	class ParamSpec:
        12	    shape: tuple[int, ...]
        13	    init: str | Callable[[torch.Tensor], None] = "zeros"  # 'zeros'|'kaiming'|'ones'|callable(t)
        14	    trainable: bool = True
        15	    as_buffer: bool = False  # if True, register_buffer instead of register_parameter
        16	
        17	    def _empty(self, dtype: torch.dtype, device) -> torch.Tensor:
        18	        t = torch.empty(self.shape, dtype=dtype, device=device)
        19	        if callable(self.init):
        20	            self.init(t)
        21	        elif self.init == "zeros":
        22	            t.zero_()
        23	        elif self.init == "ones":
        24	            t.fill_(1.0)
        25	        elif self.init == "kaiming":
        26	            # match nn.Linear default: kaiming_uniform_(a=sqrt(5))
        27	            nn.init.kaiming_uniform_(t, a=5 ** 0.5) if t.ndim >= 2 else t.normal_(0, 0.02)
        28	        else:
    ... (39 more lines)

[tool] Read config.py (completed)
  kind: read
  input: {"call_id":"call_6WRPNCijW4RyEvz1cHydfmnj","process_id":"3191","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/config.py"],"cwd":"/media/wassname/SGIronWolf/proje...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py
  output:
    1	from dataclasses import dataclass, field, asdict
         2	from typing import Any
         3	import torch
         4	
         5	
         6	@dataclass
         7	class LoraLiteConfig:
         8	    variant: str = "lora"
         9	    r: int = 8
        10	    alpha: float | int = 16.0
        11	    dtype: torch.dtype = torch.bfloat16
        12	
        13	    # targeting
        14	    target_roles: tuple[str, ...] = ("reader", "writer")
        15	    target_names: tuple[str, ...] = ()
        16	    exclude_names: tuple[str, ...] = ("lm_head", "embed_tokens")
        17	    layers: tuple[int, ...] | None = None
        18	
        19	    # variant-specific bag (e.g. lambda0 for DeLoRA)
        20	    variant_kwargs: dict[str, Any] = field(default_factory=dict)
        21	
        22	    def to_dict(self) -> dict:
        23	        d = asdict(self)
        24	        d["dtype"] = str(self.dtype).removeprefix("torch.")
        25	        return d
        26	
        27	    @classmethod
        28	    def from_dict(cls, d: dict) -> "LoraLiteConfig":
    ... (9 more lines)

[tool] Read adapter.py (completed)
  kind: read
  input: {"call_id":"call_6B9nl0I4kqlNBxHAeqADPA71","process_id":"94068","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py
  output:
    1	"""attach / detach / save / load. The whole runtime."""
         2	from __future__ import annotations
         3	import torch
         4	from torch import nn
         5	from torch.utils.hooks import RemovableHandle
         6	
         7	from .config import LoraLiteConfig
         8	from .variant import REGISTRY
         9	from .target import find_targets
        10	
        11	
        12	_ATTACHED_ATTR = "_lora_lite_attached"
        13	
        14	
        15	def _hook(layer, args, y):
        16	    (x,) = args
        17	    cfg: LoraLiteConfig = layer._lora_cfg
        18	    x_cast = x.to(cfg.dtype)
        19	    out = layer._lora_variant.forward(layer, x_cast, y)
        20	    return out.to(y.dtype)
        21	
        22	
        23	def _pre_hook(layer, args):
        24	    (x,) = args
        25	    cfg: LoraLiteConfig = layer._lora_cfg
        26	    x_cast = x.to(cfg.dtype)
        27	    x_new = layer._lora_variant.forward_input(layer, x_cast)
        28	    return (x_new.to(x.dtype),)
    ... (126 more lines)

[tool] Read target.py (completed)
  kind: read
  input: {"call_id":"call_KrDjqr1FquYfne814Qxkvot0","process_id":"55044","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/target.py"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py
  output:
    1	"""Find linear-like targets by shape (reader/writer/inner) + name regex.
         2	
         3	Structural matching is deliberate: bnb Linear4bit/8bitLt are not nn.Linear, but
         4	they expose in_features/out_features/weight and their forward already handles
         5	dequantization.
         6	"""
         7	import re
         8	from torch import nn
         9	
        10	
        11	def is_linear_like(m: nn.Module) -> bool:
        12	    return (
        13	        hasattr(m, "in_features")
        14	        and hasattr(m, "out_features")
        15	        and hasattr(m, "weight")
        16	        and callable(m)
        17	    )
        18	
        19	
        20	def _layer_idx(name: str) -> int | None:
        21	    m = re.search(r"\.layers?\.(\d+)\.", name)
        22	    return int(m.group(1)) if m else None
        23	
        24	
        25	def _classify(m: nn.Module, d_model: int, name: str) -> str:
        26	    di, do = m.in_features, m.out_features
        27	    if di == d_model and do != d_model:
        28	        return "reader"
    ... (31 more lines)

[tool] Read lora.py (running)
  input: {"call_id":"call_Jh1Na1STB1xKq0EwNgOl2tMI","process_id":"97888","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py

[tool] Read pissa.py (running)
  input: {"call_id":"call_ki80zmAxjTfxGnQFtc3gmXQU","process_id":"19824","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py

[tool] Read delora.py (running)
  input: {"call_id":"call_pBUz8jzUuqsbkFPhBMFCfJgi","process_id":"82529","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py

[tool] Read ia3.py (running)
  input: {"call_id":"call_yHMhcitdVwaR7eK4so94DuIX","process_id":"27826","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py

[tool] Read lora.py (completed)
  kind: read
  input: {"call_id":"call_Jh1Na1STB1xKq0EwNgOl2tMI","process_id":"97888","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py
  output:
    1	"""Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685
         2	
         3	    h = W x + (alpha/r) B A x
         4	
         5	Identity at t=0 from B=0. Faithful to the paper.
         6	
         7	Reference implementations (for review/cross-check):
         8	  - peft Linear.update_layer + lora_A/B init, forward:
         9	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
        10	    (see docs/refs/peft_lora_layer.py for offline copy)
        11	"""
        12	from einops import einsum
        13	from jaxtyping import Float
        14	from torch import nn, Tensor as T
        15	import torch
        16	
        17	from ..variant import register, ParamSpec
        18	
        19	
        20	@register
        21	class LoRA:
        22	    name = "lora"
        23	
        24	    @staticmethod
        25	    def param_specs(d_in, d_out, cfg):
        26	        return {
        27	            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
        28	            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
    ... (18 more lines)

[tool] Read pissa.py (completed)
  kind: read
  input: {"call_id":"call_ki80zmAxjTfxGnQFtc3gmXQU","process_id":"19824","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py
  output:
    1	"""PiSSA: top-r SVD of W into A,B; replace W with W_res = W - B@A.
         2	
         3	Meng et al. 2024  https://arxiv.org/abs/2404.02948
         4	W_eff(t=0) = W_res + B@A = W (numerically; bf16 round-trip not bit-exact).
         5	
         6	DEVIATION FROM PAPER (documented):
         7	  - Paper sets adapter scale = 1 (no alpha/r factor); we keep LoRA's alpha/r
         8	    pipeline so callers must pass alpha=r to get paper-faithful identity.
         9	  - Saved adapter does NOT include W_res (would double checkpoint size). Instead
        10	    `adapter.save` records a fingerprint of the post-init base weights and
        11	    `adapter.load` re-runs PiSSA init then verifies the fingerprint matches
        12	    -- so loading onto a different base weight raises loudly instead of
        13	    silently producing wrong outputs.
        14	
        15	Reference implementations (for review/cross-check):
        16	  - PiSSA original (NeurIPS'24 spotlight) init script (SVD on dequant W):
        17	    https://github.com/MuLabPKU/PiSSA/blob/main/utils/init_pissa.py
        18	    (offline: docs/refs/orig_pissa_init.py)
        19	  - peft PiSSA flavor (init_lora_weights='pissa') in:
        20	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
        21	    (offline: docs/refs/peft_lora_layer.py, see pissa_init / loftq_init paths)
        22	"""
        23	import torch
        24	from einops import einsum
        25	from jaxtyping import Float
        26	from torch import nn, Tensor as T
        27	
        28	from ..variant import register, ParamSpec
    ... (48 more lines)

[tool] Read delora.py (completed)
  kind: read
  input: {"call_id":"call_pBUz8jzUuqsbkFPhBMFCfJgi","process_id":"82529","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py
  output:
    1	"""DeLoRA: per-input-channel weight-norm scaling, per-rank A/B normalization.
         2	
         3	Bini et al. 2025 (ICLR'25)  https://arxiv.org/abs/2503.18225
         4	
         5	Paper Eq. 8:    W' = W + (lambda * ||W||_F / r) B Xi A
         6	where Xi_{i,i} = 1 / (||b_i|| ||a_i||) makes each rank-1 component unit-norm.
         7	
         8	Implementation follows the peft upstream (which the DeLoRA authors maintain),
         9	which differs from the paper notation in two ways that are equivalent at the
        10	forward level but matter for gradients/numerics:
        11	  1. ||W|| is captured PER INPUT CHANNEL (shape (d_in,)), not as a scalar
        12	     Frobenius norm. Used to scale `x` element-wise on the input dim.
        13	     See docs/refs/peft_delora_layer.py:150 (init) and :250 (forward).
        14	  2. Per-rank normalization applied via division (1/||A_i||*||B^j||) inside
        15	     the diagonal scaling, instead of as F.normalize on A,B themselves.
        16	     This keeps the gradient flowing through the un-normalized parameters.
        17	
        18	Identity at t=0: lambda0=0 -> delta is exactly zero (bit-identity).
        19	
        20	KNOWN GRADIENT ISSUE (flagged by external review 2026-04-26):
        21	  With lambda0=0 the *forward* is identity but `A,B` get zero gradient on step 0
        22	  (delta is proportional to lambda). Only `lora_lambda` moves first step.
        23	  The paper's true initialization (frozen-copy trick, Eq. 9) achieves both
        24	  identity AND non-zero A/B gradients; we do NOT implement it here.
        25	
        26	Reference implementations:
        27	  - DeLoRA paper authors (ExplainableML/DeLoRA) -- their fork of peft:
        28	    https://github.com/ExplainableML/DeLoRA/blob/main/peft/src/peft/tuners/delora.py
    ... (61 more lines)

[tool] Read ia3.py (completed)
  kind: read
  input: {"call_id":"call_yHMhcitdVwaR7eK4so94DuIX","process_id":"27826","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py
  output:
    1	"""IA3-style elementwise gating. Liu et al. 2022  https://arxiv.org/abs/2205.05638
         2	
         3	Two registered variants, matching the paper's two regimes:
         4	
         5	* `ia3`     -- OUTPUT-side gating, parameter shape (d_out,).
         6	              y_new = y * g.  Use for attention projections (k_proj, v_proj).
         7	
         8	* `ia3_ff`  -- INPUT-side gating, parameter shape (d_in,).
         9	              y_new = base_layer(x * g).  Use for FFN-down layers (down_proj,
        10	              fc2). Equivalent to the paper's "gate the FFN intermediate (post-
        11	              activation)" position because down_proj's input IS that
        12	              intermediate hidden state.
        13	
        14	In both cases g is initialized to 1 -> identity at t=0.
        15	
        16	To match the paper exactly on a Llama/Qwen-style block requires TWO attach
        17	passes (one per variant), since each variant uses one hook type:
        18	
        19	    cfg_attn = LoraLiteConfig(variant="ia3",    target_names=(r"\\.k_proj$", r"\\.v_proj$"))
        20	    cfg_ffn  = LoraLiteConfig(variant="ia3_ff", target_names=(r"\\.down_proj$",))
        21	
        22	Reference implementation:
        23	  - peft IA3 layer (is_feedforward toggles input-vs-output gating, see
        24	    docs/refs/peft_ia3_layer.py:177-188 forward and :214 update_layer):
        25	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py
        26	"""
        27	import torch
        28	from jaxtyping import Float
    ... (44 more lines)

[tool] Read hra.py (running)
  input: {"call_id":"call_5dApBuiwrxILvOHq0UsAWrR6","process_id":"43282","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py

[tool] Read dora.py (running)
  input: {"call_id":"call_bY1P6NZAFCUJK0gMBKDoSogD","process_id":"88868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py

[tool] Read antipasto.py (running)
  input: {"call_id":"call_yooJHvjyZ7ccdKbRP1LnXWRo","process_id":"90297","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/antipasto.py"],"cwd":"/media/wassname/SGI...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py

[tool] Read eva.py (running)
  input: {"call_id":"call_mHqxO4d6Ysrff17whtDJcMmv","process_id":"5302","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/eva.py"],"cwd":"/media/wassname/SGIronWolf...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py

[tool] Read hra.py (completed)
  kind: read
  input: {"call_id":"call_5dApBuiwrxILvOHq0UsAWrR6","process_id":"43282","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py
  output:
    1	"""HRA: Householder Reflection Adaptation. Yuan et al. 2024  https://arxiv.org/abs/2405.17484
         2	
         3	Paper formulation (Sec. 3): adapt each frozen weight as
         4	
         5	    W' = W R,    R = prod_{i=1..r} H_i,    H_i = I - 2 u_i u_i^T / ||u_i||^2
         6	
         7	so the layer output becomes  y' = W' x = W (R x).  R is in INPUT space (d_in x d_in).
         8	
         9	We implement this via a `forward_input` pre-hook that returns `R x`, then the
        10	frozen base layer (including bnb 4/8-bit Linear) computes `W (R x)` itself.
        11	
        12	Identity at t=0 (PEFT-style symmetric init, requires even r):
        13	  Rows are kaiming-init in pairs: U[0]=U[1], U[2]=U[3], ...  Adjacent pairs of
        14	  Householder reflections with identical vectors cancel exactly
        15	  (H_i H_i = I), so R = I at init -> y' = y to bit-precision.
        16	  After the first gradient step the paired rows diverge and the chain becomes a
        17	  general orthogonal matrix; gradient flows into U from step 0 (no dead-grad).
        18	  Odd r is rejected (matches peft warning behaviour).
        19	
        20	OMITTED: paper also adds an orthogonality regularizer (Eq. 6 / Sec. 3.3),
        21	a loss-side term. Add it in your training loop if you want regularized HRA.
        22	
        23	Reference implementations (for review/cross-check):
        24	  - HRA paper authors (DaShenZi721/HRA), llama variant of OFT layer with HRA:
        25	    https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py
        26	    (offline: docs/refs/orig_hra_layer.py)
        27	  - peft HRA layer, reset_hra_parameters (lines 100-108):
        28	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/layer.py
    ... (54 more lines)

[tool] Read dora.py (completed)
  kind: read
  input: {"call_id":"call_bY1P6NZAFCUJK0gMBKDoSogD","process_id":"88868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py
  output:
    1	"""DoRA: weight-decomposed LoRA. Liu et al. 2024  https://arxiv.org/abs/2402.09353
         2	
         3	W' = m * V / ||V||_c   where V = W + (alpha/r) B A   (||.||_c = per-output-row L2 norm)
         4	
         5	At t=0:  B=0 -> V=W -> y_new = (m_init / ||W||_c) (Wx + 0) = Wx   when m_init = ||W||_c.
         6	
         7	Limitation: requires materializing the dense weight to compute ||V||_c. v1 supports
         8	plain nn.Linear only; bnb 4/8-bit layers raise loudly.
         9	
        10	DEVIATION (numerical):
        11	  - We differentiate through ||V||_c every forward. The paper's sec. 4.3 suggests
        12	    a 'cost-saving' variant that detaches ||V|| in backward (gradient only flows
        13	    through V); we do NOT do that. Real impact: slower step, slightly different
        14	    gradient direction. Faithful to the eq.5 forward, not the optimized one.
        15	
        16	Reference implementations (for review/cross-check):
        17	  - peft DoRA (separate file under lora/):
        18	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py
        19	    (offline: docs/refs/peft_lora_dora.py)
        20	"""
        21	import torch
        22	from einops import einsum
        23	from jaxtyping import Float
        24	from torch import nn, Tensor as T
        25	
        26	from ..variant import register, ParamSpec
        27	
        28	
    ... (46 more lines)

[tool] Read antipasto.py (completed)
  kind: read
  input: {"call_id":"call_yooJHvjyZ7ccdKbRP1LnXWRo","process_id":"90297","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/antipasto.py"],"cwd":"/media/wassname/SGI...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py
  output:
    1	"""AntiPaSTO: SVD steering with learnable singular-value deltas + block-diagonal Cayley rotation.
         2	
         3	Lite port of wassname's AntiPaSTO3 SVD adapter (research code, not an
         4	upstream peft variant). Reference:
         5	  https://github.com/wassname/antipasto3 (offline: docs/refs/antipasto3_svd_adapter.py)
         6	
         7	Decomposition (PyTorch nn.Linear convention, weight (d_out, d_in)):
         8	
         9	    W = U diag(S) Vh + W_res     (top-r SVD; W_res = W - U_r diag(S_r) Vh_r)
        10	
        11	We freeze U, S, Vh, W_res and learn:
        12	  - delta_s : (r,)                    -- additive delta to singular values
        13	  - rot_T   : (n_blocks, bs(bs-1)/2)  -- upper-triangle of skew matrix per block
        14	
        15	Forward (matches base layer convention exactly at t=0):
        16	
        17	    R          = block_diag(Cayley(skew(rot_T)))            # (r, r) effective
        18	    Vh_rot     = R @ Vh                                     # rotates input basis
        19	    S_eff      = S + delta_s                                # learnable spectrum
        20	    delta_y    = ((x @ Vh_rot.T) * S_eff) @ U.T             # rank-r path
        21	    base_y     = x @ W_res.T                                # frozen residual
        22	    y_total    = base_y + delta_y                           # == original output at t=0
        23	
        24	At init: rot_T = 0 -> R = I -> Vh_rot = Vh, delta_s = 0 -> S_eff = S, so
        25	delta_y reconstructs the truncated SVD term and y_total == x @ W^T to numerical
        26	precision (fp32 SVD round-tripped to cfg.dtype).
        27	
        28	WHICH BASIS IS ROTATED:
    ... (121 more lines)

[tool] Read eva.py (completed)
  kind: read
  input: {"call_id":"call_mHqxO4d6Ysrff17whtDJcMmv","process_id":"5302","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/eva.py"],"cwd":"/media/wassname/SGIronWolf...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py
  output:
    1	"""EVA: Explained-Variance Adaptation. Paischer et al. 2024.
         2	
         3	Paper: https://arxiv.org/abs/2410.07170 (also referred to as ICLR'25 EVA).
         4	
         5	Idea: instead of random A and zero B (LoRA) or SVD of W (PiSSA), initialize
         6	`lora_A` to the top-r right singular vectors of the LAYER INPUT distribution
         7	on a small calibration set. Forward = `y + scale * (B @ A @ x)` exactly like
         8	LoRA; with `lora_B = 0` the adapter is identity at t=0. Only B trains
         9	afterwards (A frozen). The result: each rank slot points along a direction
        10	that actually carries information at this layer.
        11	
        12	This is a stripped-down EVA; we do NOT implement:
        13	  - rank redistribution across layers via explained-variance ratios
        14	    (peft EVA computes an explained_variance_ratio per layer then redistributes
        15	    the global rank budget; we use a uniform `cfg.r` per layer).
        16	  - Incremental PCA over many micro-batches (we run one full SVD on the
        17	    pooled calibration activations per layer).
        18	  - Equal-input deduplication (peft hashes inputs to share SVD across QKV).
        19	
        20	API stress-test: this variant requires data-driven init, so it implements
        21	`group_init(model, targets, cfg, calibration_data)` to drive a single forward
        22	pass on `calibration_data` with hooks that capture each target's input.
        23	
        24	Identity at t=0: `lora_B = 0` -> delta = 0 -> y unchanged.
        25	
        26	References:
        27	  - peft EVA (full impl, with IncrementalPCA + redistribution):
        28	    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/eva.py
    ... (98 more lines)
The runtime freezes every existing model parameter before installing adapter params, and hook dispatch casts adapter inputs to `cfg.dtype` then casts back to the base output dtype. That cast path is going to matter for every dtype verdict, especially bf16 identity claims.

[tool] Read smoke.py (running)
  input: {"call_id":"call_Vt6FggawETJefJCZf3IhuKcg","process_id":"10277","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba tests/smoke.py"],"cwd":"/media/wassname/SGIronWolf/projects5/202...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py

[tool] Read qwen_train_probe.py (running)
  input: {"call_id":"call_I0OJJCuss3rKB0LfwlTtCiIX","process_id":"80757","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba scripts/qwen_train_probe.py"],"cwd":"/media/wassname/SGIronWolf/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py

[tool] Read smoke.py (completed)
  kind: read
  input: {"call_id":"call_Vt6FggawETJefJCZf3IhuKcg","process_id":"10277","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba tests/smoke.py"],"cwd":"/media/wassname/SGIronWolf/projects5/202...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py
  output:
    1	"""Smoke test: current variants on a tiny synthetic transformer-like model.
         2	
         3	Verifies:
         4	    1. Identity at t=0 (delta ~ 0, output close to base).
         5	    2. Save/load round-trip preserves outputs.
         6	    3. A few SGD steps reduce a random loss (gradients flow).
         7	
         8	Run:
         9	    cd lora-lite
        10	    python -m pip install -e .
        11	    python tests/smoke.py
        12	
        13	BLUF format:
        14	    SHOULD: max|y_adapter - y_base| < tol_init for all variants. ELSE init or hook bug.
        15	    SHOULD: loss decreases > 5% over 20 SGD steps for all variants. ELSE grad/wiring bug.
        16	"""
        17	from __future__ import annotations
        18	import argparse
        19	import os, sys, math
        20	from pathlib import Path
        21	import torch
        22	from torch import nn
        23	
        24	# allow running as `python tests/smoke.py` without install
        25	sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        26	
        27	import lora_lite as ll  # noqa: E402
        28	
    ... (366 more lines)

[tool] Read qwen_train_probe.py (completed)
  kind: read
  input: {"call_id":"call_I0OJJCuss3rKB0LfwlTtCiIX","process_id":"80757","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba scripts/qwen_train_probe.py"],"cwd":"/media/wassname/SGIronWolf/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py
  output:
    1	from __future__ import annotations
         2	
         3	import argparse
         4	import gc
         5	import math
         6	from pathlib import Path
         7	
         8	import torch
         9	from tabulate import tabulate
        10	from transformers import AutoModelForCausalLM, AutoTokenizer
        11	
        12	import lora_lite as ll
        13	
        14	
        15	PROMPT = "LoRA-lite probe: Paris is the capital of France. The answer is"
        16	EXPECTED_TARGETS = {
        17	    "model.layers.0.self_attn.q_proj",
        18	    "model.layers.0.self_attn.v_proj",
        19	}
        20	
        21	
        22	def cfg_for_variant(variant: str, dtype: torch.dtype, r: int, alpha: float) -> ll.LoraLiteConfig:
        23	    return ll.LoraLiteConfig(
        24	        variant=variant,
        25	        r=r,
        26	        alpha=r if variant == "pissa" else alpha,
        27	        dtype=dtype,
        28	        target_roles=(),
    ... (180 more lines)

[tool] Search pissa|PiSSA|reset_lora|kaiming|lora_A|lora_B|scaling|DoRA|Dora|eva|IA3|ia3|hra|HRA|delora|DeLoRA|lambda|weight_norm|svd|Singular|Householder|lora_magnitude_vector|magnitude in peft_lora_layer.py (running)
  input: {"call_id":"call_iAPLKeevRKI2FhcpcZM4ZWfl","process_id":"72662","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","rg -n \"pissa|PiSSA|reset_lora|kaiming|lora_A|lora_B|scaling|DoRA|Dora|...

[tool] Run wc -l docs/refs/*.py (running)
  input: {"call_id":"call_bdEgZthDlRra2IlaA2Igs8Oa","process_id":"80034","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","wc -l docs/refs/*.py"],"cwd":"/media/wassname/SGIronWolf/projects5/2026...

[tool] Search pissa|PiSSA|reset_lora|kaiming|lora_A|lora_B|scaling|DoRA|Dora|eva|IA3|ia3|hra|HRA|delora|DeLoRA|lambda|weight_norm|svd|Singular|Householder|lora_magnitude_vector|magnitude in peft_lora_layer.py (completed)
  kind: search
  input: {"call_id":"call_iAPLKeevRKI2FhcpcZM4ZWfl","process_id":"72662","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","rg -n \"pissa|PiSSA|reset_lora|kaiming|lora_A|lora_B|scaling|DoRA|Dora|...
  output:
    docs/refs/peft_lora_dora.py:28:"""Whether to enable DoRA caching, which makes it faster at inference but requires more memory"""
    docs/refs/peft_lora_dora.py:32:    """Caching decorator for DoRA
    docs/refs/peft_lora_dora.py:34:    Caching is only enabled if ENABLE_DORA_CACHING is set to True (default: False), when in eval mode, and when the
    docs/refs/peft_lora_dora.py:62:class DoraLinearLayer(nn.Module):
    docs/refs/peft_lora_dora.py:69:        # cache intermediate values, e.g. weight norm of DoRA
    docs/refs/peft_lora_dora.py:86:    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None) -> torch.Tensor:
    docs/refs/peft_lora_dora.py:89:        weight = weight + scaling * lora_weight
    docs/refs/peft_lora_dora.py:90:        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
    docs/refs/peft_lora_dora.py:91:        return weight_norm
    docs/refs/peft_lora_dora.py:94:    def get_lora_weight(self, lora_A, lora_B, adapter_name: Optional[str] = None):
    docs/refs/peft_lora_dora.py:95:        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
    docs/refs/peft_lora_dora.py:97:        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=lora_A.weight.dtype)
    docs/refs/peft_lora_dora.py:98:        lora_weight = lora_B(lora_A(x_eye)).T
    docs/refs/peft_lora_dora.py:101:    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
    docs/refs/peft_lora_dora.py:103:        dtype_is_fp16 = lora_A.dtype == torch.float16
    docs/refs/peft_lora_dora.py:105:            lora_A = lora_A.float()
    docs/refs/peft_lora_dora.py:106:            lora_B = lora_B.float()
    docs/refs/peft_lora_dora.py:116:                r = lora_A.shape[0]
    docs/refs/peft_lora_dora.py:117:                lora_weight = torch.mm(lora_B.view([-1, r]), lora_A.view([r, -1]))
    docs/refs/peft_lora_dora.py:120:                lora_weight = lora_B @ lora_A
    docs/refs/peft_lora_dora.py:124...

[tool] Run wc -l docs/refs/*.py (completed)
  kind: execute
  input: {"call_id":"call_bdEgZthDlRra2IlaA2Igs8Oa","process_id":"80034","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","wc -l docs/refs/*.py"],"cwd":"/media/wassname/SGIronWolf/projects5/2026...
  output:
    375 docs/refs/antipasto3_svd_adapter.py
       471 docs/refs/baukit_nethook.py
       446 docs/refs/orig_delora.py
       420 docs/refs/orig_hra_layer.py
        59 docs/refs/orig_pissa_init.py
       274 docs/refs/peft_delora_layer.py
        96 docs/refs/peft_eva_finetuning.py
       739 docs/refs/peft_eva.py
       462 docs/refs/peft_hra_layer.py
       336 docs/refs/peft_ia3_layer.py
       287 docs/refs/peft_lora_dora.py
      2510 docs/refs/peft_lora_layer.py
       923 docs/refs/peft_lora_variants.py
      7398 total

[tool] Read peft_lora_layer.py (running)
  input: {"call_id":"call_6NarexWBfuoCsQBduKpM1wqf","process_id":"78324","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '190,276p' docs/refs/peft_lora_layer.py | nl -ba -v190"],"cwd":"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py

[tool] Read peft_lora_layer.py (running)
  input: {"call_id":"call_wzNKIz9kVoezvm1lbPlep4sO","process_id":"81918","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '360,392p' docs/refs/peft_lora_layer.py | nl -ba -v360"],"cwd":"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py

[tool] Read peft_delora_layer.py (running)
  input: {"call_id":"call_MexmC1hD4dzGon2Fl6m2LTOb","process_id":"2287","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '128,154p' docs/refs/peft_delora_layer.py | nl -ba -v128"],"cwd":...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py

[tool] Read peft_delora_layer.py (running)
  input: {"call_id":"call_FtNtpBysfs7h0C5Qc4QBDv4I","process_id":"38456","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '243,260p' docs/refs/peft_delora_layer.py | nl -ba -v243"],"cwd"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py

[tool] Read peft_lora_layer.py (completed)
  kind: read
  input: {"call_id":"call_6NarexWBfuoCsQBduKpM1wqf","process_id":"78324","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '190,276p' docs/refs/peft_lora_layer.py | nl -ba -v190"],"cwd":"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       190	            lora_dropout_layer = nn.Dropout(p=lora_dropout)
       191	        else:
       192	            lora_dropout_layer = nn.Identity()
       193	
       194	        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
       195	
       196	        # Actual trainable parameters
       197	        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
       198	        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
       199	
       200	        # Tying adapters is only implemented for Linear layers
       201	        # where the source is the embedding layer.
       202	        # Currently, this is the most prevelant way of tying layers (weight tying)
       203	        if tied_adapter:
       204	            lora_A_params = tied_adapter["lora_A"]
       205	            lora_B_params = tied_adapter["lora_B"]
       206	
       207	            self.lora_A[adapter_name].weight = torch.nn.Parameter(lora_A_params)
       208	            self.lora_B[adapter_name].weight = torch.nn.Parameter(lora_B_params)
       209	
       210	        self.lora_bias[adapter_name] = lora_bias
       211	
       212	        if use_rslora:
       213	            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
       214	        else:
       215	            self.scaling[adapter_name] = lora_alpha / r
       216	
    ... (60 more lines)

[tool] Read peft_lora_layer.py (completed)
  kind: read
  input: {"call_id":"call_wzNKIz9kVoezvm1lbPlep4sO","process_id":"81918","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '360,392p' docs/refs/peft_lora_layer.py | nl -ba -v360"],"cwd":"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       360	    def pissa_init(self, adapter_name, init_lora_weights):
       361	        weight = self.get_base_layer().weight
       362	        dtype = weight.dtype
       363	        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
       364	            raise TypeError(
       365	                "Please initialize PiSSA under float32, float16, or bfloat16. "
       366	                "Subsequently, re-quantize the residual model to help minimize quantization errors."
       367	            )
       368	        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
       369	        if init_lora_weights == "pissa":
       370	            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
       371	            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
       372	            Vr = V[:, : self.r[adapter_name]]
       373	            Sr = S[: self.r[adapter_name]]
       374	            Sr /= self.scaling[adapter_name]
       375	            Uhr = Uh[: self.r[adapter_name]]
       376	        elif len(init_lora_weights.split("_niter_")) == 2:
       377	            Vr, Sr, Ur = svd_lowrank(
       378	                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
       379	            )
       380	            Sr /= self.scaling[adapter_name]
       381	            Uhr = Ur.t()
       382	        else:
       383	            raise ValueError(
       384	                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
       385	            )
       386	
    ... (6 more lines)

[tool] Read peft_delora_layer.py (completed)
  kind: read
  input: {"call_id":"call_MexmC1hD4dzGon2Fl6m2LTOb","process_id":"2287","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '128,154p' docs/refs/peft_delora_layer.py | nl -ba -v128"],"cwd":...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       128	    def reset_delora_parameters(
       129	        self,
       130	        adapter_name: str,
       131	        init_weights: bool = True,
       132	        delora_lambda: float = 15.0,
       133	    ) -> None:
       134	        if adapter_name not in self.delora_A.keys():
       135	            return
       136	
       137	        if init_weights is True:
       138	            nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
       139	            nn.init.zeros_(self.delora_B[adapter_name])
       140	        else:
       141	            nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
       142	            nn.init.kaiming_uniform_(self.delora_B[adapter_name], a=math.sqrt(5))
       143	
       144	        self.delora_lambda[adapter_name].data.fill_(float(delora_lambda))
       145	
       146	        # capture a fixed norm for this adapter to use for future delta computations
       147	        with torch.no_grad():
       148	            w = self.get_base_layer().weight
       149	            if w.device.type != "meta":
       150	                w_norm = torch.norm(w.data, dim=0).detach()
       151	            else:
       152	                # For meta tensors, we can't compute the norm, so use a default value
       153	                w_norm = torch.ones(w.shape[1], device=w.device)
       154	            self.delora_w_norm[adapter_name] = w_norm

[tool] Read peft_delora_layer.py (completed)
  kind: read
  input: {"call_id":"call_FtNtpBysfs7h0C5Qc4QBDv4I","process_id":"38456","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '243,260p' docs/refs/peft_delora_layer.py | nl -ba -v243"],"cwd"...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       243	                if adapter not in self.delora_A:
       244	                    continue
       245	
       246	                x_d = self.delora_dropout[adapter](x)
       247	
       248	                # Decomposed delta calculation
       249	                # 1. (x * w_norm) @ A.T
       250	                h = nn.functional.linear(x_d * self.delora_w_norm[adapter], self.delora_A[adapter])
       251	
       252	                # 2. h @ diag
       253	                An = torch.clamp(self.delora_A[adapter].norm(dim=1), min=1e-4)
       254	                Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-4)
       255	                scaling = (self.delora_lambda[adapter] / self.r[adapter]) / (An * Bn)
       256	
       257	                h = h * scaling
       258	
       259	                # 3. h @ B.T
       260	                h = nn.functional.linear(h, self.delora_B[adapter])

[tool] Read peft_ia3_layer.py (running)
  input: {"call_id":"call_YHAJ6VMgq9xcD7wqoh5TnPRh","process_id":"99386","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '56,75p' docs/refs/peft_ia3_layer.py | nl -ba -v56"],"cwd":"/med...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py

[tool] Read peft_lora_dora.py (running)
  input: {"call_id":"call_u9VdPsmOvCaxZc1LPZrvu97d","process_id":"42148","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '86,165p' docs/refs/peft_lora_dora.py | nl -ba -v86"],"cwd":"/me...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_dora.py

[tool] Read peft_ia3_layer.py (running)
  input: {"call_id":"call_81sbHRsSNlWnas3IM8yKCuaQ","process_id":"52784","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '168,188p' docs/refs/peft_ia3_layer.py | nl -ba -v168"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py

[tool] Read peft_hra_layer.py (running)
  input: {"call_id":"call_B0sU5DYQP4aIu7GgcTTaNzy8","process_id":"58196","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '100,108p' docs/refs/peft_hra_layer.py | nl -ba -v100"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py

[tool] Read peft_ia3_layer.py (completed)
  kind: read
  input: {"call_id":"call_YHAJ6VMgq9xcD7wqoh5TnPRh","process_id":"99386","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '56,75p' docs/refs/peft_ia3_layer.py | nl -ba -v56"],"cwd":"/med...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        56	    def update_layer(self, adapter_name: str, config: IA3Config, **kwargs):
        57	        init_ia3_weights = config.init_ia3_weights
        58	        inference_mode = config.inference_mode
        59	
        60	        # This code works for linear layers, override for other layer types
        61	        # Actual trainable parameters
        62	        if self.is_feedforward:
        63	            weight = torch.randn((1, self.in_features))
        64	        else:
        65	            weight = torch.randn((self.out_features, 1))
        66	        self.ia3_l[adapter_name] = nn.Parameter(weight)
        67	        if init_ia3_weights:
        68	            self.reset_ia3_parameters(adapter_name)
        69	        self._move_adapter_to_device_of_base_layer(adapter_name)
        70	        self.set_adapter(self.active_adapters, inference_mode=inference_mode)
        71	
        72	    def reset_ia3_parameters(self, adapter_name):
        73	        if adapter_name in self.ia3_l.keys():
        74	            # initialize learned vector with torch.ones
        75	            nn.init.constant_(self.ia3_l[adapter_name], 1.0)

[tool] Read peft_ia3_layer.py (completed)
  kind: read
  input: {"call_id":"call_81sbHRsSNlWnas3IM8yKCuaQ","process_id":"52784","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '168,188p' docs/refs/peft_ia3_layer.py | nl -ba -v168"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       168	            result = self.base_layer(x, *args, **kwargs)
       169	        else:
       170	            ia3_scaling = 1
       171	            for active_adapter in self.active_adapters:
       172	                if active_adapter not in self.ia3_l.keys():
       173	                    continue
       174	                dtype = self.ia3_l[active_adapter].dtype
       175	                ia3_scaling *= self.ia3_l[active_adapter].flatten()
       176	
       177	            if self.is_feedforward:
       178	                x = x.to(dtype)
       179	                # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
       180	                # e.g. bf16 vs fp32. Is that okay?
       181	                interm = (x * ia3_scaling).to(previous_dtype)
       182	                result = self.base_layer(interm, *args, **kwargs)
       183	            else:
       184	                result = self.base_layer(x, *args, **kwargs)
       185	                result_dtype = result.dtype
       186	                result = (result * ia3_scaling).to(result_dtype)
       187	
       188	        return result

[tool] Read peft_lora_dora.py (completed)
  kind: read
  input: {"call_id":"call_u9VdPsmOvCaxZc1LPZrvu97d","process_id":"42148","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '86,165p' docs/refs/peft_lora_dora.py | nl -ba -v86"],"cwd":"/me...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_dora.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        86	    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None) -> torch.Tensor:
        87	        # calculate L2 norm of weight matrix, column-wise
        88	        weight = transpose(weight, self.fan_in_fan_out)
        89	        weight = weight + scaling * lora_weight
        90	        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        91	        return weight_norm
        92	
        93	    @cache_decorator("lora-weight")
        94	    def get_lora_weight(self, lora_A, lora_B, adapter_name: Optional[str] = None):
        95	        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        96	        # calculate the same but using forward.
        97	        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=lora_A.weight.dtype)
        98	        lora_weight = lora_B(lora_A(x_eye)).T
        99	        return lora_weight
       100	
       101	    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
       102	        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
       103	        dtype_is_fp16 = lora_A.dtype == torch.float16
       104	        if dtype_is_fp16:
       105	            lora_A = lora_A.float()
       106	            lora_B = lora_B.float()
       107	
       108	        with gather_params_ctx(base_layer.parameters()):
       109	            if base_layer.__class__.__name__ == "Linear4bit":
       110	                # We have to create a copy of the base layer, otherwise, FSDP will throw an error. 8bit does not work
       111	                # yet because Int8Params cannot be correctly deep-copied (attributes vanish)
       112	                base_layer = deepcopy(base_layer)
    ... (53 more lines)

[tool] Read peft_hra_layer.py (completed)
  kind: read
  input: {"call_id":"call_B0sU5DYQP4aIu7GgcTTaNzy8","process_id":"58196","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '100,108p' docs/refs/peft_hra_layer.py | nl -ba -v100"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       100	    def reset_hra_parameters(self, adapter_name: str):
       101	        if self.hra_r[adapter_name] % 2 != 0:
       102	            warnings.warn("The symmetric initialization can NOT be performed when r is odd!")
       103	            nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))
       104	        else:
       105	            shape = self.hra_u[adapter_name].shape
       106	            half_u = torch.zeros(shape[0], shape[1] // 2)
       107	            nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
       108	            self.hra_u[adapter_name] = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))

[tool] Read peft_eva.py (running)
  input: {"call_id":"call_690oIE1mGi9p5KwBfYRgFCdI","process_id":"47440","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '466,533p' docs/refs/peft_eva.py | nl -ba -v466"],"cwd":"/media/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py

[tool] Read peft_hra_layer.py (running)
  input: {"call_id":"call_UI8XKTEBvQpPGk7VSsCZvLFm","process_id":"20694","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '209,236p' docs/refs/peft_hra_layer.py | nl -ba -v209"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_JCuFK2l4lqHJxxajl0MKxKM4","process_id":"80090","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '35,65p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v35"],"cw...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_Hc7YfPlC2HNtnJxmPJFafOs4","process_id":"81194","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '117,158p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v117"],...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read peft_hra_layer.py (completed)
  kind: read
  input: {"call_id":"call_UI8XKTEBvQpPGk7VSsCZvLFm","process_id":"20694","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '209,236p' docs/refs/peft_hra_layer.py | nl -ba -v209"],"cwd":"/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       209	        rank = self.hra_r[adapter_name]
       210	        apply_GS = self.hra_apply_GS[adapter_name]
       211	        opt_u = self.hra_u[adapter_name]
       212	        shape = opt_u.shape
       213	
       214	        if apply_GS:
       215	            weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
       216	            for i in range(1, rank):
       217	                ui = opt_u[:, i].view(-1, 1)
       218	                for j in range(i):
       219	                    ui = ui - (weight[j].t() @ ui) * weight[j]
       220	                weight.append((ui / ui.norm()).view(-1, 1))
       221	            weight = torch.cat(weight, dim=1)
       222	            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()
       223	
       224	        else:
       225	            opt_u = opt_u / opt_u.norm(dim=0)
       226	            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
       227	            if reverse:
       228	                indices = range(rank - 1, -1, -1)
       229	            else:
       230	                indices = range(rank)
       231	
       232	            for i in indices:
       233	                ui = opt_u[:, i].view(-1, 1)
       234	                weight = weight - 2 * weight @ ui @ ui.t()
       235	
    ... (1 more lines)

[tool] Read peft_eva.py (completed)
  kind: read
  input: {"call_id":"call_690oIE1mGi9p5KwBfYRgFCdI","process_id":"47440","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '466,533p' docs/refs/peft_eva.py | nl -ba -v466"],"cwd":"/media/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       466	    eva_state_dict = {}
       467	    for name, rank in rank_dist.items():
       468	        hook = hooks[layer_hook_map[name]][0]
       469	        if not torch.all(hook.converged[:rank]):
       470	            raise ValueError(
       471	                f"Layer {name} has not converged but was assigned rank {rank}. "
       472	                "Please report this issue at https://github.com/huggingface/peft/issues"
       473	            )
       474	        u = hook.svd.components_[:rank]
       475	        if peft_config.eva_config.whiten:
       476	            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
       477	        eva_state_dict[name] = u
       478	
       479	    # restore model state
       480	    model.train(training)
       481	
       482	    # move tensors to device
       483	    if device is not None:
       484	        eva_state_dict = {k: v.to(device) for k, v in eva_state_dict.items()}
       485	
       486	    return eva_state_dict
       487	
       488	
       489	def _load_eva_state_dict(
       490	    model: torch.nn.Module,
       491	    eva_state_dict: dict,
       492	    adapter_name: str,
    ... (41 more lines)

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_JCuFK2l4lqHJxxajl0MKxKM4","process_id":"80090","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '35,65p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v35"],"cw...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        35	class DeltaSParam(SVDParam):
        36	    """Trainable delta_s scaling parameters (full LR)."""
        37	    pass
        38	
        39	
        40	class RotationParam(SVDParam):
        41	    """Block-diagonal rotation parameters (lower LR via rotation_lr_scale config)."""
        42	    pass
        43	
        44	
        45	class SVDFrozen(nnx.Variable):
        46	    """Frozen SVD component. Not differentiated."""
        47	    pass
        48	
        49	
        50	# -- SVD Steering Linear (replaces nnx.Linear) ------------------------------
        51	
        52	class SVDSteeringLinear(nnx.Module):
        53	    """SVD steering adapter replacing a linear layer.
        54	
        55	    y = ((x @ U_rot) * S_scaled) @ Vh + x @ W_res
        56	
        57	    where U_rot, S_scaled depend on alpha (the steering coefficient).
        58	    Frozen: U, S, Vh, W_res.  Learnable: delta_s, rotation_params.
        59	    """
        60	
        61	    def __init__(
    ... (4 more lines)

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_Hc7YfPlC2HNtnJxmPJFafOs4","process_id":"81194","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '117,158p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v117"],...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       117	        U = self.svd_U.value
       118	        S = self.svd_S.value
       119	        Vh = self.svd_Vh.value
       120	        W_res = self.svd_W_res.value
       121	        params = self.rotation_params.value  # (n_blocks, n_triu)
       122	        bs = self.block_size
       123	        n_blocks = params.shape[0]
       124	
       125	        # Reconstruct skew-symmetric from upper-triangle params (like OFT/PSOFT).
       126	        # 0.5 factor matches BOFT convention: cancels the 2x gradient from A - A^T.
       127	        A = jnp.zeros((n_blocks, bs, bs), dtype=jnp.float32)
       128	        A = A.at[:, self._triu_rows, self._triu_cols].set(params.astype(jnp.float32))
       129	        A = 0.5 * (A - jnp.swapaxes(A, -1, -2))
       130	
       131	        # Angle clamping (element-wise tanh, bounds bidirectional symmetry error)
       132	        a_limit = 2 * math.tan(self.max_angle / 2)
       133	        A = a_limit * jnp.tanh(A / a_limit)
       134	
       135	        # Cayley transform in float32: R = (I - X)^{-1}(I + X)
       136	        eye = jnp.eye(bs, dtype=jnp.float32)
       137	        X = alpha * A / 2
       138	        R_blocks = jnp.linalg.solve(
       139	            eye[None] - X,
       140	            eye[None] + X,
       141	        )
       142	
       143	        # Apply rotation to U (input singular vectors)
    ... (15 more lines)

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_VyW5j2uOyRzOTWVgvcecQYHo","process_id":"51183","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '158,205p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v158"],...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_VyW5j2uOyRzOTWVgvcecQYHo","process_id":"51183","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '158,205p' docs/refs/antipasto3_svd_adapter.py | nl -ba -v158"],...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
       158	            Vh_rot = Vh
       159	
       160	        S_scaled = S + alpha * self.delta_s.value if self.use_delta_s else S
       161	
       162	        dt = x.dtype
       163	        out = (x @ U_rot.astype(dt)) * S_scaled.astype(dt)
       164	        out = out @ Vh_rot.astype(dt)
       165	        out = out + x @ W_res.astype(dt)
       166	        return out
       167	
       168	
       169	def create_svd_adapter(
       170	    kernel: jax.Array,
       171	    r: int,
       172	    rotation_block_size: int,
       173	    max_rotation_angle: float,
       174	    rngs: nnx.Rngs,
       175	    selected_indices: jax.Array | None = None,
       176	    rotate_U: bool = True,
       177	    rotate_V: bool = False,
       178	    use_delta_s: bool = True,
       179	) -> SVDSteeringLinear:
       180	    """Create SVD adapter from a kernel matrix (in_dim, out_dim).
       181	
       182	    If selected_indices is provided, uses those SVD dimensions.
       183	    Otherwise uses top-r by singular value.
       184	    """
    ... (21 more lines)

[tool] Searching the Web (running)

[tool] Opening: https://arxiv.org/abs/2106.09685 (completed)
  kind: fetch
  input: https://arxiv.org/abs/2106.09685

[tool] Searching the Web (running)

[tool] Opening: https://arxiv.org/abs/2503.18225 (completed)
  kind: fetch
  input: https://arxiv.org/abs/2503.18225

[tool] Searching the Web (running)

[tool] Opening: https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py (completed)
  kind: fetch
  input: https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py

[tool] Searching the Web (running)

[tool] Opening: https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py (completed)
  kind: fetch
  input: https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py

[tool] Searching the Web (running)

[tool] Opening: https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py (completed)
  kind: fetch
  input: https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py

[tool] Searching the Web (running)

[tool] Opening: https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/hra/layer.py (completed)
  kind: fetch
  input: https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/hra/layer.py

[tool] Searching the Web (running)

[plan]
  - [completed] Inventory source, refs, tests
  - [in_progress] Verify external links and snapshot drift
  - [pending] Audit each variant against refs
  - [pending] Assess smoke and Qwen probe validity
  - [pending] Assemble concise findings with line citations
The external links are live for the paper and GitHub references I checked; AntiPaSTO is the standout because the docstring names only a GitHub research repo and no paper, which fails the reference standard in your prompt. I’m doing the math pass now, especially around PiSSA scaling, HRA orientation, and AntiPaSTO basis naming.

[tool] Searching for: site:github.com/huggingface/peft src/peft/tuners/delora/layer.py PEFT DeLoRA layer.py, github huggingface peft delora layer.py (completed)
  kind: fetch
  input: site:github.com/huggingface/peft src/peft/tuners/delora/layer.py PEFT DeLoRA layer.py

[tool] Read variants_review_v4.md (running)
  input: {"call_id":"call_jJh1eAt0PAr2or6I9kT26Nge","process_id":"63414","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba docs/audit/variants_review_v4.md"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/audit/variants_review_v4.md

[tool] Read REVIEW_PROMPT_VARIANTS_V4.md (running)
  input: {"call_id":"call_kWky1g3kEyZFT5QS99mtsU42","process_id":"76948","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba docs/audit/REVIEW_PROMPT_VARIANTS_V4.md"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/audit/REVIEW_PROMPT_VARIANTS_V4.md

[tool] Read variants_review_v4.md (completed)
  kind: read
  input: {"call_id":"call_jJh1eAt0PAr2or6I9kT26Nge","process_id":"63414","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba docs/audit/variants_review_v4.md"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/audit/variants_review_v4.md
  output:
    1	[client] initialize (running)
         2	
         3	[client] session/new (running)
         4	I’ll use the `gsd` skill here because this is a multi-part audit: first I’ll read the local implementation and references, then verify URLs/snapshots, then review the two harnesses and collapse it into the requested verdict tables.
         5	
         6	[tool] Read SKILL.md (running)
         7	  input: {"call_id":"call_hMhlWCMXk8cPaOEh4mCYl3wE","process_id":"55868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
         8	  files: /home/wassname/.claude/skills/gsd/SKILL.md
         9	
        10	[tool] Read SKILL.md (completed)
        11	  kind: read
        12	  input: {"call_id":"call_hMhlWCMXk8cPaOEh4mCYl3wE","process_id":"55868","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
        13	  files: /home/wassname/.claude/skills/gsd/SKILL.md
        14	  output:
        15	    ---
        16	    name: gsd
        17	    description: "Get Shit Done workflow for non-trivial tasks: spec -> implement -> test -> review -> wrap. Mandatory for anything with more than one step."
        18	    ---
        19	    
        20	    <!--
        21	    Design rationale and sources — for human readers, not the agent.
        22	    
        23	    - GSD spec-first: spec-driven development, scope fencing, verify-before-done
        24	    - Manus file-as-memory: one file is working memory, checklist is state, append-only log
        25	    - Boomerang: return concrete artifacts not summaries
        26	    - DCP: invisible dedup, superseded writes, error purging
        27	    
        28	    Core principles:
    ... (1214 more lines)

[tool] Read REVIEW_PROMPT_VARIANTS_V4.md (completed)
  kind: read
  input: {"call_id":"call_kWky1g3kEyZFT5QS99mtsU42","process_id":"76948","turn_id":"019dc9b0-4472-7811-b19f-df7e11d4e3b9","command":["/usr/bin/zsh","-lc","nl -ba docs/audit/REVIEW_PROMPT_VARIANTS_V4.md"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/audit/REVIEW_PROMPT_VARIANTS_V4.md
  output:
    1	# V4 Variant Review — per-component vs reference + smoke/probe validity
         2	
         3	You are an expert ML engineer reviewing a from-scratch PEFT library
         4	(`lora-lite`, ~500 LOC) that re-implements 8 LoRA variants. Three prior
         5	reviews already happened (V1 paper-vs-code, V2 with refs provided, V3
         6	per-component). Your job is V4: re-run the per-component check and
         7	additionally validate the test harness.
         8	
         9	# Part A — per-variant audit (re-do, more rigorous)
        10	
        11	8 variants live in `src/lora_lite/variants/`:
        12	- lora.py
        13	- pissa.py
        14	- delora.py
        15	- ia3.py (registers `ia3` and `ia3_ff`)
        16	- dora.py
        17	- hra.py
        18	- eva.py
        19	- antipasto.py
        20	
        21	Plus runtime in `src/lora_lite/{adapter.py,variant.py,target.py,config.py}`.
        22	Reference implementations are in `docs/refs/` and URLs are pasted in each
        23	variant's module docstring.
        24	
        25	## For EACH variant, in this order, every time:
        26	
        27	1. **REFERENCE EXISTS** — verify the variant has a real, citeable
        28	   reference. Required:
    ... (158 more lines)

[tool] Searching the Web (running)

[tool] Open page (completed)
  kind: fetch
  input: {"query":"","action":{"type":"open_page"}}

[tool] Searching the Web (running)

[plan]
  - [completed] Inventory source, refs, tests
  - [completed] Verify external links and snapshot drift
  - [completed] Audit each variant against refs
  - [completed] Assess smoke and Qwen probe validity
  - [in_progress] Assemble concise findings with line citations
I’ve got the core review shape. The main correctness blockers are not evenly distributed: most variants are fine or intentionally simplified, but DeLoRA init, EVA’s frozen-A choice, HRA pre-hook order, and AntiPaSTO’s missing paper/reference status are the ones that change the conclusion.## lora

### references
- paper: https://arxiv.org/abs/2106.09685 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py -- OK
- author ref: MISSING in docstring, but PEFT ref satisfies prompt
- offline snapshot (`docs/refs/peft_lora_layer.py`): MATCH by line count/key snippets
- VERDICT: HAS_REFERENCE

### params
- `lora_A`: `(r, d_in)`, trainable Parameter, `cfg.dtype`; matches PEFT A weight shape, [src/lora_lite/variants/lora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py:27).
- `lora_B`: `(d_out, r)`, trainable Parameter, zero-init; matches PEFT B zero-init, [lora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py:28).

### init / group_init
- `init()` no-op; identity comes from `B=0`, [lora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py:31).
- GRADIENT FLOW at t=0: `lora_B` gets non-zero grad; `lora_A` gets zero grad until `B` moves.

### dtype
- Params allocated in `cfg.dtype`; hook casts `x` to `cfg.dtype` and output back to base output dtype, [adapter.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py:15). Identity survives bf16 because delta is exactly zero.

### forward
Math (ours): `y + (alpha/r) * x A^T B^T`, [lora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py:42)  
Math (ref): `base(x) + lora_B(lora_A(dropout(x))) * alpha/r`  
Match? YES, with documented no-dropout design.

### verdict
CORRECT -- faithful vanilla LoRA.

## pissa

### references
- paper: https://arxiv.org/abs/2404.02948 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py -- OK
- author ref: https://github.com/MuLabPKU/PiSSA/blob/main/utils/init_pissa.py -- OK
- offline snapshot (`docs/refs/orig_pissa_init.py`, `docs/refs/peft_lora_layer.py`): MATCH by key snippets
- VERDICT: HAS_REFERENCE

### params
- `lora_A`: `(r, d_in)`, trainable, zero placeholder then SVD-filled, [pissa.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py:38).
- `lora_B`: `(d_out, r)`, trainable, zero placeholder then SVD-filled, [pissa.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py:39).

### init / group_init
- SVDs `W.float()`, sets `B=U sqrt(S)`, `A=sqrt(S) Vh`, then mutates base weight to `W - scale*BA`, [pissa.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py:49).
- PEFT divides singular values by scaling before forming A/B; ours does not, so param magnitudes differ when `alpha != r` even though t=0 effective weight reconstructs.
- GRADIENT FLOW at t=0: both `A` and `B` are non-zero, so both get gradients.

### dtype
- SVD in fp32, A/B stored in `cfg.dtype`, residual copied to base dtype, [pissa.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py:55). bf16 identity is approximate, not bit-exact.

### forward
Math (ours): `(W - s BA)x + s BAx`, `s=alpha/r`, [pissa.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py:63)  
Math (ref): same effective formula, but `BA` initialized as top-r component divided by `s` in PEFT.  
Match? PARTIAL -- effective t=0 matches; parameterization differs for `alpha != r`.

### verdict
PARTIAL -- correct only under the documented `alpha=r` convention; otherwise PEFT-equivalent init scaling is off.

## delora

### references
- paper: https://arxiv.org/abs/2503.18225 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py -- AMBIGUOUS in web fetch, but HF docs for DeLoRA are live
- author ref: https://github.com/ExplainableML/DeLoRA -- OK
- offline snapshot (`docs/refs/orig_delora.py`, `docs/refs/peft_delora_layer.py`): MATCH to local expected snippets; live exact-file drift not fully proven
- VERDICT: HAS_REFERENCE

### params
- `lora_A`: `(r, d_in)`, trainable, kaiming; OK, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:50).
- `lora_B`: `(d_out, r)`, trainable, kaiming; BUG vs PEFT default, which zero-inits B when `init_weights=True` (`docs/refs/peft_delora_layer.py:137-140`), [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:51).
- `lora_lambda`: scalar trainable; OK, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:52).
- `lora_wnorm`: `(d_in,)` persistent buffer; matches PEFT per-input norm, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:57).

### init / group_init
- Captures `||W||` over output dim into buffer, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:65).
- With `lambda0=0`, identity but A/B dead on step 1; this is documented, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:20).
- With `lambda0>0`, because B is kaiming, not zero/frozen-copy compensated, t=0 is not identity.

### dtype
- W norm captured after fp32 read but stored in `cfg.dtype`, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:66). bf16 can quantize the scaling vector.

### forward
Math (ours): `y + ((x*wnorm)A^T * (lambda/r)/(||A_i||||B_i||))B^T`, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:82)  
Math (ref): same PEFT forward, with optional dropout before A.  
Match? YES forward formula; NO init.

### verdict
BUGGY -- forward matches PEFT, but B init makes nonzero-`lambda` initialization non-identity and mismatches PEFT default.

## ia3 / ia3_ff

### references
- paper: https://arxiv.org/abs/2205.05638 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py -- OK
- author ref: MISSING in docstring, PEFT ref satisfies prompt
- offline snapshot (`docs/refs/peft_ia3_layer.py`): MATCH
- VERDICT: HAS_REFERENCE

### params
- `ia3`: `lora_g` `(d_out,)`, trainable ones; matches output-side PEFT after flattening, [ia3.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py:39).
- `ia3_ff`: `lora_g` `(d_in,)`, trainable ones; matches feedforward input-side gate, [ia3.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py:60).

### init / group_init
- No-op; identity from ones, [ia3.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py:42).
- GRADIENT FLOW at t=0: `lora_g` receives non-zero grad if the gated activation/output participates in loss.

### dtype
- Gate stored in `cfg.dtype`; pre/post hook casts around base dtype, [adapter.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py:23). Ones survive bf16 exactly.

### forward
Math (ours): `ia3: y*g`; `ia3_ff: W(x*g)`, [ia3.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py:52)  
Math (ref): same split controlled by `is_feedforward`.  
Match? YES.

### verdict
CORRECT -- faithful, with the documented need for two attach passes to match paper placement.

## dora

### references
- paper: https://arxiv.org/abs/2402.09353 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py -- OK
- author ref: MISSING in docstring, PEFT ref satisfies prompt
- offline snapshot (`docs/refs/peft_lora_dora.py`): MATCH
- VERDICT: HAS_REFERENCE

### params
- `lora_A`: `(r, d_in)`, trainable kaiming; OK, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:36).
- `lora_B`: `(d_out, r)`, trainable zero; OK, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:37).
- `lora_m`: `(d_out,)`, trainable magnitude vector; OK, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:39).

### init / group_init
- Requires exact `nn.Linear`; fills `m=||W||` per output row, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:43).
- GRADIENT FLOW at t=0: `m` and `B` get grad; `A` initially zero-grad via `B=0`.

### dtype
- Norm computed fp32 then stored in `cfg.dtype`, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:50). bf16 identity is approximate due norm/division.

### forward
Math (ours): `bias + (m/||W+sBA||) * (Wx + sBAx)`, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:63)  
Math (ref): same forward; PEFT detaches norm in backward.  
Match? YES for forward; backward deviation documented, [dora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py:10).

### verdict
CORRECT -- forward faithful; documented backward-cost deviation.

## hra

### references
- paper: https://arxiv.org/abs/2405.17484 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/layer.py -- OK
- author ref: https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer_GS_HRA.py -- OK
- offline snapshot (`docs/refs/orig_hra_layer.py`, `docs/refs/peft_hra_layer.py`): MATCH
- VERDICT: HAS_REFERENCE

### params
- `lora_U`: `(r, d_in)`, trainable; transposed relative to PEFT `(d_in, r)` but equivalent storage, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:53).

### init / group_init
- Rejects odd rank instead of PEFT warning+random init; stricter but documented, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:45).
- Paired kaiming rows cancel at init, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:62).
- GRADIENT FLOW at t=0: U receives gradients despite identity.

### dtype
- U in `cfg.dtype`; repeated Householder products in that dtype, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:75). bf16 may accumulate orthogonality/identity error after updates.

### forward
Math (ours): prehook applies `x H_0 H_1 ... H_{r-1}`, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:77)  
Math (ref): PEFT builds `R=H_0...H_{r-1}`, then uses weight `W R`, so row-input equivalent is `x R^T W^T` (`docs/refs/peft_hra_layer.py:224-235`).  
Match? NO -- after paired init breaks, ours applies `R` where PEFT/paper row convention needs `R^T` or reversed Householder order.

### verdict
BUGGY -- identity smoke hides a forward-orientation/order mismatch after training.

## eva

### references
- paper: https://arxiv.org/abs/2410.07170 -- OK
- peft ref: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/eva.py -- OK
- author ref: MISSING in docstring, PEFT ref satisfies prompt
- offline snapshot (`docs/refs/peft_eva.py`, `docs/refs/peft_eva_finetuning.py`): MATCH
- VERDICT: HAS_REFERENCE

### params
- `lora_A`: `(r, d_in)`, frozen buffer; BUG vs PEFT LoRA layer, where `lora_A` remains a trainable Linear weight and EVA only initializes it, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:51).
- `lora_B`: `(d_out, r)`, trainable zero; OK, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:53).

### init / group_init
- Captures target inputs via pre-hooks and one full SVD per target, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:62).
- Omits redistribution/incremental PCA/equal-input dedup; documented, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:12).
- GRADIENT FLOW at t=0: only `B` gets grad because `A` is a buffer.

### dtype
- SVD in fp32, A stored in `cfg.dtype`, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:112). B=0 gives exact identity.

### forward
Math (ours): `y + (alpha/r) * x A^T B^T`, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:122)  
Math (ref): same LoRA forward after EVA initializes A, but A remains a LoRA parameter in PEFT.  
Match? PARTIAL -- forward formula matches; trainability/buffer semantics do not.

### verdict
PARTIAL/BUGGY -- data init is recognizable, but freezing A is not PEFT-equivalent.

## antipasto

### references
- paper: MISSING -- no arXiv/conference link in docstring, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:1)
- peft ref: MISSING
- author ref: https://github.com/wassname/antipasto3 -- OK repo, not a citeable paper
- offline snapshot (`docs/refs/antipasto3_svd_adapter.py`): MATCH to local research-code snapshot
- VERDICT: NO_REFERENCE

### params
- `lora_U/S/Vh`: frozen persistent buffers; align with frozen SVD components, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:90).
- `lora_delta_s`: `(r,)` trainable; OK vs snapshot, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:94).
- `lora_rot_T`: `(n_blocks, bs(bs-1)/2)` trainable; OK, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:95).

### init / group_init
- SVDs `W`, stores top-r factors, mutates base weight to residual, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:105).
- GRADIENT FLOW at t=0: `delta_s` gets grad; `rot_T` should get grad through Cayley unless symmetry/task cancels.

### dtype
- SVD buffers stored in `cfg.dtype`; rotation built in fp32 then cast to input dtype, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:129). bf16 reconstruction approximate.

### forward
Math (ours): `y_res + (x Vh_eff^T) * (S+delta_s) @ U_eff^T`, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:145)  
Math (ref): JAX snapshot equivalent under PyTorch weight transpose; default rotates input singular basis.  
Match? YES to local research snapshot, but no paper/upstream validation target.

### verdict
PARTIAL -- implementation matches local snapshot, but prompt requires `NO_REFERENCE`, severity HIGH.

## smoke.py validity

### per-variant SHOULD checks

| check | distinguishes silent failure? | tolerance ok? | notes |
|---|---:|---:|---|
| t=0 identity, [tests/smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:121) | NO | mostly OK fp32 | A forward returning `y` unchanged passes. Stronger: perturb every trainable param and require nonzero, variant-specific output delta. |
| save/load output, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:140) | WEAK | same as identity | Empty/ignored adapter can pass if both sides are identity; state-key checks help load, but not semantic use. |
| 20-step loss drop, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:159) | PARTIAL | N/A | Catches some dead adapters, but can pass by any one param moving; does not verify each ParamSpec receives grad. |
| EVA A populated, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:307) | YES for group_init | OK | Does not check A equals top-r PCA directions or frozen-vs-trainable semantics. |
| structural linear-like, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:195) | PARTIAL | exact | Only LoRA and only B grad. |
| bnb CUDA, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:216) | PARTIAL | loose `1e-2` | Claims DeLoRA bnb OK, but HF DeLoRA docs say quantized layers unsupported; also omits EVA/AntiPaSTO and marks only PiSSA/DoRA fail, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:241). |

### gaps
- `ia3_ff` is not covered by main variant loop; only `ia3` output gating is tested, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:384).
- No bf16/fp16 main smoke despite dtype concerns; main calls only float32, [smoke.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py:384).
- Does not test per-parameter first-step grad expectations.
- Does not test HRA post-update against explicit `W @ R` reference, so the order bug passes.
- Does not test EVA `len(calibration_tokens) < r` except code path exists, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:107).
- Does not test mixed variants, detach/reattach conflicts, dtype mismatch attach/load, or loading onto non-identical base except fingerprint only.

### must-add tests
- Per variant: perturb each trainable param/buffer role expected to affect forward; require output delta.
- Per variant: assert grad nonzero/zero for each ParamSpec on step 1 according to expected gradient flow.
- HRA: compare post-random-U output to explicit PEFT-style `F.linear(x, W @ R)`.
- EVA: check A trainability decision against chosen reference and test `<r` calibration failure.
- Save/load: corrupt or omit one lora key and require failure plus semantic output equality after nonzero perturbation.

## qwen_train_probe.py validity

### claim-by-claim

| assertion | catches silent failure? | notes |
|---|---:|---|
| exact q/v layer-0 targets, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:88) | YES narrow | Good for regex path, but only two attention layers. |
| `assert_only_lora_trainable`, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:39) | YES | Since `attach()` freezes all base params before installing adapter params, [adapter.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py:39), this catches leaked `requires_grad=True` base params. |
| perturb one adapter param, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:50) | NO | It misses `lora_U`, `lora_delta_s`, `lora_rot_T`; default `hra` will raise no perturbable param, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:181). One-param perturb also does not prove all params are used. |
| finite/nonzero grad + adapter delta, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:117) | PARTIAL | Sum can be nonzero while some required params are dead. |
| `loss_last < loss0`, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:132) | WEAK | Same prompt/labels, 8 steps, high lr: can pass from overfitting/noisy optimizer motion. Need held-out prompt/token loss and require train improves more than held-out degradation threshold. |
| saved state keys equal, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:140) | YES for empty/missing state | Good key-level check. |
| tensor equality after load, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:151) | YES for state restore | Stronger than reload logits. |
| reload logits `<2e-2`, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:155) | WEAK | bf16 `2e-2` can mask wrong small deltas, especially if adapter effect is itself small. Compare adapter-vs-base delta preservation, not absolute logits only. |

### gaps
- Default variants exclude EVA and AntiPaSTO, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:181).
- HRA default is broken by `perturb_first_adapter` not handling `lora_U`.
- Only layer 0 q/v: no multi-layer, FFN, `ia3_ff`, `lm_head` exclusion edge, or target role classification coverage.
- No per-param gradient table.
- No held-out validation or generation behavior check.

### must-add tests
- Extend perturb to every variant param family and require per-param output influence where expected.
- Add held-out prompt loss before/after; require train loss drop plus no reload mismatch on adapter delta.
- Add target suites: multi-layer q/k/v/o, FFN down for `ia3_ff`, excluded `lm_head`, and at least one no-match failure.
- Include EVA with calibration and AntiPaSTO, or explicitly mark them out-of-scope.

## summary

### variant verdicts

| variant | has_ref | params | init | dtype | forward | verdict |
|---|---|---|---|---|---|---|
| lora | yes | OK | OK | OK | OK | CORRECT |
| pissa | yes | OK | PARTIAL | approximate bf16 | PARTIAL | PARTIAL |
| delora | yes | BUG | BUG | OK-ish | OK | BUGGY |
| ia3 / ia3_ff | yes | OK | OK | OK | OK | CORRECT |
| dora | yes | OK | OK | approximate bf16 | OK | CORRECT |
| hra | yes | OK | OK | OK-ish | BUG | BUGGY |
| eva | yes | BUG | PARTIAL | OK | PARTIAL | PARTIAL/BUGGY |
| antipasto | no | OK vs snapshot | OK vs snapshot | approximate bf16 | OK vs snapshot | PARTIAL, NO_REFERENCE |

### MUST-FIX (severity HIGH, blocks correctness claim)
1. AntiPaSTO has no citeable paper reference, so per prompt it is `NO_REFERENCE`, [antipasto.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py:1).
2. HRA pre-hook applies Householder product in the wrong row-vector order versus PEFT `W @ R`, [hra.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py:77).
3. DeLoRA `lora_B` kaiming init mismatches PEFT default zero-B and breaks identity for nonzero `lambda0`, [delora.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py:51).
4. EVA freezes `lora_A` as a buffer, unlike PEFT’s trainable LoRA A after EVA initialization, [eva.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py:51).
5. `qwen_train_probe.py` default includes HRA but cannot perturb HRA params, so the probe is not valid as written, [qwen_train_probe.py](/media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py:50).

### NICE-TO-HAVE
- Make PiSSA PEFT-equivalent for `alpha != r` or document that only effective t=0 reconstruction, not parameterization, is matched.
- Add bf16/fp16 main smoke, per-param grad checks, and stronger non-identity semantic save/load tests.
- Align bnb smoke claims with reference support, especially DeLoRA quantized-layer support.

[tool] Searching for: github.com/ExplainableML/DeLoRA peft/src/peft/tuners/delora.py, github.com/huggingface/peft src/peft/tuners/delora/layer.py (completed)
  kind: fetch
  input: github.com/ExplainableML/DeLoRA peft/src/peft/tuners/delora.py

[done] end_turn
