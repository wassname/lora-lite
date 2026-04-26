[client] initialize (running)

[client] session/new (running)
I’ll use the `gsd` workflow because this is a multi-step audit, plus the repo’s local AGENTS instructions you pasted. I’m going to read the variant/runtime files and the relevant skill instructions first, then compare each module against its local refs and live links.

[tool] Read SKILL.md (running)
  input: {"call_id":"call_wNiPFhdnB1WBW8omQwnkCZZz","process_id":"49611","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
  files: /home/wassname/.claude/skills/gsd/SKILL.md

[tool] Run git status --short (running)
  input: {"call_id":"call_Y2GWC695I6QVJq1SlrwGzpjU","process_id":"56247","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","git status --short"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/l...

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (running)
  input: {"call_id":"call_qs5ejtYjuQgiZd7VjFAvs439","process_id":"70787","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite tests docs/refs"],"cwd":"/media/wassname/SGIro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite

[tool] Read SKILL.md (completed)
  kind: read
  input: {"call_id":"call_wNiPFhdnB1WBW8omQwnkCZZz","process_id":"49611","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/me...
  files: /home/wassname/.claude/skills/gsd/SKILL.md
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
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

[tool] Run git status --short (completed)
  kind: execute
  input: {"call_id":"call_Y2GWC695I6QVJq1SlrwGzpjU","process_id":"56247","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","git status --short"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/l...
  output:
    ?? docs/audit/REVIEW_PROMPT_VARIANTS_V3.md
    ?? docs/audit/variants_review_v3.md

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (completed)
  kind: search
  input: {"call_id":"call_qs5ejtYjuQgiZd7VjFAvs439","process_id":"70787","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite tests docs/refs"],"cwd":"/media/wassname/SGIro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite
  output:
    src/lora_lite/__init__.py
    src/lora_lite/variant.py
    src/lora_lite/config.py
    src/lora_lite/adapter.py
    tests/test_lora_lite.py
    src/lora_lite/variants/ia3.py
    tests/smoke.py
    src/lora_lite/variants/lora.py
    src/lora_lite/variants/dora.py
    src/lora_lite/variants/delora.py
    src/lora_lite/variants/__init__.py
    src/lora_lite/variants/hra.py
    src/lora_lite/variants/pissa.py
    src/lora_lite/variants/eva.py
    src/lora_lite/variants/antipasto.py
    src/lora_lite/target.py
    docs/refs/peft_lora_dora.py
    docs/refs/orig_pissa_init.py
    docs/refs/baukit_nethook.py
    docs/refs/peft_eva.py
    docs/refs/peft_lora_layer.py
    docs/refs/peft_delora_layer.py
    docs/refs/peft_lora_variants.py
    docs/refs/orig_delora.py
    docs/refs/antipasto3_svd_adapter.py
    docs/refs/peft_hra_layer.py
    docs/refs/orig_hra_layer.py
    docs/refs/peft_ia3_layer.py
    ... (17 more lines)
I found two untracked audit files already present, so I’ll leave them untouched and treat the repo as read-only for this review. Next I’m pulling numbered source listings and the offline refs so the final can cite exact lines instead of hand-waving.

[tool] Read lora.py (running)
  input: {"call_id":"call_gMlEVjLyAI4tEO17fDRNFYdy","process_id":"32210","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py

[tool] Read lora.py (completed)
  kind: read
  input: {"call_id":"call_gMlEVjLyAI4tEO17fDRNFYdy","process_id":"32210","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
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
        13	from torch import nn
        14	import torch
        15	
        16	from ..variant import register, ParamSpec
        17	
        18	
        19	@register
        20	class LoRA:
        21	    name = "lora"
        22	
        23	    @staticmethod
        24	    def param_specs(d_in, d_out, cfg):
        25	        return {
        26	            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
        27	            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
        28	        }
    ... (13 more lines)

[tool] Read pissa.py (running)
  input: {"call_id":"call_DJiZUptMa5DJfJiqjcVAbgMf","process_id":"25734","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py

[tool] Read delora.py (running)
  input: {"call_id":"call_yYuKd7kdZxGCZo8zkQtSGEiW","process_id":"76438","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py

[tool] Read ia3.py (running)
  input: {"call_id":"call_IcHq9MoulfbWnWmcWPErVlK9","process_id":"12881","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py

[tool] Read delora.py (completed)
  kind: read
  input: {"call_id":"call_yYuKd7kdZxGCZo8zkQtSGEiW","process_id":"76438","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
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
    ... (56 more lines)

[tool] Read ia3.py (completed)
  kind: read
  input: {"call_id":"call_IcHq9MoulfbWnWmcWPErVlK9","process_id":"12881","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
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
        28	from torch import nn
    ... (36 more lines)

[tool] Read pissa.py (completed)
  kind: read
  input: {"call_id":"call_DJiZUptMa5DJfJiqjcVAbgMf","process_id":"25734","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
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
        25	from torch import nn
        26	
        27	from ..variant import register, ParamSpec
        28	
    ... (43 more lines)

[tool] Read dora.py (running)
  input: {"call_id":"call_kGWNeMmPZ4fynlP2xkya29ml","process_id":"83073","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py

[tool] Read hra.py (running)
  input: {"call_id":"call_AUbNamw7Vff6IHpnxeDKla65","process_id":"18622","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py

[tool] Read antipasto.py (running)
  input: {"call_id":"call_K1UnwByBUsEV01RApE7a6wa0","process_id":"76559","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/antipasto.py"],"cwd":"/media/wassname/SGI...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/antipasto.py

[tool] Read eva.py (running)
  input: {"call_id":"call_aT1ZKpUsBsF0ukmeSQy5wQSa","process_id":"30592","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/eva.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/eva.py

[tool] Read dora.py (completed)
  kind: read
  input: {"call_id":"call_kGWNeMmPZ4fynlP2xkya29ml","process_id":"83073","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
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
        22	import torch.nn.functional as F
        23	from einops import einsum
        24	from torch import nn
        25	
        26	from ..variant import register, ParamSpec
        27	
        28	
    ... (38 more lines)

[tool] Read hra.py (completed)
  kind: read
  input: {"call_id":"call_AUbNamw7Vff6IHpnxeDKla65","process_id":"18622","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
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
    ... (50 more lines)

[tool] Read antipasto.py (completed)
  kind: read
  input: {"call_id":"call_K1UnwByBUsEV01RApE7a6wa0","process_id":"76559","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/antipasto.py"],"cwd":"/media/wassname/SGI...
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
    ... (117 more lines)

[tool] Read eva.py (completed)
  kind: read
  input: {"call_id":"call_aT1ZKpUsBsF0ukmeSQy5wQSa","process_id":"30592","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/eva.py"],"cwd":"/media/wassname/SGIronWol...
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
    ... (93 more lines)

[tool] Read adapter.py (running)
  input: {"call_id":"call_ZASqXizFcUeYWy0BfWtn7awK","process_id":"56583","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py

[tool] Read variant.py (running)
  input: {"call_id":"call_DTjBD9RvOp7yhpRxBMhsmmpH","process_id":"59269","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py

[tool] Read config.py (running)
  input: {"call_id":"call_gFTpgHVNt3AjUpsu0uccR8Xj","process_id":"57355","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/config.py"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/config.py

[tool] Read smoke.py (running)
  input: {"call_id":"call_x0N2faRQBhIMJ6dP09QwZiFv","process_id":"65832","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba tests/smoke.py"],"cwd":"/media/wassname/SGIronWolf/projects5/202...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/tests/smoke.py

[tool] Read target.py (running)
  input: {"call_id":"call_J7iRBknduLUEbwo0vQAyXzkS","process_id":"13496","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/target.py"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/target.py

[tool] Read adapter.py (completed)
  kind: read
  input: {"call_id":"call_ZASqXizFcUeYWy0BfWtn7awK","process_id":"56583","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
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

[tool] Read variant.py (completed)
  kind: read
  input: {"call_id":"call_DTjBD9RvOp7yhpRxBMhsmmpH","process_id":"59269","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
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
  input: {"call_id":"call_gFTpgHVNt3AjUpsu0uccR8Xj","process_id":"57355","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/config.py"],"cwd":"/media/wassname/SGIronWolf/proj...
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
        10	    alpha: float = 16.0
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

[tool] Read smoke.py (completed)
  kind: read
  input: {"call_id":"call_x0N2faRQBhIMJ6dP09QwZiFv","process_id":"65832","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba tests/smoke.py"],"cwd":"/media/wassname/SGIronWolf/projects5/202...
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
    ... (317 more lines)

[tool] Read target.py (completed)
  kind: read
  input: {"call_id":"call_J7iRBknduLUEbwo0vQAyXzkS","process_id":"13496","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/target.py"],"cwd":"/media/wassname/SGIronWolf/proj...
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
A few early runtime facts matter for every variant: adapter buffers are persistent (`register_buffer(..., persistent=True)`), all adapter tensors default to `cfg.dtype`, and hooks cast inputs into that dtype before variant forward. I’m now checking the reference snapshots around their init/forward code paths, then I’ll sanity-check the live docstring URLs.

[tool] Search lora_A|lora_B|scaling|kaiming|reset_lora|pissa|PiSSA|svd|dora|weight_norm|lora_magnitude_vector|magnitude in peft_lora_layer.py (running)
  input: {"call_id":"call_CXL2dLkBuKVYv7swdUx31r7B","process_id":"21601","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"lora_A|lora_B|scaling|kaiming|reset_lora|pissa|PiSSA|svd|dora|w...

[tool] Search ia3|ia3_l|is_feedforward|forward|update_layer|ones|Parameter in peft_ia3_layer.py (running)
  input: {"call_id":"call_QNRan5JW29Lt2z32ejlnQSiE","process_id":"83887","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"ia3|ia3_l|is_feedforward|forward|update_layer|ones|Parameter\" ...

[tool] Search hra|reflection|householder|reset_hra|hra_u|Parameter|forward|apply in peft_hra_layer.py (running)
  input: {"call_id":"call_l6wFam8sUpDhQje5Kw1ddbP9","process_id":"88970","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"hra|reflection|householder|reset_hra|hra_u|Parameter|forward|ap...

[tool] Search lora_A|lora_B|scaling|kaiming|reset_lora|pissa|PiSSA|svd|dora|weight_norm|lora_magnitude_vector|magnitude in peft_lora_layer.py (completed)
  kind: search
  input: {"call_id":"call_CXL2dLkBuKVYv7swdUx31r7B","process_id":"21601","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"lora_A|lora_B|scaling|kaiming|reset_lora|pissa|PiSSA|svd|dora|w...
  output:
    docs/refs/peft_lora_dora.py:66:        self._dora_cache: dict[str, Any] = {}  # small ad hoc cache; values are not part of the state_dict
    docs/refs/peft_lora_dora.py:70:        self._dora_cache[key] = value
    docs/refs/peft_lora_dora.py:74:        return self._dora_cache.get(key, default)
    docs/refs/peft_lora_dora.py:77:        self._dora_cache.clear()
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
    docs/refs/peft_lora_dora.py:124:            weight_norm = self.get_weight_norm(
    docs/refs/peft_lora_dora.py:125:                weight=weight.to(lora_A.device), lora_weight=lora_...

[tool] Search delora|lora_lambda|wnorm|weight_norm|norm|scaling|forward|reset|kaiming in peft_delora_layer.py (running)
  input: {"call_id":"call_eZ4HOEgS7NuaCwjmwX8tSURH","process_id":"24489","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"delora|lora_lambda|wnorm|weight_norm|norm|scaling|forward|reset...

[tool] Search ia3|ia3_l|is_feedforward|forward|update_layer|ones|Parameter in peft_ia3_layer.py (completed)
  kind: search
  input: {"call_id":"call_QNRan5JW29Lt2z32ejlnQSiE","process_id":"83887","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"ia3|ia3_l|is_feedforward|forward|update_layer|ones|Parameter\" ...
  output:
    30:    adapter_layer_names = ("ia3_l",)
    32:    def __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
    34:        self.ia3_l = nn.ParameterDict({})
    38:        self.is_feedforward = is_feedforward
    56:    def update_layer(self, adapter_name: str, config: IA3Config, **kwargs):
    57:        init_ia3_weights = config.init_ia3_weights
    62:        if self.is_feedforward:
    66:        self.ia3_l[adapter_name] = nn.Parameter(weight)
    67:        if init_ia3_weights:
    68:            self.reset_ia3_parameters(adapter_name)
    72:    def reset_ia3_parameters(self, adapter_name):
    73:        if adapter_name in self.ia3_l.keys():
    74:            # initialize learned vector with torch.ones
    75:            nn.init.constant_(self.ia3_l[adapter_name], 1.0)
    85:        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
    90:        IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)
    94:        self.update_layer(adapter_name, config=config)
    115:            if active_adapter in self.ia3_l.keys():
    117:                ia3_l = transpose(self.ia3_l[active_adapter].data, self.fan_in_fan_out)
    121:                    orig_weights = torch.mul(orig_weights, ia3_l)
    129:                    base_layer.weight.data = torch.mul(base_layer.weight.data, ia3_l).to(orig_dtype)
    131:                if not self.is_feedforward and (base_layer.bias is not None):
    132:                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
    149:            if active_adapter in self.ia3_l.keys():
    152:                ia3_l = transpose(self.ia3_l[active_adapter].data, self.fan_in_fan_out) + 1e-8
    154:                base_layer.weight.data = torch.div(base_layer.weight.data, ia3_l).to(orig_dtype)
    156:                if not self.is_feedforward and (base_layer.bias is not None):
    157:                    scaling = self.ia3_l[active_adapter].reshape(base_layer.bias.shape)
    ... (42 more lines)

[tool] Search hra|reflection|householder|reset_hra|hra_u|Parameter|forward|apply in peft_hra_layer.py (completed)
  kind: search
  input: {"call_id":"call_l6wFam8sUpDhQje5Kw1ddbP9","process_id":"88970","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"hra|reflection|householder|reset_hra|hra_u|Parameter|forward|ap...
  output:
    docs/refs/peft_hra_layer.py:30:    adapter_layer_names = ("hra_u",)
    docs/refs/peft_hra_layer.py:32:    other_param_names = ("hra_r", "hra_apply_GS")
    docs/refs/peft_hra_layer.py:36:        self.hra_r = {}
    docs/refs/peft_hra_layer.py:37:        self.hra_apply_GS = {}
    docs/refs/peft_hra_layer.py:38:        self.hra_u = nn.ParameterDict({})
    docs/refs/peft_hra_layer.py:42:        # flag to enable/disable casting of input to weight dtype during forward call
    docs/refs/peft_hra_layer.py:61:        """Internal function to create hra adapter
    docs/refs/peft_hra_layer.py:68:        apply_GS = config.apply_GS
    docs/refs/peft_hra_layer.py:75:        self.hra_r[adapter_name] = r
    docs/refs/peft_hra_layer.py:76:        self.hra_apply_GS[adapter_name] = apply_GS
    docs/refs/peft_hra_layer.py:81:            self.hra_u[adapter_name] = nn.Parameter(torch.empty(self.in_features, r), requires_grad=True)
    docs/refs/peft_hra_layer.py:83:            self.hra_u[adapter_name] = nn.Parameter(
    docs/refs/peft_hra_layer.py:92:            self.reset_hra_parameters(adapter_name)
    docs/refs/peft_hra_layer.py:94:            self.reset_hra_parameters_random(adapter_name)
    docs/refs/peft_hra_layer.py:100:    def reset_hra_parameters(self, adapter_name: str):
    docs/refs/peft_hra_layer.py:101:        if self.hra_r[adapter_name] % 2 != 0:
    docs/refs/peft_hra_layer.py:103:            nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))
    docs/refs/peft_hra_layer.py:105:            shape = self.hra_u[adapter_name].shape
    docs/refs/peft_hra_layer.py:108:            self.hra_u[adapter_name] = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))
    docs/refs/peft_hra_layer.py:110:    def reset_hra_parameters_random(self, adapter_name: str):
    docs/refs/peft_hra_layer.py:111:        nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))
    docs/refs/peft_hra_layer.py:118:            if active_adapter not in self.hra_u.keys():
    docs/refs/peft_hra_layer.py:125:            if active_adapter not in self....

[tool] Search delora|lora_lambda|wnorm|weight_norm|norm|scaling|forward|reset|kaiming in peft_delora_layer.py (completed)
  kind: search
  input: {"call_id":"call_eZ4HOEgS7NuaCwjmwX8tSURH","process_id":"24489","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"delora|lora_lambda|wnorm|weight_norm|norm|scaling|forward|reset...
  output:
    docs/refs/peft_delora_layer.py:32:        "delora_A",
    docs/refs/peft_delora_layer.py:33:        "delora_B",
    docs/refs/peft_delora_layer.py:34:        "delora_lambda",
    docs/refs/peft_delora_layer.py:39:        "delora_dropout",
    docs/refs/peft_delora_layer.py:40:        "delora_w_norm",
    docs/refs/peft_delora_layer.py:46:        self.delora_dropout = nn.ModuleDict({})
    docs/refs/peft_delora_layer.py:47:        self.delora_A = nn.ParameterDict({})
    docs/refs/peft_delora_layer.py:48:        self.delora_B = nn.ParameterDict({})
    docs/refs/peft_delora_layer.py:49:        self.delora_lambda = nn.ParameterDict({})
    docs/refs/peft_delora_layer.py:51:        self.delora_w_norm = BufferDict({}, persistent=True)
    docs/refs/peft_delora_layer.py:65:        A: torch.Tensor, B: torch.Tensor, delora_lambda: torch.Tensor, r: int, w_norm: torch.Tensor
    docs/refs/peft_delora_layer.py:67:        """Compute delta = B @ diag(delora_lambda/r / (||A_i||*||B^j||)) @ A, scaled by provided w_norm (per-input channel)"""
    docs/refs/peft_delora_layer.py:68:        An = torch.clamp(A.norm(dim=1), min=1e-4)
    docs/refs/peft_delora_layer.py:69:        Bn = torch.clamp(B.norm(dim=0), min=1e-4)
    docs/refs/peft_delora_layer.py:70:        diag = torch.diag_embed(delora_lambda / r / (An * Bn))
    docs/refs/peft_delora_layer.py:72:        delta = delta * w_norm.unsqueeze(0)
    docs/refs/peft_delora_layer.py:76:        if adapter not in self.delora_A or adapter not in self.delora_B:
    docs/refs/peft_delora_layer.py:80:            self.delora_A[adapter],
    docs/refs/peft_delora_layer.py:81:            self.delora_B[adapter],
    docs/refs/peft_delora_layer.py:82:            self.delora_lambda[adapter],
    docs/refs/peft_delora_layer.py:84:            self.delora_w_norm[adapter],
    docs/refs/peft_delora_layer.py:92:        delora_lambda: float,
    docs/refs/peft_delora_layer.py:96:        """Internal function to create delora adapter
    docs/refs/peft_delora_layer.py:101:            delora_lambda (`float`): Boundary for the adapter's norm.
    ...

[tool] Read peft_lora_layer.py (running)
  input: {"call_id":"call_JRbF1Xn6kmbu1aeGenPoTbn3","process_id":"19417","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '360,395p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py

[tool] Search SVD|svd|delta_s|rot|Cayley|rotate|U|Vh|V in antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_eDElMn5nrdWzS4IgBNYQZ4gS","process_id":"39796","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"SVD|svd|delta_s|rot|Cayley|rotate|U|Vh|V\" docs/refs/antipasto3...

[tool] Search initialize_lora_eva_weights|eva|SVD|svd|IncrementalPCA|lora_A|lora_B|rank|explained|dropout in peft_eva.py (running)
  input: {"call_id":"call_XGlVYcHZnz3nyKUx46IAmL8z","process_id":"62921","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"initialize_lora_eva_weights|eva|SVD|svd|IncrementalPCA|lora_A|l...

[tool] Read peft_lora_layer.py (running)
  input: {"call_id":"call_ppwxQIwOtJyElrbeCwjGwIvU","process_id":"22962","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '958,972p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py

[tool] Search SVD|svd|delta_s|rot|Cayley|rotate|U|Vh|V in antipasto3_svd_adapter.py (completed)
  kind: search
  input: {"call_id":"call_eDElMn5nrdWzS4IgBNYQZ4gS","process_id":"39796","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"SVD|svd|delta_s|rot|Cayley|rotate|U|Vh|V\" docs/refs/antipasto3...
  output:
    1:"""SVD adapter for bidirectional steering via block-diagonal Cayley rotations.
    5:kernel = U @ diag(S) @ Vh + W_res   (kernel is (in, out), standard Flax convention)
    6:Learnable: delta_s (additive S scaling), rotation_params (block-diagonal V rotation).
    7:alpha scales both: S + alpha*delta_s, U @ R(alpha).
    9:Why Cayley (not Givens or matrix exponential):
    10:Cayley gives exact analytical reversibility: R(-alpha) = R(alpha)^{-1}.
    14:At alpha=0: U_rot = U and S_scaled = S, so the layer is identical to frozen weights.
    30:class SVDParam(nnx.Param):
    31:    """Trainable SVD adapter parameter (base class -- use subclasses for per-group LR)."""
    35:class DeltaSParam(SVDParam):
    36:    """Trainable delta_s scaling parameters (full LR)."""
    40:class RotationParam(SVDParam):
    41:    """Block-diagonal rotation parameters (lower LR via rotation_lr_scale config)."""
    45:class SVDFrozen(nnx.Variable):
    46:    """Frozen SVD component. Not differentiated."""
    50:# -- SVD Steering Linear (replaces nnx.Linear) ------------------------------
    52:class SVDSteeringLinear(nnx.Module):
    53:    """SVD steering adapter replacing a linear layer.
    55:    y = ((x @ U_rot) * S_scaled) @ Vh + x @ W_res
    57:    where U_rot, S_scaled depend on alpha (the steering coefficient).
    58:    Frozen: U, S, Vh, W_res.  Learnable: delta_s, rotation_params.
    63:        U: jax.Array,               # (in_dim, r) - input singular vectors
    65:        Vh: jax.Array,              # (r, out_dim) - output singular vectors
    67:        rotation_block_size: int,
    68:        max_rotation_angle: float,
    69:        rotate_U: bool = True,
    70:        rotate_V: bool = False,
    71:        use_delta_s: bool = True,
    ... (90 more lines)

[tool] Read peft_lora_layer.py (completed)
  kind: read
  input: {"call_id":"call_JRbF1Xn6kmbu1aeGenPoTbn3","process_id":"19417","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '360,395p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        def pissa_init(self, adapter_name, init_lora_weights):
            weight = self.get_base_layer().weight
            dtype = weight.dtype
            if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                raise TypeError(
                    "Please initialize PiSSA under float32, float16, or bfloat16. "
                    "Subsequently, re-quantize the residual model to help minimize quantization errors."
                )
            weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
            if init_lora_weights == "pissa":
                # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
                V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
                Vr = V[:, : self.r[adapter_name]]
                Sr = S[: self.r[adapter_name]]
                Sr /= self.scaling[adapter_name]
                Uhr = Uh[: self.r[adapter_name]]
            elif len(init_lora_weights.split("_niter_")) == 2:
                Vr, Sr, Ur = svd_lowrank(
                    weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
                )
                Sr /= self.scaling[adapter_name]
                Uhr = Ur.t()
            else:
                raise ValueError(
                    f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
                )
    
    ... (9 more lines)

[tool] Read peft_lora_dora.py (running)
  input: {"call_id":"call_76bagUsTDjh2isFAy7PqqYus","process_id":"47762","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '132,166p' docs/refs/peft_lora_dora.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_dora.py

[tool] Search initialize_lora_eva_weights|eva|SVD|svd|IncrementalPCA|lora_A|lora_B|rank|explained|dropout in peft_eva.py (completed)
  kind: search
  input: {"call_id":"call_XGlVYcHZnz3nyKUx46IAmL8z","process_id":"62921","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","rg -n \"initialize_lora_eva_weights|eva|SVD|svd|IncrementalPCA|lora_A|l...
  output:
    docs/refs/peft_eva_finetuning.py:21:from peft import EvaConfig, LoraConfig, get_peft_model, initialize_lora_eva_weights
    docs/refs/peft_eva_finetuning.py:29:rank = 16
    docs/refs/peft_eva_finetuning.py:33:svd_batch_size = 4  # can be different from the batch size used in finetuning
    docs/refs/peft_eva_finetuning.py:61:    batch_size=svd_batch_size,
    docs/refs/peft_eva_finetuning.py:66:eva_config = EvaConfig(rho=rho)
    docs/refs/peft_eva_finetuning.py:68:    r=rank, lora_alpha=alpha, target_modules=target_modules, init_lora_weights="eva", eva_config=eva_config
    docs/refs/peft_eva_finetuning.py:74:# to optimize memory usage during eva initialization, set low_cpu_mem_usage=True
    docs/refs/peft_eva_finetuning.py:76:initialize_lora_eva_weights(peft_model, dataloader)
    docs/refs/peft_eva.py:31:from peft.utils.incremental_pca import IncrementalPCA
    docs/refs/peft_eva.py:106:class SVDHook(_Hook):
    docs/refs/peft_eva.py:108:    A forward hook for calculating incremental SVD on layer inputs. The hook is designed to be registered to a PyTorch
    docs/refs/peft_eva.py:111:    This hook performs a step of incremental Singular Value Decomposition (SVD) on the inputs of a specified layer
    docs/refs/peft_eva.py:119:        prepare_layer_inputs_fn (Optional[callable]): Function to prepare layer inputs for SVD.
    docs/refs/peft_eva.py:138:        self.svd = IncrementalPCA(
    docs/refs/peft_eva.py:141:            lowrank=True,
    docs/refs/peft_eva.py:142:            lowrank_seed=42,
    docs/refs/peft_eva.py:150:        if hasattr(self.svd, "components_"):
    docs/refs/peft_eva.py:151:            previous_components = self.svd.components_.clone().detach()
    docs/refs/peft_eva.py:156:            print(f"skipping SVD for {self.name} because there are less than {self.n_components} examples")
    docs/refs/peft_eva.py:158:        self.svd.partial_fit(states.to(torch.float32))
    docs/refs/peft_eva.py:162:        components = self.svd.components_
    docs/refs/peft_eva.py:171:# This is used to determine if inputs of two differe...

[tool] Read peft_lora_layer.py (completed)
  kind: read
  input: {"call_id":"call_ppwxQIwOtJyElrbeCwjGwIvU","process_id":"22962","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '958,972p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
                lora_A_keys = self.lora_A.keys()
                for active_adapter in self.active_adapters:
                    if active_adapter not in lora_A_keys:
                        continue
    
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                    else:
                        result = self.lora_variant[active_adapter].forward(
                            self,

[tool] Read peft_lora_dora.py (completed)
  kind: read
  input: {"call_id":"call_76bagUsTDjh2isFAy7PqqYus","process_id":"47762","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '132,166p' docs/refs/peft_lora_dora.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_dora.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
            """
            For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
            output.
            """
            lora_weight = self.get_lora_weight(lora_A=lora_A, lora_B=lora_B, adapter_name=adapter_name)
            lora_weight = lora_weight.to(x.dtype)
    
            magnitude = self.weight
            weight = dequantize_module_weight(base_layer)
            weight = weight.to(x.dtype)
            weight_norm = self.get_weight_norm(
                weight=weight, lora_weight=lora_weight.detach(), scaling=scaling, adapter_name=adapter_name
            )
            # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
            # "[...] we suggest treating ||V +∆V ||_c in
            # Eq. (5) as a constant, thereby detaching it from the gradient
            # graph. This means that while ||V + ∆V ||_c dynamically
            # reflects the updates of ∆V , it won’t receive any gradient
            # during backpropagation"
            weight_norm = weight_norm.detach()
            mag_norm_scale = (magnitude / weight_norm).view(1, -1)
    
            lora_result = lora_B(lora_A(x))
    
            bias = None
            if base_result is not None:
    ... (8 more lines)

[tool] Read peft_delora_layer.py (running)
  input: {"call_id":"call_STRTnAVOq9WZ4NOQ1m9F24J2","process_id":"78221","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '246,261p' docs/refs/peft_delora_layer.py"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py

[tool] Read peft_delora_layer.py (running)
  input: {"call_id":"call_mcVVrgkceSeX6A8QIU5qL1OQ","process_id":"75092","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '128,161p' docs/refs/peft_delora_layer.py"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py

[tool] Read peft_ia3_layer.py (running)
  input: {"call_id":"call_mVR5vVoBRVvdD7i08awFEP5E","process_id":"7411","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '291,317p' docs/refs/peft_ia3_layer.py"],"cwd":"/media/wassname/S...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py

[tool] Read peft_hra_layer.py (running)
  input: {"call_id":"call_L8erIsjTuP4GFpaT5V50afZP","process_id":"81187","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '75,108p' docs/refs/peft_hra_layer.py"],"cwd":"/media/wassname/S...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py

[tool] Read peft_delora_layer.py (completed)
  kind: read
  input: {"call_id":"call_mcVVrgkceSeX6A8QIU5qL1OQ","process_id":"75092","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '128,161p' docs/refs/peft_delora_layer.py"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        def reset_delora_parameters(
            self,
            adapter_name: str,
            init_weights: bool = True,
            delora_lambda: float = 15.0,
        ) -> None:
            if adapter_name not in self.delora_A.keys():
                return
    
            if init_weights is True:
                nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
                nn.init.zeros_(self.delora_B[adapter_name])
            else:
                nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.delora_B[adapter_name], a=math.sqrt(5))
    
            self.delora_lambda[adapter_name].data.fill_(float(delora_lambda))
    
            # capture a fixed norm for this adapter to use for future delta computations
            with torch.no_grad():
                w = self.get_base_layer().weight
                if w.device.type != "meta":
                    w_norm = torch.norm(w.data, dim=0).detach()
                else:
                    # For meta tensors, we can't compute the norm, so use a default value
                    w_norm = torch.ones(w.shape[1], device=w.device)
                self.delora_w_norm[adapter_name] = w_norm
    ... (7 more lines)

[tool] Read peft_delora_layer.py (completed)
  kind: read
  input: {"call_id":"call_STRTnAVOq9WZ4NOQ1m9F24J2","process_id":"78221","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '246,261p' docs/refs/peft_delora_layer.py"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_delora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
                    x_d = self.delora_dropout[adapter](x)
    
                    # Decomposed delta calculation
                    # 1. (x * w_norm) @ A.T
                    h = nn.functional.linear(x_d * self.delora_w_norm[adapter], self.delora_A[adapter])
    
                    # 2. h @ diag
                    An = torch.clamp(self.delora_A[adapter].norm(dim=1), min=1e-4)
                    Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-4)
                    scaling = (self.delora_lambda[adapter] / self.r[adapter]) / (An * Bn)
    
                    h = h * scaling
    
                    # 3. h @ B.T
                    h = nn.functional.linear(h, self.delora_B[adapter])

[tool] Read peft_ia3_layer.py (running)
  input: {"call_id":"call_T5LuGO9J3PslVssywEOuH7uX","process_id":"64921","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '208,218p' docs/refs/peft_ia3_layer.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py

[tool] Read peft_hra_layer.py (running)
  input: {"call_id":"call_cK0RZeM4kle9U9I3sben8qMP","process_id":"80820","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '209,265p' docs/refs/peft_hra_layer.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py

[tool] Read peft_ia3_layer.py (completed)
  kind: read
  input: {"call_id":"call_mVR5vVoBRVvdD7i08awFEP5E","process_id":"7411","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '291,317p' docs/refs/peft_ia3_layer.py"],"cwd":"/media/wassname/S...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            dtype = previous_dtype = x.dtype
    
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                ia3_scaling = 1
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.ia3_l.keys():
                        continue
                    dtype = self.ia3_l[active_adapter].dtype
                    ia3_scaling *= self.ia3_l[active_adapter]
    
                if self.is_feedforward:
                    x = x.to(dtype)
                    # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
                    # e.g. bf16 vs fp32. Is that okay?
                    interm = (x * ia3_scaling).to(self.get_base_layer().weight.dtype)
                    result = self.base_layer(interm, *args, **kwargs)
                else:
                    result = self.base_layer(x, *args, **kwargs)
                    result = result.to(dtype) * ia3_scaling

[tool] Read peft_hra_layer.py (completed)
  kind: read
  input: {"call_id":"call_L8erIsjTuP4GFpaT5V50afZP","process_id":"81187","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '75,108p' docs/refs/peft_hra_layer.py"],"cwd":"/media/wassname/S...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
            self.hra_r[adapter_name] = r
            self.hra_apply_GS[adapter_name] = apply_GS
    
            # Determine shape of HRA weights
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.Linear):
                self.hra_u[adapter_name] = nn.Parameter(torch.empty(self.in_features, r), requires_grad=True)
            elif isinstance(base_layer, nn.Conv2d):
                self.hra_u[adapter_name] = nn.Parameter(
                    torch.empty(self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], r),
                    requires_grad=True,
                )
            else:
                raise TypeError(f"HRA is not implemented for base layers of type {type(base_layer).__name__}")
    
            # Initialize weights
            if init_weights:
                self.reset_hra_parameters(adapter_name)
            else:
                self.reset_hra_parameters_random(adapter_name)
    
            # Move new weights to device
            self._move_adapter_to_device_of_base_layer(adapter_name)
            self.set_adapter(self.active_adapters, inference_mode=inference_mode)
    
        def reset_hra_parameters(self, adapter_name: str):
            if self.hra_r[adapter_name] % 2 != 0:
    ... (7 more lines)

[tool] Read peft_ia3_layer.py (completed)
  kind: read
  input: {"call_id":"call_T5LuGO9J3PslVssywEOuH7uX","process_id":"64921","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '208,218p' docs/refs/peft_ia3_layer.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_ia3_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        def update_layer(self, adapter_name: str, config: IA3Config, **kwargs):
            init_ia3_weights = config.init_ia3_weights
            inference_mode = config.inference_mode
    
            # Actual trainable parameters
            num_features = self.in_features if self.is_feedforward else self.out_features
            weights_size = (1, num_features) + (1,) * (self._kernel_dim - 2)
            weight = torch.randn(weights_size)
            self.ia3_l[adapter_name] = nn.Parameter(weight)
            if init_ia3_weights:
                self.reset_ia3_parameters(adapter_name)

[tool] Read peft_hra_layer.py (completed)
  kind: read
  input: {"call_id":"call_cK0RZeM4kle9U9I3sben8qMP","process_id":"80820","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '209,265p' docs/refs/peft_hra_layer.py"],"cwd":"/media/wassname/...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_hra_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
            rank = self.hra_r[adapter_name]
            apply_GS = self.hra_apply_GS[adapter_name]
            opt_u = self.hra_u[adapter_name]
            shape = opt_u.shape
    
            if apply_GS:
                weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
                for i in range(1, rank):
                    ui = opt_u[:, i].view(-1, 1)
                    for j in range(i):
                        ui = ui - (weight[j].t() @ ui) * weight[j]
                    weight.append((ui / ui.norm()).view(-1, 1))
                weight = torch.cat(weight, dim=1)
                weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()
    
            else:
                opt_u = opt_u / opt_u.norm(dim=0)
                weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
                if reverse:
                    indices = range(rank - 1, -1, -1)
                else:
                    indices = range(rank)
    
                for i in indices:
                    ui = opt_u[:, i].view(-1, 1)
                    weight = weight - 2 * weight @ ui @ ui.t()
    
    ... (30 more lines)

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_3DbrzJWMQFTTW4ljfMAkk6He","process_id":"80755","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '76,104p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/wa...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_AlPlAWgDyADHefwo8wb8BEUN","process_id":"30191","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '117,169p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/w...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read peft_lora_layer.py (running)
  input: {"call_id":"call_mCvMQv5zxOdAqhUiNiSkW8xv","process_id":"99551","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '197,215p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py

[tool] Read antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_Vo9wquYekZsvuw8ec5SyrNMu","process_id":"25940","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '180,212p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/w...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py

[tool] Read peft_eva.py (running)
  input: {"call_id":"call_ATOXB8ZOgMjTzLGjtkSiskTI","process_id":"71854","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '466,532p' docs/refs/peft_eva.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_AlPlAWgDyADHefwo8wb8BEUN","process_id":"30191","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '117,169p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/w...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
            U = self.svd_U.value
            S = self.svd_S.value
            Vh = self.svd_Vh.value
            W_res = self.svd_W_res.value
            params = self.rotation_params.value  # (n_blocks, n_triu)
            bs = self.block_size
            n_blocks = params.shape[0]
    
            # Reconstruct skew-symmetric from upper-triangle params (like OFT/PSOFT).
            # 0.5 factor matches BOFT convention: cancels the 2x gradient from A - A^T.
            A = jnp.zeros((n_blocks, bs, bs), dtype=jnp.float32)
            A = A.at[:, self._triu_rows, self._triu_cols].set(params.astype(jnp.float32))
            A = 0.5 * (A - jnp.swapaxes(A, -1, -2))
    
            # Angle clamping (element-wise tanh, bounds bidirectional symmetry error)
            a_limit = 2 * math.tan(self.max_angle / 2)
            A = a_limit * jnp.tanh(A / a_limit)
    
            # Cayley transform in float32: R = (I - X)^{-1}(I + X)
            eye = jnp.eye(bs, dtype=jnp.float32)
            X = alpha * A / 2
            R_blocks = jnp.linalg.solve(
                eye[None] - X,
                eye[None] + X,
            )
    
            # Apply rotation to U (input singular vectors)
    ... (26 more lines)

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_3DbrzJWMQFTTW4ljfMAkk6He","process_id":"80755","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '76,104p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/wa...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
            bs = min(rotation_block_size, r)
            assert r % bs == 0, f"r={r} must be divisible by block_size={bs}"
    
            # Frozen SVD components
            self.svd_U = SVDFrozen(U.astype(jnp.float32))
            self.svd_S = SVDFrozen(S.astype(jnp.float32))
            self.svd_Vh = SVDFrozen(Vh.astype(jnp.float32))
            self.svd_W_res = SVDFrozen(W_res.astype(jnp.bfloat16))
    
            # Trainable: delta_s with small positive bias for symmetry breaking.
            # The +4e-4 nudges the optimizer to scale up selected dims rather than just
            # rotating them. Rotation alone cannot break sign symmetry at init.
            key_s, key_r = jax.random.split(rngs.params())
            self.delta_s = DeltaSParam(
                jax.random.truncated_normal(key_s, -2.0, 2.0, (r,)) * 4e-4 + 4e-4
            )
    
            # Block-diagonal skew-symmetric rotation params (lower LR via RotationParam type)
            # Upper-triangle parameterization: store only bs*(bs-1)/2 elements per block,
            # like OFT/PSOFT. Avoids dead diagonal gradients and redundant (i,j)/(j,i) states.
            n_blocks = r // bs
            n_triu = bs * (bs - 1) // 2
            self.rotation_params = RotationParam(
                jax.random.truncated_normal(key_r, -2.0, 2.0, (n_blocks, n_triu)) * 1e-4
            )
            # Pre-compute upper-triangle indices for skew-symmetric reconstruction
            rows, cols = jnp.triu_indices(bs, k=1)
    ... (2 more lines)

[tool] Read antipasto3_svd_adapter.py (completed)
  kind: read
  input: {"call_id":"call_Vo9wquYekZsvuw8ec5SyrNMu","process_id":"25940","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '180,212p' docs/refs/antipasto3_svd_adapter.py"],"cwd":"/media/w...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/antipasto3_svd_adapter.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        """Create SVD adapter from a kernel matrix (in_dim, out_dim).
    
        If selected_indices is provided, uses those SVD dimensions.
        Otherwise uses top-r by singular value.
        """
        kernel_f32 = kernel.astype(jnp.float32)
        U_full, S_full, Vh_full = jnp.linalg.svd(kernel_f32, full_matrices=False)
    
    
        r_actual = min(r, S_full.shape[0])
        # Ensure divisible by block size
        bs = min(rotation_block_size, r_actual)
        r_actual = (r_actual // bs) * bs
        if r_actual == 0:
            r_actual = bs
    
        if selected_indices is not None:
            indices = selected_indices[:r_actual]
        else:
            indices = jnp.arange(r_actual)
    
        U = U_full[:, indices]
        S = S_full[indices]
        Vh = Vh_full[indices, :]
        W_res = kernel_f32 - U @ jnp.diag(S) @ Vh
    
        return SVDSteeringLinear(
    ... (6 more lines)

[tool] Read peft_lora_layer.py (completed)
  kind: read
  input: {"call_id":"call_mCvMQv5zxOdAqhUiNiSkW8xv","process_id":"99551","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '197,215p' docs/refs/peft_lora_layer.py"],"cwd":"/media/wassname...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_lora_layer.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
    
            # Tying adapters is only implemented for Linear layers
            # where the source is the embedding layer.
            # Currently, this is the most prevelant way of tying layers (weight tying)
            if tied_adapter:
                lora_A_params = tied_adapter["lora_A"]
                lora_B_params = tied_adapter["lora_B"]
    
                self.lora_A[adapter_name].weight = torch.nn.Parameter(lora_A_params)
                self.lora_B[adapter_name].weight = torch.nn.Parameter(lora_B_params)
    
            self.lora_bias[adapter_name] = lora_bias
    
            if use_rslora:
                self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            else:
                self.scaling[adapter_name] = lora_alpha / r

[tool] Read peft_eva.py (completed)
  kind: read
  input: {"call_id":"call_ATOXB8ZOgMjTzLGjtkSiskTI","process_id":"71854","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '466,532p' docs/refs/peft_eva.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/refs/peft_eva.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        eva_state_dict = {}
        for name, rank in rank_dist.items():
            hook = hooks[layer_hook_map[name]][0]
            if not torch.all(hook.converged[:rank]):
                raise ValueError(
                    f"Layer {name} has not converged but was assigned rank {rank}. "
                    "Please report this issue at https://github.com/huggingface/peft/issues"
                )
            u = hook.svd.components_[:rank]
            if peft_config.eva_config.whiten:
                u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
            eva_state_dict[name] = u
    
        # restore model state
        model.train(training)
    
        # move tensors to device
        if device is not None:
            eva_state_dict = {k: v.to(device) for k, v in eva_state_dict.items()}
    
        return eva_state_dict
    
    
    def _load_eva_state_dict(
        model: torch.nn.Module,
        eva_state_dict: dict,
        adapter_name: str,
    ... (40 more lines)

[tool] Searching the Web (running)

[tool] Opening: https://arxiv.org/abs/2106.09685 (completed)
  kind: fetch
  input: https://arxiv.org/abs/2106.09685

[tool] Searching the Web (running)

[tool] Opening: https://arxiv.org/abs/2402.09353 (completed)
  kind: fetch
  input: https://arxiv.org/abs/2402.09353

[tool] Searching the Web (running)

[tool] Opening: https://github.com/MuLabPKU/PiSSA/blob/main/utils/init_pissa.py (completed)
  kind: fetch
  input: https://github.com/MuLabPKU/PiSSA/blob/main/utils/init_pissa.py

[tool] Searching the Web (running)

[tool] Opening: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py (completed)
  kind: fetch
  input: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py

[tool] Searching the Web (running)

[tool] Opening: https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py (completed)
  kind: fetch
  input: https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py

[tool] Searching the Web (running)
The live GitHub pages mostly resolve, though a couple of `web.open` calls return GitHub-side/internal fetch errors for specific pages. Since this is explicitly a GitHub-link task, I’m also checking the `gh` skill instructions so I can use the local GitHub CLI path for raw-file sanity where possible.

[tool] Opening: https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py (completed)
  kind: fetch
  input: https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py

[tool] Read SKILL.md (running)
  input: {"call_id":"call_hKiKdMrKAYVBO6nRKIdT6Ulh","process_id":"5086","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gh/SKILL.md"],"cwd":"/medi...
  files: /home/wassname/.claude/skills/gh/SKILL.md

[tool] Read SKILL.md (completed)
  kind: read
  input: {"call_id":"call_hKiKdMrKAYVBO6nRKIdT6Ulh","process_id":"5086","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gh/SKILL.md"],"cwd":"/medi...
  files: /home/wassname/.claude/skills/gh/SKILL.md
  output:
    ---
    name: gh
    description: "Search GitHub, fetch files, and explore repos using the `gh` CLI. Use when a user provides a GitHub URL, asks to find repos/code, or wants to read files from GitHub."
    ---
    
    # GitHub CLI (gh)
    
    Requires authentication: run `gh auth status` to check, `gh auth login` if needed.
    
    ## 1. Search
    
    Delegate search to a sub-agent (Task tool) to scan many results (50-100+) without filling
    your context. The sub-agent runs the search, filters/ranks results, and returns a short summary.
    
    ### Search repos (start here)
    
    Include `--match name,description,readme` for best recall (default only matches name+description).
    
    ```bash
    GH_PAGER="" gh search repos "query" \
      --match name,description,readme \
      --sort stars \
      --limit 50 \
      --json fullName,description,stargazersCount,url
    ```
    
    Additional filters: `--owner OWNER`, `--language python`, `--topic ML`, `--stars ">=100"`,
    `--created ">2024-01-01"`, `--include-forks true` (off by default, adds noise but improves recall).
    ... (106 more lines)

[tool] Run wc -l docs/refs/peft_lora_layer.py docs/refs/peft_lora_dora.py docs/refs/peft_delora_layer.py docs/refs/peft_ia3_layer.py docs/refs/peft_hra_layer.py docs/refs/peft_eva.py docs/refs/antipasto3_svd_adapter.py (running)
  input: {"call_id":"call_WBalwg2A0Sgj102Nrdm80gLZ","process_id":"79817","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","wc -l docs/refs/peft_lora_layer.py docs/refs/peft_lora_dora.py docs/ref...

[tool] Run gh auth status (running)
  input: {"call_id":"call_MDZ9JcfpcHiA7VSnDD4Hj27Q","process_id":"22378","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","gh auth status"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/lora-...

[tool] Run wc -l docs/refs/peft_lora_layer.py docs/refs/peft_lora_dora.py docs/refs/peft_delora_layer.py docs/refs/peft_ia3_layer.py docs/refs/peft_hra_layer.py docs/refs/peft_eva.py docs/refs/antipasto3_svd_adapter.py (completed)
  kind: execute
  input: {"call_id":"call_WBalwg2A0Sgj102Nrdm80gLZ","process_id":"79817","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","wc -l docs/refs/peft_lora_layer.py docs/refs/peft_lora_dora.py docs/ref...
  output:
    2510 docs/refs/peft_lora_layer.py
       287 docs/refs/peft_lora_dora.py
       274 docs/refs/peft_delora_layer.py
       336 docs/refs/peft_ia3_layer.py
       462 docs/refs/peft_hra_layer.py
       739 docs/refs/peft_eva.py
       375 docs/refs/antipasto3_svd_adapter.py
      4983 total

[tool] Run gh auth status (failed)
  kind: execute
  input: {"call_id":"call_MDZ9JcfpcHiA7VSnDD4Hj27Q","process_id":"22378","turn_id":"019dc999-8ad5-7161-9386-95542608fab7","command":["/usr/bin/zsh","-lc","gh auth status"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/lora-...
  output:
    github.com
      X Failed to log in to github.com account wassname (/home/wassname/.config/gh/hosts.yml)
      - Active account: true
      - The token in /home/wassname/.config/gh/hosts.yml is invalid.
      - To re-authenticate, run: gh auth login -h github.com
      - To forget about this account, run: gh auth logout -h github.com -u wassname
    
      X Failed to log in to github.com account ShouldHaveSeenThat (default)
      - Active account: false
      - The token in default is invalid.
      - To re-authenticate, run: gh auth login -h github.com
      - To forget about this account, run: gh auth logout -h github.com -u ShouldHaveSeenThat

[tool] Searching the Web (running)

[tool] Open page (completed)
  kind: fetch
  input: {"query":"","action":{"type":"open_page"}}

[tool] Searching the Web (running)
## lora

### params
- `lora_A`: `(r, d_in)`, `cfg.dtype`, trainable Parameter, not buffer at `src/lora_lite/variants/lora.py:26`; matches PEFT LoRA shape.
- `lora_B`: `(d_out, r)`, `cfg.dtype`, trainable Parameter, not buffer at `src/lora_lite/variants/lora.py:27`; matches PEFT. Buffers would persist if used via `persistent=True` at `src/lora_lite/adapter.py:51-53`.

### init / group_init
- `init()` no-op at `src/lora_lite/variants/lora.py:31-33`; ParamSpec does Kaiming `A`, zero `B`.
- GRADIENT FLOW at t=0: `lora_B` gets non-zero grad; `lora_A` gets zero grad because `B=0`.

### dtype
- Params allocated in `cfg.dtype` at `src/lora_lite/adapter.py:51-56`; hook casts `x` to `cfg.dtype` then returns `y.dtype` at `src/lora_lite/adapter.py:18-20`.
- No fp32-sensitive init. bf16 identity survives because `B=0`.

### forward
Math: `y + (alpha/r) x A^T B^T` at `src/lora_lite/variants/lora.py:37-41`.
Ref math: `result + scaling * B(A(dropout(x)))`; PEFT has dropout, this library has no dropout field in `src/lora_lite/config.py:6-21`.
Match? YES, modulo intentionally absent dropout.

### links
- paper: OK
- peft ref: OK
- author ref: n/a
- offline snapshot drift: AMBIGUOUS; live URL resolves, but `gh auth status` token is invalid so I could not hash raw snapshot.

### verdict
CORRECT -- standard LoRA mechanism matches PEFT/paper.

## pissa

### params
- `lora_A`: `(r, d_in)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/pissa.py:37`.
- `lora_B`: `(d_out, r)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/pissa.py:38`.
- Shapes match PEFT PiSSA; no buffers needed.

### init / group_init
- Requires exact `nn.Linear`, then fp32 SVD of `W`, copies `B=U_r sqrt(S_r)`, `A=sqrt(S_r) Vh_r`, and mutates base to `W - scale*BA` at `src/lora_lite/variants/pissa.py:42-63`.
- GRADIENT FLOW at t=0: both `A` and `B` get non-zero grad because both are non-zero.

### dtype
- SVD is correctly fp32 at `src/lora_lite/variants/pissa.py:48-55`; subtraction uses fp32 `BA` at `src/lora_lite/variants/pissa.py:58-63`.
- Precision loss: stored `A/B` are cast to `cfg.dtype` before residual subtraction at `src/lora_lite/variants/pissa.py:54-59`; this mirrors the doc’s “bf16 round-trip not bit-exact” caveat at `src/lora_lite/variants/pissa.py:4`.

### forward
Math: `y + (alpha/r) x A^T B^T` at `src/lora_lite/variants/pissa.py:66-71`.
Ref math: PEFT divides singular values by scaling before forming `A/B`, then subtracts `scaling*BA`; equivalent identity, but different adapter parameter scale when `alpha/r != 1`.
Match? PARTIAL; documented deviation says pass `alpha=r` for paper-faithful identity at `src/lora_lite/variants/pissa.py:6-13`.

### links
- paper: OK
- peft ref: OK
- author ref: OK
- offline snapshot drift: AMBIGUOUS; live URLs resolve, raw hash not checked due invalid `gh` auth.

### verdict
PARTIAL -- mechanism is correct for `alpha=r`; non-unit scale does not match PEFT’s PiSSA parameterization.

## delora

### params
- `lora_A`: `(r, d_in)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/delora.py:49`.
- `lora_B`: `(d_out, r)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/delora.py:50`.
- `lora_lambda`: scalar `()`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/delora.py:51-53`.
- `lora_wnorm`: `(d_in,)`, `cfg.dtype`, frozen persistent buffer at `src/lora_lite/variants/delora.py:54-56` and `src/lora_lite/adapter.py:51-53`; matches PEFT’s persistent `delora_w_norm`.

### init / group_init
- Captures per-input-column norm of `W` into `lora_wnorm` at `src/lora_lite/variants/delora.py:60-66`.
- Default `lambda0=0` at `src/lora_lite/variants/delora.py:47`; this is intentionally different from PEFT/paper defaults.
- GRADIENT FLOW at t=0: with default `lambda0=0`, only `lora_lambda` gets non-zero grad; `A/B` get zero first-step grad. This is already documented at `src/lora_lite/variants/delora.py:20-24`.

### dtype
- `wnorm` is computed fp32 then stored as adapter dtype at `src/lora_lite/variants/delora.py:64-66`.
- Norms in forward use adapter dtype at `src/lora_lite/variants/delora.py:79-81`; fp16/bf16 can perturb normalization, but clamp prevents division blow-up.

### forward
Math: `y + ((x*||W||_col)A^T) * ((lambda/r)/(||A_i||||B_i||)) B^T` at `src/lora_lite/variants/delora.py:77-84`.
Ref math: PEFT DeLoRA same per-input norm and per-rank normalization; PEFT also has dropout.
Match? YES for upstream PEFT forward, modulo no dropout and documented init deviation.

### links
- paper: OK
- peft ref: DEAD/UNVERIFIED via browser cache miss for exact URL
- author ref: DEAD/UNVERIFIED via browser cache miss for exact URL
- offline snapshot drift: AMBIGUOUS

### verdict
PARTIAL -- forward matches PEFT, but identity/default init intentionally sacrifices first-step `A/B` gradients.

## ia3

### params
- `ia3`: `lora_g` `(d_out,)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/ia3.py:38-39`.
- `ia3_ff`: `lora_g` `(d_in,)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/ia3.py:55-56`.
- Shapes match PEFT’s output vs feedforward feature choice.

### init / group_init
- Both init no-op; ParamSpec initializes `g=1` at `src/lora_lite/variants/ia3.py:41-43` and `src/lora_lite/variants/ia3.py:58-60`.
- GRADIENT FLOW at t=0: `lora_g` gets non-zero grad immediately for both variants.

### dtype
- Gate stored in `cfg.dtype`; output hook casts `x`, not `y`, but multiplies `y * lora_g` at `src/lora_lite/variants/ia3.py:46-47`; result then returns to `y.dtype` at `src/lora_lite/adapter.py:19-20`.
- Identity survives bf16 because `1.0` is exact.

### forward
Math: `ia3: y*g`; `ia3_ff: W(x*g)` at `src/lora_lite/variants/ia3.py:46-47` and `src/lora_lite/variants/ia3.py:63-64`.
Ref math: PEFT uses output gating for non-feedforward and input gating for feedforward.
Match? YES.

### links
- paper: OK
- peft ref: OK
- author ref: n/a
- offline snapshot drift: AMBIGUOUS

### verdict
CORRECT -- both IA3 regimes match PEFT’s mechanism.

## dora

### params
- `lora_A`: `(r, d_in)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/dora.py:36`.
- `lora_B`: `(d_out, r)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/dora.py:37`.
- `lora_m`: `(d_out,)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/dora.py:38-39`; matches PEFT magnitude vector.

### init / group_init
- Requires exact `nn.Linear`; initializes `m=||W||_row` in fp32 then casts to `lora_m.dtype` at `src/lora_lite/variants/dora.py:43-52`.
- GRADIENT FLOW at t=0: `lora_m` and `lora_B` get non-zero grad; `lora_A` gets zero grad because `B=0`.

### dtype
- `V = layer.weight + scale*BA` is computed in current tensor dtype at `src/lora_lite/variants/dora.py:59-61`; not fp32.
- PEFT detaches norm; this implementation intentionally differentiates through norm at `src/lora_lite/variants/dora.py:10-14`.

### forward
Math: `(m/||W+sBA||) * (y + s*xA^T B^T)` at `src/lora_lite/variants/dora.py:57-66`.
Ref math: `bias + (m/||V||)*(Wx + s*BAx)`; PEFT subtracts bias before scaling.
Match? NO for biased `nn.Linear`: this code scales `bias` because `y` includes it at `src/lora_lite/variants/dora.py:65-66`.

### links
- paper: OK
- peft ref: OK
- author ref: n/a
- offline snapshot drift: AMBIGUOUS

### verdict
BUGGY -- DoRA is wrong for linear layers with bias; bias is multiplied by `m/||V||`.

## hra

### params
- `lora_U`: `(r, d_in)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/hra.py:49-52`; transpose of PEFT’s `(d_in, r)` but mathematically equivalent.
- Rejects odd rank at `src/lora_lite/variants/hra.py:44-48`; stricter than PEFT warning, but documented.

### init / group_init
- Kaiming initializes half rows and repeats adjacent pairs at `src/lora_lite/variants/hra.py:56-65`.
- GRADIENT FLOW at t=0: repeated Householder pairs cancel to identity, but each row participates in the chain and gets non-zero gradient.

### dtype
- Householder math runs in adapter/input cast dtype via pre-hook at `src/lora_lite/adapter.py:23-28`.
- bf16 identity is near-exact for paired reflections, but norm/division in bf16 can introduce small non-bit-exactness at `src/lora_lite/variants/hra.py:73-77`.

### forward
Math: pre-hook applies `R x`, `R=Π_i(I-2u_i u_i^T/||u_i||^2)` at `src/lora_lite/variants/hra.py:69-78`.
Ref math: PEFT computes `W R` then `F.linear(x, WR)`.
Match? YES for linear layers.

### links
- paper: OK
- peft ref: DEAD/WRONG; current search finds `hra/model.py`, exact `hra/layer.py` URL did not fetch.
- author ref: OK
- offline snapshot drift: MAJOR/AMBIGUOUS for PEFT HRA URL path.

### verdict
CORRECT -- local mechanism matches the HRA reflection chain; link is stale.

## eva

### params
- `lora_A`: `(r, d_in)`, `cfg.dtype`, frozen persistent buffer at `src/lora_lite/variants/eva.py:50-52`; BUG vs PEFT LoRA `lora_A`, which remains a trainable Parameter after EVA initialization.
- `lora_B`: `(d_out, r)`, `cfg.dtype`, trainable Parameter at `src/lora_lite/variants/eva.py:53-54`.

### init / group_init
- `init()` no-op at `src/lora_lite/variants/eva.py:57-60`.
- `group_init()` requires calibration data, captures layer inputs, runs fp32 SVD, copies top right singular vectors into `lora_A` at `src/lora_lite/variants/eva.py:63-113`.
- GRADIENT FLOW at t=0: only `lora_B` gets non-zero grad; `lora_A` never trains because it is a buffer.

### dtype
- Captured activations are fp32 CPU at `src/lora_lite/variants/eva.py:78-79`; SVD is fp32 at `src/lora_lite/variants/eva.py:111`.
- `A` is cast to buffer dtype/device at `src/lora_lite/variants/eva.py:112`; bf16 loses PCA-vector precision.

### forward
Math: `y + (alpha/r) x A^T B^T` at `src/lora_lite/variants/eva.py:116-121`.
Ref math: LoRA forward after EVA copies PCA components into trainable `lora_A`; PEFT also supports rank redistribution/whitening.
Match? PARTIAL; forward equation matches, trainability does not.

### links
- paper: OK
- peft ref: OK
- author ref: n/a
- offline snapshot drift: AMBIGUOUS

### verdict
BUGGY -- `lora_A` is frozen as a buffer, and `adapter.load()` cannot load EVA because it calls `attach(model, cfg)` without calibration data at `src/lora_lite/adapter.py:132-135` while EVA requires it at `src/lora_lite/variants/eva.py:63-70`.

## antipasto

### params
- Buffers: `lora_U (d_out,r)`, `lora_S (r,)`, `lora_Vh (r,d_in)`, all `cfg.dtype`, frozen persistent buffers at `src/lora_lite/variants/antipasto.py:88-92`.
- Trainable: `lora_delta_s (r,)`, `lora_rot_T (n_blocks, bs(bs-1)/2)`, `cfg.dtype`, Parameters at `src/lora_lite/variants/antipasto.py:93-95`.
- Requires `r % block_size == 0` at `src/lora_lite/variants/antipasto.py:81-86`.

### init / group_init
- Requires exact `nn.Linear`, fp32 SVD of `W`, stores top-r factors, and mutates base to residual `W_res` at `src/lora_lite/variants/antipasto.py:99-116`.
- GRADIENT FLOW at t=0: `delta_s` gets non-zero grad; `rot_T` should get non-zero grad unless symmetry/data cancels.

### dtype
- SVD is fp32, but frozen factors are stored in `cfg.dtype` at `src/lora_lite/variants/antipasto.py:106-115`.
- Rotation is built in fp32 from `rot_T.float()` at `src/lora_lite/variants/antipasto.py:129`, then cast to `x.dtype` at `src/lora_lite/variants/antipasto.py:130`.

### forward
Math: `y_res + ((x Vh_eff^T) * (S+delta_s)) U_eff^T` at `src/lora_lite/variants/antipasto.py:132-145`.
Ref math: AntiPaSTO3 Flax convention `xW_res + ((xU_rot)*(S+alpha delta_s))Vh_rot`.
Match? PARTIAL; PyTorch basis mapping is coherent, but reference initializes `delta_s` and rotation params with small random non-zero values, while local uses exact zeros at `src/lora_lite/variants/antipasto.py:94-95`.

### links
- paper: n/a
- peft ref: n/a
- author ref: OK repo URL
- offline snapshot drift: AMBIGUOUS

### verdict
PARTIAL -- mechanism maps correctly, but initialization differs from the referenced AntiPaSTO3 code.

## summary

| variant | verdict | main issue |
|---|---:|---|
| lora | CORRECT | no mechanism bug |
| pissa | PARTIAL | scale parameterization differs unless `alpha=r` |
| delora | PARTIAL | documented init/gradient deviation |
| ia3 / ia3_ff | CORRECT | no mechanism bug |
| dora | BUGGY | bias is scaled incorrectly |
| hra | CORRECT | code OK; PEFT link stale |
| eva | BUGGY | `A` frozen; load path broken |
| antipasto | PARTIAL | zero init differs from reference random init |

MUST-FIX:
1. DoRA: handle bias like PEFT; do not multiply `base_layer.bias` by `m/||V||` (`src/lora_lite/variants/dora.py:65-66`).
2. EVA: make `lora_A` a trainable Parameter if matching PEFT, not a buffer (`src/lora_lite/variants/eva.py:50-52`).
3. EVA: fix `load()`/`group_init` interaction; current `load()` cannot attach EVA checkpoints without calibration data (`src/lora_lite/adapter.py:132-135`, `src/lora_lite/variants/eva.py:63-70`).

Nice-to-haves:
1. PiSSA: match PEFT’s `Sr /= scaling` parameterization for arbitrary `alpha/r`, or keep documenting `alpha=r`.
2. AntiPaSTO: decide whether exact identity zero init is intentional vs reference random `delta_s/rot` init.
3. Refresh stale GitHub refs, especially HRA and DeLoRA.

[tool] Searching for: site:github.com/huggingface/peft src/peft/tuners/hra/layer.py, site:github.com/huggingface/peft src/peft/tuners/delora/layer.py, site:github.com ExplainableML DeLoRA peft tuners delora.py (completed)
  kind: fetch
  input: site:github.com/huggingface/peft src/peft/tuners/hra/layer.py

[done] end_turn
