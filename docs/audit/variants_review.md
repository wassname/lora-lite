[client] initialize (running)

[client] session/new (running)
I’ll use the `gsd` skill because this is a multi-step audit, then I’ll read the variant code, paper excerpts, logs, and refs only as needed. No code edits, just a verdict with quoted evidence.

[tool] Read SKILL.md (running)
  input: {"call_id":"call_ZJ8jUrIxVXwLtDtWUHoN5Nyw","process_id":"8344","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/med...
  files: /home/wassname/.claude/skills/gsd/SKILL.md

[tool] Read SKILL.md (completed)
  kind: read
  input: {"call_id":"call_ZJ8jUrIxVXwLtDtWUHoN5Nyw","process_id":"8344","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,220p' /home/wassname/.claude/skills/gsd/SKILL.md"],"cwd":"/med...
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
I’m treating the GSD “spec” as an in-memory checklist here because your request explicitly says no edits and the filesystem is read-only. I’m going to gather hard evidence first: line-numbered code, paper anchors, logs, and any reference snippets that resolve ambiguity.

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (running)
  input: {"call_id":"call_LrMg9wIgw4ohnwkqUAtJRrss","process_id":"31422","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite docs/papers docs/refs logs"],"cwd":"/media/was...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite

[tool] Read smoke.log (running)
  input: {"call_id":"call_UGl5ewNsDGBeS2Dhqvjea8gK","process_id":"66572","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,240p' logs/smoke.log"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/smoke.log

[tool] Read qwen_probe.log (running)
  input: {"call_id":"call_NVGdpn8BCjAn1aXHyrZGHVGE","process_id":"83314","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,240p' logs/qwen_probe.log"],"cwd":"/media/wassname/SGIronWolf...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/qwen_probe.log

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite (completed)
  kind: search
  input: {"call_id":"call_LrMg9wIgw4ohnwkqUAtJRrss","process_id":"31422","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg --files src/lora_lite docs/papers docs/refs logs"],"cwd":"/media/was...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/lora_lite
  output:
    docs/refs/peft_eva_finetuning.py
    src/lora_lite/target.py
    docs/refs/antipasto3_svd_adapter.py
    docs/refs/peft_eva.py
    docs/refs/baukit_nethook.py
    src/lora_lite/__init__.py
    src/lora_lite/variant.py
    src/lora_lite/config.py
    src/lora_lite/adapter.py
    src/lora_lite/variants/pissa.py
    src/lora_lite/variants/ia3.py
    src/lora_lite/variants/lora.py
    src/lora_lite/variants/dora.py
    src/lora_lite/variants/delora.py
    src/lora_lite/variants/__init__.py
    src/lora_lite/variants/hra.py

[tool] Read qwen_probe.log (completed)
  kind: read
  input: {"call_id":"call_NVGdpn8BCjAn1aXHyrZGHVGE","process_id":"83314","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,240p' logs/qwen_probe.log"],"cwd":"/media/wassname/SGIronWolf...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/qwen_probe.log
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    New task added (id 84).
    SHOULD: exact q_proj/v_proj layer-0 targets, lora-only grads, lossN<loss0, perturb>0, reload<tol. ELSE hook/target/train/save bug.
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11183.76it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10483.91it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11231.62it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11419.02it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11501.48it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10138.41it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11177.81it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10532.84it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10290.29it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11146.68it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10274.97it/s]
    Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10701.60it/s]
    variant  	  targets	  trainable	  id_err	  perturb	  loss0	  lossN	  drop%	   grad	   dθ	  reload	out
    lora     	        2	      20480	  0     	   0.2969	  5.25 	  2.509	  52.22	1.349  	5....

[tool] Read smoke.log (completed)
  kind: read
  input: {"call_id":"call_UGl5ewNsDGBeS2Dhqvjea8gK","process_id":"66572","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '1,240p' logs/smoke.log"],"cwd":"/media/wassname/SGIronWolf/proj...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/smoke.log
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/.venv/lib/python3.13/site-packages/bitsandbytes/autograd/_functions.py:123: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
      warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
    /media/wassname/SGIronWolf/projects5/2026/lora-lite/.venv/lib/python3.13/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
      torch._check_is_size(blocksize)
    
    === variant=lora dtype=torch.float32 ===
      attached 28 targets, trainable params=17408
      t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
      SHOULD: err<4.6e-06. PASS.
      save/load: max|y_loaded - y_adapt| = 0.000e+00
      SHOULD: err2<4.6e-06. PASS.
      loss[0]=1.0882  loss[-1]=1.0220  drop=6.1%
      SHOULD: drop>5%. PASS.
    
    === variant=pissa dtype=torch.float32 ===
      attached 28 targets, trainable params=17408
      t=0 identity: max|y_adapt - y_base| = 1.550e-06  (base scale 4.609e+00)
      SHOULD: err<2.3e-03. PASS.
      save/load: max|y_loaded - y_adapt| = 0.000e+00
      SHOULD: err2<2.3e-03. PASS.
      loss[0]=1.0823  loss[-1]=0.9583  drop=11.5%
      SHOULD: drop>5%. PASS.
    
    === variant=delora dtype=torch.float32 ===
      attached 28 targets, trainable params=17436
      t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
      SHOULD: err<4.6e-06. PASS.
    ... (54 more lines)

[tool] Read pissa.py (running)
  input: {"call_id":"call_PHAInlKqf4e79zhlDJiBvHfL","process_id":"17726","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py

[tool] Read lora.py (running)
  input: {"call_id":"call_wgFgKkJ4EWh2naGB2Dxzmxeq","process_id":"53887","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py

[tool] Read lora.py (completed)
  kind: read
  input: {"call_id":"call_wgFgKkJ4EWh2naGB2Dxzmxeq","process_id":"53887","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/lora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/lora.py
  output:
    1	"""Vanilla LoRA. Hu et al. 2021  https://arxiv.org/abs/2106.09685
         2	
         3	    h = W x + (alpha/r) B A x
         4	
         5	Identity at t=0 from B=0. Faithful to the paper.
         6	"""
         7	from einops import einsum
         8	from torch import nn
         9	import torch
        10	
        11	from ..variant import register, ParamSpec
        12	
        13	
        14	@register
        15	class LoRA:
        16	    name = "lora"
        17	
        18	    @staticmethod
        19	    def param_specs(d_in, d_out, cfg):
        20	        return {
        21	            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
        22	            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
        23	        }
        24	
        25	    @staticmethod
        26	    def init(layer: nn.Linear, cfg) -> None:
        27	        # B is zeros => delta=0 at t=0; identity invariant holds.
        28	        return
    ... (8 more lines)

[tool] Read pissa.py (completed)
  kind: read
  input: {"call_id":"call_PHAInlKqf4e79zhlDJiBvHfL","process_id":"17726","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/pissa.py"],"cwd":"/media/wassname/SGIronW...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/pissa.py
  output:
    1	"""PiSSA: top-r SVD of W into A,B; replace W with W_res = W - B@A.
         2	
         3	Meng et al. 2024  https://arxiv.org/abs/2404.02948
         4	W_eff(t=0) = W_res + B@A = W (numerically; bf16 round-trip not bit-exact).
         5	"""
         6	import torch
         7	from einops import einsum
         8	from torch import nn
         9	
        10	from ..variant import register, ParamSpec
        11	
        12	
        13	@register
        14	class PiSSA:
        15	    name = "pissa"
        16	
        17	    @staticmethod
        18	    def param_specs(d_in, d_out, cfg):
        19	        return {
        20	            "lora_A": ParamSpec((cfg.r, d_in), init="zeros", trainable=True),
        21	            "lora_B": ParamSpec((d_out, cfg.r), init="zeros", trainable=True),
        22	        }
        23	
        24	    @staticmethod
        25	    def init(layer: nn.Linear, cfg) -> None:
        26	        if type(layer) is not nn.Linear:
        27	            raise TypeError(
        28	                "PiSSA mutates layer.weight into W_res, so v1 only supports plain nn.Linear. "
    ... (26 more lines)

[tool] Read ia3.py (running)
  input: {"call_id":"call_vTvRjbwmBsdDiIngobGbh5cB","process_id":"83902","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py

[tool] Read dora.py (running)
  input: {"call_id":"call_Vw9lIGlnQoz5jeOy3IYrKUHS","process_id":"14313","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/dora.py

[tool] Read ia3.py (completed)
  kind: read
  input: {"call_id":"call_vTvRjbwmBsdDiIngobGbh5cB","process_id":"83902","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/ia3.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/ia3.py
  output:
    1	"""IA3-style output gating. Liu et al. 2022  https://arxiv.org/abs/2205.05638
         2	
         3	    y_new = y * g,    g initialized to 1  (identity at t=0)
         4	
         5	DEVIATION FROM PAPER:
         6	    The original IA3 gates only three positions per transformer block:
         7	        l_k * (k_proj output),  l_v * (v_proj output),  l_ff * (FFN intermediate after activation)
         8	    This implementation gates ANY linear layer the targeting system selects.
         9	    To match the paper exactly on a typical Llama/Qwen-style block, attach with:
        10	
        11	        cfg = LoraLiteConfig(
        12	            variant="ia3",
        13	            target_names=(r"\\.k_proj$", r"\\.v_proj$", r"\\.up_proj$"),
        14	            target_roles=(),
        15	        )
        16	
        17	    `up_proj` is the closest stand-in for "FFN intermediate" in gated-MLP blocks
        18	    (Llama uses gate * up; gating the up branch is the IA3-spirit choice).
        19	"""
        20	import torch
        21	from torch import nn
        22	
        23	from ..variant import register, ParamSpec
        24	
        25	
        26	@register
        27	class IA3:
        28	    name = "ia3"
    ... (12 more lines)

[tool] Read dora.py (completed)
  kind: read
  input: {"call_id":"call_Vw9lIGlnQoz5jeOy3IYrKUHS","process_id":"14313","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/dora.py"],"cwd":"/media/wassname/SGIronWo...
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
         9	"""
        10	import torch
        11	import torch.nn.functional as F
        12	from einops import einsum
        13	from torch import nn
        14	
        15	from ..variant import register, ParamSpec
        16	
        17	
        18	@register
        19	class DoRA:
        20	    name = "dora"
        21	
        22	    @staticmethod
        23	    def param_specs(d_in, d_out, cfg):
        24	        return {
        25	            "lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
        26	            "lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
        27	            # m is filled from ||W||_c during init(); shape (d_out,)
        28	            "lora_m": ParamSpec((d_out,), init="zeros", trainable=True),
    ... (27 more lines)

[tool] Read delora.py (running)
  input: {"call_id":"call_1QShJSCjExZn5oGZqc78uF7b","process_id":"51148","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py

[tool] Read adapter.py (running)
  input: {"call_id":"call_QOJFiy0Ooh70IhQEnPLRA2tX","process_id":"61139","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/adapter.py

[tool] Read hra.py (running)
  input: {"call_id":"call_mXp2R9UAQaI8I95Mc8fZh0Fm","process_id":"29257","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/hra.py

[tool] Read variant.py (running)
  input: {"call_id":"call_6XIUIUymL7F6gKF9QAvv5ljV","process_id":"69737","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variant.py

[tool] Read delora.py (completed)
  kind: read
  input: {"call_id":"call_1QShJSCjExZn5oGZqc78uF7b","process_id":"51148","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/delora.py"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/src/lora_lite/variants/delora.py
  output:
    1	"""DeLoRA: column-normalised A, B, scaled by lambda * ||W||_F / r.
         2	
         3	Bini et al. 2025  https://arxiv.org/abs/2503.18225
         4	
         5	Paper Eq. 8:    W' = W + (lambda * ||W||_F / r) B Xi A
         6	where Xi_{i,i} = 1 / (||b_i|| ||a_i||) makes each rank-1 component unit-norm.
         7	This is equivalent to row-normalising A and column-normalising B (each column of
         8	B and row of A has unit norm), so each rank-1 outer product b_i a_i^T has unit
         9	spectral norm -> the whole low-rank update is bounded.
        10	
        11	Identity at t=0: paper uses kaiming init for both A and B with `lambda` initialised
        12	to 0 (or small) so the effective delta starts near zero. We honour that:
        13	default lambda0 == 0 gives bit-identity; user can override via variant_kwargs.
        14	
        15	The frozen ||W||_F factor is captured once at init() into a buffer `lora_wnorm`.
        16	"""
        17	import torch
        18	import torch.nn.functional as F
        19	from einops import einsum
        20	from torch import nn
        21	
        22	from ..variant import register, ParamSpec
        23	
        24	
        25	@register
        26	class DeLoRA:
        27	    name = "delora"
        28	
    ... (32 more lines)

[tool] Read adapter.py (completed)
  kind: read
  input: {"call_id":"call_QOJFiy0Ooh70IhQEnPLRA2tX","process_id":"61139","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/adapter.py"],"cwd":"/media/wassname/SGIronWolf/pro...
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
    ... (86 more lines)

[tool] Read hra.py (completed)
  kind: read
  input: {"call_id":"call_mXp2R9UAQaI8I95Mc8fZh0Fm","process_id":"29257","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variants/hra.py"],"cwd":"/media/wassname/SGIronWol...
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
        12	Identity at t=0: `lora_gate` is initialized to 0 and gates each Householder
        13	vector, so the effective u_i starts at 0 -> H_i = I -> R = I -> y' = y.
        14	At training time the gate scales the active reflection direction.
        15	
        16	OMITTED: paper also adds an orthogonality regularizer
        17	    lambda * sum_i (u_i^T u_j)^2          (Eq. 6 / Sec. 3.3)
        18	which is a loss term, not a forward-pass change. Add it in your training loop if
        19	you want the regularized HRA variant.
        20	"""
        21	import torch
        22	from einops import einsum
        23	from torch import nn
        24	
        25	from ..variant import register, ParamSpec
        26	
        27	
        28	@register
    ... (27 more lines)

[tool] Read variant.py (completed)
  kind: read
  input: {"call_id":"call_6XIUIUymL7F6gKF9QAvv5ljV","process_id":"69737","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba src/lora_lite/variant.py"],"cwd":"/media/wassname/SGIronWolf/pro...
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
        15	
        16	    def make(self, dtype: torch.dtype, device) -> nn.Parameter:
        17	        t = torch.empty(self.shape, dtype=dtype, device=device)
        18	        if callable(self.init):
        19	            self.init(t)
        20	        elif self.init == "zeros":
        21	            t.zero_()
        22	        elif self.init == "ones":
        23	            t.fill_(1.0)
        24	        elif self.init == "kaiming":
        25	            # match nn.Linear default: kaiming_uniform_(a=sqrt(5))
        26	            nn.init.kaiming_uniform_(t, a=5 ** 0.5) if t.ndim >= 2 else t.normal_(0, 0.02)
        27	        else:
        28	            raise ValueError(f"unknown init: {self.init}")
    ... (28 more lines)
The code already shows several documented deviations, so I’m checking those against the extracted papers rather than relying on the docstrings. Next I’m pulling section/equation anchors from the paper text and logs with line numbers so the review can be evidence-heavy.

[tool] Search PiSSA|SVD|singular|residual|A =|B =|W_res|Eq|equation|principal|initialize|initial in pissa_*.txt (running)
  input: {"call_id":"call_E9ZDiQW4LYUvXfsbLERUR8VE","process_id":"40256","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"PiSSA|SVD|singular|residual|A =|B =|W_res|Eq|equation|principal...

[tool] Search IA3|l_k|l_v|l_ff|key|value|feed-forward|activation|element-wise|Eq|equation|rescale|learned vector in ia3_*.txt (running)
  input: {"call_id":"call_g8eWpsF9Wv2q0sr9uIOTKAXt","process_id":"6527","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"IA3|l_k|l_v|l_ff|key|value|feed-forward|activation|element-wise|...

[tool] Search LoRA|low-rank|A|B|alpha|rank|Eq|equation|ΔW|Delta|zero|Gaussian|Kaiming|initial in lora_*.txt (running)
  input: {"call_id":"call_bWnP0o9dE2sMNEN6N5a9w6Jp","process_id":"2585","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"LoRA|low-rank|A|B|alpha|rank|Eq|equation|ΔW|Delta|zero|Gaussian|...

[tool] Search DoRA|magnitude|direction|decompose|norm|V =|Eq|equation|initialize|column|row in dora_*.txt (running)
  input: {"call_id":"call_u0hqS6Uap4DQRdjRTfjZS35I","process_id":"15602","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"DoRA|magnitude|direction|decompose|norm|V =|Eq|equation|initial...

[tool] Search PiSSA|SVD|singular|residual|A =|B =|W_res|Eq|equation|principal|initialize|initial in pissa_*.txt (completed)
  kind: search
  input: {"call_id":"call_E9ZDiQW4LYUvXfsbLERUR8VE","process_id":"40256","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"PiSSA|SVD|singular|residual|A =|B =|W_res|Eq|equation|principal...
  output:
    binary file matches (found "\0" byte around offset 4333)

[tool] Search IA3|l_k|l_v|l_ff|key|value|feed-forward|activation|element-wise|Eq|equation|rescale|learned vector in ia3_*.txt (completed)
  kind: search
  input: {"call_id":"call_g8eWpsF9Wv2q0sr9uIOTKAXt","process_id":"6527","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"IA3|l_k|l_v|l_ff|key|value|feed-forward|activation|element-wise|...
  output:
    19:method called (IA)3 that scales activations by learned vectors, attaining stronger
    34:task with an entirely new set of parameter values, which can become impractical when ﬁne-tuning a
    43:∗Equal contribution.
    66:learned vectorslk,l v, andlﬀ which respectively rescale (via element-wise multiplication, visualized as
    67:⊙) the keys and values in attention mechanisms and the inner activations in position-wise feed-forward
    92:that multiplies intermediate activations by learned vectors. (IA)3 attains stronger performance than
    141:costs is to cache the key and value vectors for in-context examples. This is possible because decoder-
    142:only Transformer language models have a causal masking pattern, so the model’s activations for the
    144:tokens per in-context example would result in over 144 gigabytes of cached key and value vectors for
    145:the GPT-3 model (32 examples× 512 tokens× 96 layers× 12288 dmodel× 32 bits each for the key
    146:and value vectors). Separately, Min et al. [21] proposed ensemble ICL, where instead of using the
    155:proposed adding adapters [22–24], which are small trainable feed-forward networks inserted between
    160:tuning [29] concatenate learned continuous embeddings to the model’s input or activations to induce
    287:graph. A more convenient alternative is provided by methods that directly modify the activations of
    290:concatenating learned vectors to activation or embedding sequences and are therefore examples of
    291:activation-modifying PEFT methods that allow for mixed-task batches. However, as we will discuss
    296:As an alternative, we explored element-wise multiplication (i.e. rescaling) of the model’s activations
    297:against a learned vector. Speciﬁcally, we consider adaptation of the forml⊙x wherel∈ Rd is a
    298:learned task-speciﬁc vector,⊙ represents element-wise multiplication, andx∈ RT ×d is a length-T
    299:sequence of activations. We use “broadcasting notation” [46] so that the (i,j )th entry ofl⊙x isljxi,j.
    301:for each set of activations...

[tool] Search LoRA|low-rank|A|B|alpha|rank|Eq|equation|ΔW|Delta|zero|Gaussian|Kaiming|initial in lora_*.txt (completed)
  kind: search
  input: {"call_id":"call_bWnP0o9dE2sMNEN6N5a9w6Jp","process_id":"2585","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"LoRA|low-rank|A|B|alpha|rank|Eq|equation|ΔW|Delta|zero|Gaussian|...
  output:
    1:LORA: L OW-R ANK ADAPTATION OF LARGE LAN-
    2:GUAGE MODELS
    3:Edward Hu∗ Yelong Shen∗ Phillip Wallis Zeyuan Allen-Zhu
    10:ABSTRACT
    11:An important paradigm of natural language processing consists of large-scale pre-
    12:training on general domain data and adaptation to particular tasks or domains. As
    14:becomes less feasible. Using GPT-3 175B as an example – deploying indepen-
    15:dent instances of ﬁne-tuned models, each with 175B parameters, is prohibitively
    16:expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-
    17:trained model weights and injects trainable rank decomposition matrices into each
    19:rameters for downstream tasks. Compared to GPT-3 175B ﬁne-tuned with Adam,
    20:LoRA can reduce the number of trainable parameters by 10,000 times and the
    21:GPU memory requirement by 3 times. LoRA performs on-par or better than ﬁne-
    22:tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite hav-
    25:rank-deﬁciency in language model adaptation, which sheds light on the efﬁcacy of
    26:LoRA. We release a package that facilitates the integration of LoRA with PyTorch
    27:models and provide our implementations and model checkpoints for RoBERTa,
    28:DeBERTa, and GPT-2 athttps://github.com/microsoft/LoRA.
    46:tion. We only train A and B.
    52:parameters as in the original model. As larger models are trained
    54:GPT-2 (Radford et al., b) or RoBERTa large (Liu et al., 2019) to a
    55:critical deployment challenge for GPT-3 (Brown et al., 2020) with
    62:∗Equal contribution.
    64:1While GPT-3 175B achieves non-trivial performance with few-shot learning, ﬁne-tuning boosts its perfor-
    65:mance signiﬁcantly as shown in Appendix A.
    72:We take inspiration from Li et al. (2018a); Aghajanyan et al. (2020) which show that the learned
    74:change in weights during model adaptation also has a low “intrinsic rank”, leading to our proposed
    75:Low-Rank Adaptation (LoRA) approach. LoRA allows us to train some dense layers in a neural
    ... (519 more lines)

[tool] Search DoRA|magnitude|direction|decompose|norm|V =|Eq|equation|initialize|column|row in dora_*.txt (completed)
  kind: search
  input: {"call_id":"call_u0hqS6Uap4DQRdjRTfjZS35I","process_id":"15602","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"DoRA|magnitude|direction|decompose|norm|V =|Eq|equation|initial...
  output:
    1:DoRA: Weight-Decomposed Low-Rank Adaptation
    16:Rank Adaptation (DoRA). DoRA decomposes
    18:nitude and direction, for fine-tuning, specifically
    19:employing LoRA for directional updates to ef-
    21:rameters. By employing DoRA, we enhance
    24:ference overhead. DoRA consistently outper-
    30:NVlabs/DoRA.
    63:Figure 1. An overview of our proposed DoRA, which decomposes
    64:the pre-trained weight into magnitude and direction components
    66:direction component. Note that || · || c denotes the vector-wise
    67:norm of a matrix across each column vector.
    86:tially reparameterizes model weights into magnitude and di-
    90:DoRA: Weight-Decomposed Low-Rank Adaptation
    91:in magnitude and direction introduced by LoRA and FT.
    96:Low-Rank Adaptation (DoRA), which begins by decompos-
    97:ing the pre-trained weight into its magnitude and directional
    99:size of the directional component in terms of parameters,
    100:we exploit LoRA for the directional adaptation to enable
    104:closely resembling FT, we have validated DoRA across a
    107:experimental results show that DoRA consistently outper-
    114:• We introduce DoRA, a novel PEFT method that incor-
    121:• DoRA consistently surpasses LoRA on various tasks,
    183:trix A is initialized with uniform Kaiming distribution (He
    188:ditionally, based on Eq. (1), we can merge the learned ∆W
    191:DoRA: Weight-Decomposed Low-Rank Adaptation
    208:into magnitude and direction for accelerating optimization,
    211:arate components, magnitude and direction, to reveal the
    214:both magnitude and direction of the LoRA and FT weights
    ... (270 more lines)

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/papers (running)
  input: {"call_id":"call_WDgDsyoegpxmYzNxHapO8Fv5","process_id":"37661","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","ls docs/papers"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/lora-...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/papers

[tool] Search DeLoRA|lambda|Xi|Frobenius|normalize|norm|Eq|equation|initial|BXiA|rank-1|bounded|Kaiming in delora_*.txt (running)
  input: {"call_id":"call_kCjqhNBUefgB2d9cLgpLVQXu","process_id":"93391","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"DeLoRA|lambda|Xi|Frobenius|normalize|norm|Eq|equation|initial|B...

[tool] Run tr '\000' ' ' < docs/papers/pissa_2404.02948.txt | rg -n "PiSSA|SVD|singular|principal|residual|W_res|Equation|Eq|initialize|A =|B =|W\^res|Σ|U|V" (running)
  input: {"call_id":"call_IYEogQ1vzPNOesA5pz8BR3Rm","process_id":"70162","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/pissa_2404.02948.txt | rg -n \"PiSSA|SVD|s...

[tool] Search Householder|HRA|reflection|orthogonal|orthogonality|regular|Eq|equation|W'|R =|H_i|identity|initialize in hra_*.txt (running)
  input: {"call_id":"call_LyXBP8FEJOi8PqBIwdFguyLO","process_id":"39215","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"Householder|HRA|reflection|orthogonal|orthogonality|regular|Eq|...

[tool] List /media/wassname/SGIronWolf/projects5/2026/lora-lite/papers (completed)
  kind: search
  input: {"call_id":"call_WDgDsyoegpxmYzNxHapO8Fv5","process_id":"37661","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","ls docs/papers"],"cwd":"/media/wassname/SGIronWolf/projects5/2026/lora-...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/papers
  output:
    delora_2503.18225.pdf
    delora_2503.18225.txt
    dora_2402.09353.pdf
    dora_2402.09353.txt
    hra_2405.17484.pdf
    hra_2405.17484.txt
    ia3_2205.05638.pdf
    ia3_2205.05638.txt
    lora_2106.09685.pdf
    lora_2106.09685.txt
    pissa_2404.02948.pdf
    pissa_2404.02948.txt

[tool] Search DeLoRA|lambda|Xi|Frobenius|normalize|norm|Eq|equation|initial|BXiA|rank-1|bounded|Kaiming in delora_*.txt (completed)
  kind: search
  input: {"call_id":"call_kCjqhNBUefgB2d9cLgpLVQXu","process_id":"93391","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"DeLoRA|lambda|Xi|Frobenius|normalize|norm|Eq|equation|initial|B...
  output:
    15:bounded approaches, such as ETHER, provide greater robustness but are limited
    18:rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales
    20:DeLoRA effectively decouples the angular learning from the adaptation strength,
    23:tion tuning, we show that DeLoRA matches or surpasses performance of com-
    25:https://github.com/ExplainableML/DeLoRA.
    29:et al., 2023a;b; Radford et al., 2021; Rombach et al., 2022). However, the enormous size of these
    41:Therefore, we propose DeLoRA, an enhanced version of LoRA that introduces a boundary on the
    42:weight updates through normalization, decoupling the angular learning from the adaptation strength.
    44:and merging at inference time. We motivate DeLoRA from two distinct perspectives: as an extension
    45:of LoRA through the introduction of additional normalization, and as an evolution of ETHER by
    47:improvements over both LoRA and ETHER. Additionally, we validate the advantages of DeLoRA
    50:arXiv:2503.18225v2  [cs.LG]  19 May 2025
    55:LoRA DeLoRA
    67:norm
    72:method DeLoRA. In addition to the low-rank matrices B, A, we introduce a normalization Ξ and a
    75:mulations of LoRA and ETHER and derive a novel PEFT method, DeLoRA; (ii) we demonstrate
    76:DeLoRA enhanced robustness and decoupling compared to alternatives; (iii) we extensively ablate
    77:the formulation of DeLoRA by deriving it from both LoRA and ETHER; (iv) we evaluate DeLoRA
    85:describe how we derive our proposed DeLoRA method from both perspectives (Section 2.2), along
    105:avoiding catastrophic overwriting of the pretrained weights. In our proposed DeLoRA, we mitigate
    109:ments, Bini et al. (2024) propose to employ bounded transformations for finetuning, namely ETHER
    112:and ETHER+. ETHER (left side in Eq. (2)) and ETHER+ (right side) introduce multiplicative trans-
    120:vectors) are bounded in terms of their distance to the identity transformation, as per
    125:where the subscript F denotes the Frobenius norm. This upper bound on the transformation distanc...

[tool] Run tr '\000' ' ' < docs/papers/pissa_2404.02948.txt | rg -n "PiSSA|SVD|singular|principal|residual|W_res|Equation|Eq|initialize|A =|B =|W\^res|Σ|U|V" (completed)
  kind: execute
  input: {"call_id":"call_IYEogQ1vzPNOesA5pz8BR3Rm","process_id":"70162","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/pissa_2404.02948.txt | rg -n \"PiSSA|SVD|s...
  output:
    1:PiSSA: Principal Singular Values and Singular
    2:Vectors Adaptation of Large Language Models
    4:1Institute for Artificial Intelligence, Peking University
    5:2State Key Laboratory of General Artificial Intelligence, Peking University
    6:https://github.com/GraphPKU/PiSSA
    11:r ≪ min(m, n), A is initialized with Gaussian noise, and B with zeros. LoRA
    14:Singular values and Singular vectors Adaptation (PiSSA). PiSSA shares the same
    15:architecture as LoRA, but initializes the adaptor matrices A and B with the princi-
    17:residual matrix W res ∈ Rm×n which is frozen during fine-tuning. Compared to
    18:LoRA, PiSSA updates the principal components while freezing the “residual”
    20:periments of PiSSA and LoRA across 11 different models, ranging from 184M to
    21:70B, encompassing 5 NLG and 8 NLU tasks, reveal that PiSSA consistently out-
    23:Gemma-7B fine-tuned with PiSSA achieves an accuracy of 77.7%, surpassing
    24:LoRA’s 74.53% by 3.25%. Due to the same architecture, PiSSA is also compat-
    26:Compared to QLoRA, QPiSSA (PiSSA with 4-bit quantization) exhibits smaller
    28:QPiSSA attains an accuracy of 86.05%, exceeding the performance of QLoRA at
    29:81.73%. Leveraging a fast SVD technique, PiSSA can be initialized in only a few
    30:seconds, presenting a negligible cost for transitioning from LoRA to PiSSA.
    36:of a LLaMA 65B parameter model requires over 780 GB of GPU memory [ 10], and the VRAM
    62:𝑆[:𝑟]𝑟 (c) PiSSA
    63:Figure 1: The comparison among Full Fine-tuning, training with LoRA, and PiSSA. In this visualiza-
    66:LoRA to 4-bit, whereas QPiSSA quantizes the residual matrix in PiSSA.
    67:Table 1: Comparison of similarities and differences between PiSSA and LoRA. In this table,bold
    68:highlights the model’s primary component, while underlinedenotes the residual component.
    69:LoRA PiSSA
    71:A ∼ N (0, σ2) ∈ Rm×r A = U[:,:r] S1/2
    74:B = 0 ∈ Rr×n B = S1/2
    75:[:r,:r] V ⊤
    ... (426 more lines)

[tool] Search Householder|HRA|reflection|orthogonal|orthogonality|regular|Eq|equation|W'|R =|H_i|identity|initialize in hra_*.txt (completed)
  kind: search
  input: {"call_id":"call_LyXBP8FEJOi8PqBIwdFguyLO","process_id":"39215","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"Householder|HRA|reflection|orthogonal|orthogonality|regular|Eq|...
  output:
    binary file matches (found "\0" byte around offset 8064)

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | rg -n "Householder|HRA|reflection|orthogonal|orthogonality|regular|Equation|Eq|W|R =|H_i|identity|initialize|product" (running)
  input: {"call_id":"call_OMzCypXuxmB4HY1su5fUTB41","process_id":"87863","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | rg -n \"Householder|H...

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | rg -n "Householder|HRA|reflection|orthogonal|orthogonality|regular|Equation|Eq|W|R =|H_i|identity|initialize|product" (completed)
  kind: execute
  input: {"call_id":"call_OMzCypXuxmB4HY1su5fUTB41","process_id":"87863","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | rg -n \"Householder|H...
  output:
    2:Adaptation via Householder Reflection Adaptation
    8:While following different technical routes, both low-rank and orthogonal adaptation
    12:method based on Householder reflections. Given a pre-trained model, our method
    13:fine-tunes its layers by multiplying each frozen weight matrix with an orthogonal
    14:matrix constructed by a chain of learnable Householder reflections (HRs). This
    15:HR-based orthogonal fine-tuning is equivalent to an adaptive low-rank adaptation.
    16:Moreover, we show that the orthogonality of the reflection planes corresponding
    17:to the HRs impacts the model capacity and regularity. The analysis motivates us
    18:to regularize the orthogonality of the HRs, leading to different implementations
    19:of the proposed Householder reflection adaptation (HRA) method. Compared
    20:with state-of-the-art methods, HRA achieves superior performance with fewer
    23:DaShenZi721/HRA, and the method has been merged into the PEFT package.
    38:in the inference phase. Currently, given a parameter matrix of a pre-trained model, i.e., W ∈ Rdout×d,
    45:<latexit sha1_base64="MobDUxOzzDZfhHAOBLBO1va0R4E=">AAAB83icbVDLSsNAFL2pr1pfVZduBovgqiQircuiG5cV7AOaUCbTSTt0MgnzEErob7hxoYhbf8adf+OkzUJbDwwczrmXe+aEKWdKu+63U9rY3NreKe9W9vYPDo+qxyddlRhJaIckPJH9ECvKmaAdzTSn/VRSHIec9sLpXe73nqhULBGPepbSIMZjwSJGsLaS78dYT8IoM/OhHFZrbt1dAK0TryA1KNAeVr/8UUJMTIUmHCs18NxUBxmWmhFO5xXfKJpiMsVjOrBU4JiqIFtknqMLq4xQlEj7hEYL9fdGhmOlZnFoJ/OMatXLxf+8gdHRTZAxkRpNBVkeigxHOkF5AWjEJCWazyzBRDKbFZEJlphoW1PFluCtfnmddK/qXqPeeLiutW6LOspwBudwCR40oQX30IYOEEjhGV7hzTHOi/PufCxHS06xcwp/4Hz+AIvhkgk=</latexit>
    47:<latexit sha1_base64="NtkTar0wxbOJ9f/2GlH11RAPyrY=">AAAB83icbVDLSsNAFL2pr1pfVZduBovgqiRFqsuiG5cV7AOaUCbTSTt0MgnzEErob7hxoYhbf8adf+OkzUJbDwwczrmXe+aEKWdKu+63U9rY3NreKe9W9vYPDo+qxyddlRhJaIckPJH9ECvKmaAdzTSn/VRSHIec9sLpXe73nqhULBGPepbSIMZjwSJGsLaS78dYT8IoM/NhY1ituXV3AbROvILUoEB7WP3yRwkxMRWacKzUwHNTHWRYakY4nVd8o2iKyRSP6cBSgWOqgmyReY4urDJCUSLtExot1N8bGY6VmsWhncwzqlUvF//zBkZHN0HGRGo0FWR5KDIc6QTlBaARk5RoPrMEE8lsVkQmWGKibU0V...

[tool] Read dora_2402.09353.txt (running)
  input: {"call_id":"call_eZ8F71QoWNQkvORpG7wnzPKP","process_id":"73985","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '220,250p' docs/papers/dora_2402.09353.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/dora_2402.09353.txt

[tool] Read delora_2503.18225.txt (running)
  input: {"call_id":"call_dRseeZXlFuJtlcV6O6B7Pp6e","process_id":"96401","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '282,300p' docs/papers/delora_2503.18225.txt"],"cwd":"/media/was...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/delora_2503.18225.txt

[tool] Read ia3_2205.05638.txt (running)
  input: {"call_id":"call_bQPmoF4wuY4LJjDU4wKFTGb0","process_id":"31176","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '296,312p' docs/papers/ia3_2205.05638.txt"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/ia3_2205.05638.txt

[tool] Read lora_2106.09685.txt (running)
  input: {"call_id":"call_UdRYpT88AHSMm73Ft3sb0vdK","process_id":"42170","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '190,210p' docs/papers/lora_2106.09685.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/lora_2106.09685.txt

[tool] Read dora_2402.09353.txt (completed)
  kind: read
  input: {"call_id":"call_eZ8F71QoWNQkvORpG7wnzPKP","process_id":"73985","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '220,250p' docs/papers/dora_2402.09353.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/dora_2402.09353.txt
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    = ||W ||c
    W
    ||W ||c
    (2)
    where m ∈ R1×k is the magnitude vector, V ∈ Rd×k is the
    directional matrix, with || · ||c being the vector-wise norm of
    a matrix across each column. This decomposition ensures
    that each column of V /||V ||c remains a unit vector, and the
    corresponding scalar in m defines the magnitude of each
    vector.
    For our weight decomposition analysis, we select the VL-
    BART model fine-tuned on four image-text tasks as outlined
    in (Sung et al., 2022) for a case study. Following (Sung et al.,
    2022), which applies LoRA only to the query/value weight
    matrix in the self-attention module. We decompose the pre-
    trained weight W0, the full fine-tuned weight WFT, and the
    merged LoRA weight WLoRA of query/value weight matrix
    using Eq. (2). The magnitude and directional variations
    between W0 and WFT can be defined as follows:
    ∆M t
    FT =
    Pk
    n=1 |mn,t
    FT − mn
    0 |
    k (3)
    ∆Dt
    ... (4 more lines)

[tool] Read delora_2503.18225.txt (completed)
  kind: read
  input: {"call_id":"call_dRseeZXlFuJtlcV6O6B7Pp6e","process_id":"96401","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '282,300p' docs/papers/delora_2503.18225.txt"],"cwd":"/media/was...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/delora_2503.18225.txt
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    DeLoRA formulation. Summarizing, our proposed DeLoRA finetuning method consists in learn-
    ing a normalized low-rank matrix BΞA and a scale λ, updating the pretrained weights as in
    W + λ∥ ¯W ∥
    r BΞA
    ⊺
    x + b (15)
    This formulation inherently constrains the learnable finetuning updates in aλ∥ ¯W ∥-sized ball, where
    ¯W is the norm of the pretrained weights, achieving a decoupling of the transformation strength from
    the angular learning.
    In more detail, the key components are:
    • Normalization: Ξ is a r-dimensional diagonal matrix that normalizes LoRA’s inner low-
    dimensional bottleneck (Eq. (5)), bounding the Frobenius norm of BΞA to r (Eq. (6)).
    • Scaling Factors: (i) 1/r is used to remove the rank dependency on the boundary dimensionality,
    (ii) ∥ ¯W ∥ to make the weight updates proportional to the pretrained weights, and (iii)λ to control
    the adaptation strength and allow for a layer-specific boundary adaptation (Eq. (7))
    • Initialization: Pretrained initialization follows by merging to the pretrained weights a frozen
    copy of the initialized finetuning adaptation matrices (Eq. (9)).
    DoRA vs DeLoRA discussion. DoRA (Liu et al., 2024a), similarly to our work, addresses fine-
    tuning targeting the decoupling of angular and magnitude components, by using a formulation that

[tool] Read ia3_2205.05638.txt (completed)
  kind: read
  input: {"call_id":"call_bQPmoF4wuY4LJjDU4wKFTGb0","process_id":"31176","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '296,312p' docs/papers/ia3_2205.05638.txt"],"cwd":"/media/wassna...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/ia3_2205.05638.txt
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    As an alternative, we explored element-wise multiplication (i.e. rescaling) of the model’s activations
    against a learned vector. Speciﬁcally, we consider adaptation of the forml⊙x wherel∈ Rd is a
    learned task-speciﬁc vector,⊙ represents element-wise multiplication, andx∈ RT ×d is a length-T
    sequence of activations. We use “broadcasting notation” [46] so that the (i,j )th entry ofl⊙x isljxi,j.
    In preliminary experiments, we found it was not necessary to introduce a learned rescaling vector
    for each set of activations in the Transformer model. Instead, we found it was sufﬁcient to introduce
    rescaling vectors on the keys and values in self-attention and encoder-decoder attention mechanisms
    and on the intermediate activation of the position-wise feed-forward networks. Speciﬁcally, using
    the notation from Vaswani et al. [33], we introduce three learned vectors lk∈ Rdk,l v∈ Rdv, and
    lﬀ∈ Rdff , which are introduced into the attention mechanisms as:
    softmax
    (Q(lk⊙K T )√dk
    )
    (lv⊙V )
    and in the position-wise feed-forward networks as (lﬀ⊙γ(W1x))W2, whereγ is the feed-forward
    network nonlinearity. We introduce a separate set oflk,l v, andlﬀ vectors in each Transformer layer
    block. This adds a total ofL(dk +dv +dﬀ ) new parameters for aL-layer-block Transformer encoder

[tool] Read lora_2106.09685.txt (completed)
  kind: read
  input: {"call_id":"call_UdRYpT88AHSMm73Ft3sb0vdK","process_id":"42170","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '190,210p' docs/papers/lora_2106.09685.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/lora_2106.09685.txt
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    We describe the simple design of LoRA and its practical beneﬁts. The principles outlined here apply
    to any dense layers in deep learning models, though we only focus on certain weights in Transformer
    language models in our experiments as the motivating use case.
    4.1 L OW-R ANK -PARAMETRIZED UPDATE MATRICES
    A neural network contains many dense layers which perform matrix multiplication. The weight
    matrices in these layers typically have full-rank. When adapting to a speciﬁc task, Aghajanyan et al.
    (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
    learn efﬁciently despite a random projection to a smaller subspace. Inspired by this, we hypothe-
    size the updates to the weights also have a low “intrinsic rank” during adaptation. For a pre-trained
    weight matrix W0∈ Rd×k, we constrain its update by representing the latter with a low-rank de-
    composition W0 + ∆W = W0 + BA, where B∈ Rd×r, A∈ Rr×k, and the rank r≪ min(d, k).
    During training, W0 is frozen and does not receive gradient updates, whileA and B contain trainable
    parameters. Note both W0 and ∆W = BA are multiplied with the same input, and their respective
    output vectors are summed coordinate-wise. For h = W0x, our modiﬁed forward pass yields:
    h = W0x + ∆W x = W0x + BAx (3)
    We illustrate our reparametrization in Figure 1. We use a random Gaussian initialization for A and
    zero for B, so ∆W = BA is zero at the beginning of training. We then scale ∆W x by α
    r , where α
    is a constant in r. When optimizing with Adam, tuning α is roughly the same as tuning the learning
    rate if we scale the initialization appropriately. As a result, we simply set α to the ﬁrst r we try
    and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary

[tool] Run tr '\000' ' ' < docs/papers/pissa_2404.02948.txt | sed -n '224,250p' (running)
  input: {"call_id":"call_XuGUouRx8rVTGidWifmXxTHK","process_id":"10872","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/pissa_2404.02948.txt | sed -n '224,250p'"]...

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '239,266p' (running)
  input: {"call_id":"call_YUdE51vQ2pQzxJJ0de05qxBT","process_id":"84041","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '239,266p'"],"...

[tool] Read dora_2402.09353.txt (running)
  input: {"call_id":"call_PHQxP0pE81Ax1UO7FOrwQoyg","process_id":"65692","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '370,391p' docs/papers/dora_2402.09353.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/dora_2402.09353.txt

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '417,445p' (running)
  input: {"call_id":"call_RoD8Og7c3808QGTVmwmNrSUR","process_id":"43418","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '417,445p'"],"...

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '239,266p' (completed)
  kind: execute
  input: {"call_id":"call_YUdE51vQ2pQzxJJ0de05qxBT","process_id":"84041","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '239,266p'"],"...
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    a specular reflection hyperplane, denoted as H. For any x ∈ Rd, Hx corresponds to reflecting x
    across the hyperplane H, which reverses the component of x that is orthogonal to the hyperplane.
    Because H is an orthogonal matrix, it is natural for us to implement orthogonal adaptation based on
    it — we can treat H as an adapter and multiply it with the weight matrix of the pre-trained model.
    Moreover, since the set of all d × d orthogonal matrices, denoted as Od×d, satisfies all the axioms of
    a group which is a compact Lie group of dimension d(d − 1)/2, the product of orthogonal matrices is
    also an orthogonal matrix [1]. Therefore, we can enhance the capacity of the adapter by constructing
    a chain of r trainable Householder reflections, leading to our HRA method. As shown in Figure 1a,
    given a weight matrix W ∈ Rdout×d and an input x ∈ Rd, the forward step of HRA is
    z = W H(r)x = W
    Yr
    i=1
    Hi
    
    x = W
    Yr
    i=1
    (I − 2uiu⊤
    i )
    
    x, with {ui ∈ Sd−1}r
    i=1. (1)
    Although (1) involves the chained product of r + 1 dense matrices, we can leverage the struc-
    ture of Householder reflection to simplify the computation. Let x(0) = x and x(j+1) =
    (I − 2ur−ju⊤
    r−j)x(j) for j = 0, ..., r − 1. We implement (1) by the following two steps:
    1) x(j+1) = x(j) − 2⟨ur−j, x(j)⟩ur−j, for j = 0, ..., r − 1. 2) z = W x(r). (2)
    ... (1 more lines)

[tool] Run tr '\000' ' ' < docs/papers/pissa_2404.02948.txt | sed -n '224,250p' (completed)
  kind: execute
  input: {"call_id":"call_XuGUouRx8rVTGidWifmXxTHK","process_id":"10872","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/pissa_2404.02948.txt | sed -n '224,250p'"]...
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    3 PiSSA: Principal Singular Values and Singular Vectors Adaptation
    This section formally presents our Principal Singular values and Singular vectors Adaptation method.
    PiSSA computes the singular value decomposition (SVD) of matrices W within the self-attention
    and multilayer perceptron (MLP) layers. The (economy size) SVD of a matrix W ∈ Rm×n is
    given by W = U SV ⊤, where U ∈ Rm×min(m,n), V ∈ Rn×min(m,n) are the singular vectors with
    orthonormal columns, and V ⊤ is the transpose of V . S = diag(s) ∈ Rmin(m,n)×min(m,n), where the
    operation diag(s) transforms s to a diagonal matrix S, and s ∈ Rmin(m,n)
    ≥0 represents the singular
    values arranged in descending order. When the top r singular values s[:r] are significantly larger
    than the remaining singular values s[r:], we denote the intrinsic rank of W as r. Consequently, S,
    along with U and V , can be divided into two groups: the principal singular values and vectors—
    {U[:,:r], S[:r,:r], V[:,:r]}, and the residual singular values and vectors— {U[:,r:], S[r:,r:], V[:,r:]}, where
    the matrix slicing notations are the same as those in PyTorch and [: r] denotes the first r dimensions.
    The principal singular values and vectors are utilized to initialize the injected adapter consisting of
    A ∈ Rm×r and B ∈ Rr×n:
    A = U[:,:r] S1/2
    [:r,:r] ∈ Rm×r, (2)
    B = S1/2
    [:r,:r] V ⊤
    [:,:r] ∈ Rr×n. (3)
    The residual singular values and vectors are used to build the residual matrix which is frozen during
    fine-tuning:
    W res = U[:,r:] S[r:,r:] V ⊤
    [:,r:] ∈ Rm×n. (4)
    As indicated by Equation 5, the integration of AB with the residual matrix also preserves the full
    capability of the pre-trained model in the beginning of fine-tuning:
    Y = XW = X(W res + W pri) = X(W res + AB). (5)

[tool] Read dora_2402.09353.txt (completed)
  kind: read
  input: {"call_id":"call_PHQxP0pE81Ax1UO7FOrwQoyg","process_id":"65692","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","sed -n '370,391p' docs/papers/dora_2402.09353.txt"],"cwd":"/media/wassn...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/docs/papers/dora_2402.09353.txt
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    & Kingma, 2016) lies in their training approaches. Weight
    normalization trains both components from scratch, mak-
    ing the method sensitive to different initializations. Con-
    versely, DoRA avoids such initialization concerns since both
    components begin with pre-trained weights. We initialize
    DoRA with pre-trained weight W0 as outlined in Eq. (2),
    where m = ||W0||c and V = W0 after initialization. We
    then keep V frozen and m a trainable vector. The directional
    component is then updated through LoRA. DoRA can be
    formulated similar to Eq. (1) as:
    W ′ = m V + ∆V
    ||V + ∆V ||c
    = m W0 + BA
    ||W0 + BA||c
    (5)
    where ∆V is the incremental directional update learned
    by multiplying two low-rank matrices B and A, and the
    underlined parameters denote the trainable parameters. The
    matrices B ∈ Rd×r and A ∈ Rr×k are initialized in line
    with LoRA’s strategy to ensure thatW ′ equals W0 before
    the finetuning. Furthermore, DoRA can be merged with the
    pre-trained weight before inference, thereby not introducing

[tool] Run tr '\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '417,445p' (completed)
  kind: execute
  input: {"call_id":"call_RoD8Og7c3808QGTVmwmNrSUR","process_id":"43418","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","tr '\\000' ' ' < docs/papers/hra_2405.17484.txt | sed -n '417,445p'"],"...
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
    Motivated by the above analysis, we can implement HRA with an orthogonality regularizer. Typically,
    given a pre-trained model M, we can adapt L weight matrices of the model based on a dataset D by
    solving the following optimization problem:
    min{U (l)
    r }L
    l=1
    Loss(D; {U (l)
    r }L
    l=1) + λ
    XL
    l=1
    ∥Ir − (U (l)
    r )⊤U (l)
    r ∥2
    F , (6)
    where U (l)
    r denotes the parameters of HRA for the l-th weight matrix. In (6), the first term denotes
    the loss function, while the second term is the proposed regularizer that encourages the orthogonality
    of all U (l)
    r ’s, whose significance is controlled byλ > 0. Because it does not change the forward step
    of HRA, this regularizer only increases the adaptation cost slightly.
    As shown in Figure 1a, by controlling the strength of the orthogonality regularizer, we can achieve a
    trade-off between the model capacity and regularity. When λ = 0, the feasible domain of Ur is the
    set of column-normalized matrices, and accordingly, the model capacity is maximized. In contrast,
    when λ → ∞, the feasible domain of Ur is the set of orthogonal matrices (i.e., Od×r), leading to
    the strongest regularity. When λ = ∞, we implement a strictly-orthogonal HRA based on Gram-
    Schmidt (GS) orthogonalization. For each layer’s HRA adapter, we initialize its parameter matrix as
    ... (2 more lines)

[tool] Search === variant=|t=0 identity|save/load|loss\[0\]|Linear8bitLt|Linear4bit|ALL PASS|variant  \t in smoke.log (running)
  input: {"call_id":"call_Cq3AKAdomLL3nfVH6sdVw5Og","process_id":"52954","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"=== variant=|t=0 identity|save/load|loss\\[0\\]|Linear8bitLt|Li...

[tool] Search ^(variant|lora|pissa|delora|ia3|dora|hra)|ALL QWEN in qwen_probe.log (running)
  input: {"call_id":"call_UybdTwAPyZghPblXqP44hq58","process_id":"59149","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"^(variant|lora|pissa|delora|ia3|dora|hra)|ALL QWEN\" logs/qwen_...

[tool] Read smoke.log (running)
  input: {"call_id":"call_7n1MahUfgq21v3LwRKTww9W4","process_id":"42160","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba logs/smoke.log | sed -n '1,120p'"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/smoke.log

[tool] Read qwen_probe.log (running)
  input: {"call_id":"call_w3TWFELhGDScRzTyE8wD25oI","process_id":"41693","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba logs/qwen_probe.log | sed -n '1,80p'"],"cwd":"/media/wassname/SG...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/qwen_probe.log

[tool] Search === variant=|t=0 identity|save/load|loss\[0\]|Linear8bitLt|Linear4bit|ALL PASS|variant  \t in smoke.log (completed)
  kind: search
  input: {"call_id":"call_Cq3AKAdomLL3nfVH6sdVw5Og","process_id":"52954","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"=== variant=|t=0 identity|save/load|loss\\[0\\]|Linear8bitLt|Li...
  output:
    6:=== variant=lora dtype=torch.float32 ===
    8:  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
    10:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    12:  loss[0]=1.0882  loss[-1]=1.0220  drop=6.1%
    15:=== variant=pissa dtype=torch.float32 ===
    17:  t=0 identity: max|y_adapt - y_base| = 1.550e-06  (base scale 4.609e+00)
    19:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    21:  loss[0]=1.0823  loss[-1]=0.9583  drop=11.5%
    24:=== variant=delora dtype=torch.float32 ===
    26:  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
    28:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    30:  loss[0]=1.0909  loss[-1]=0.0277  drop=97.5%
    33:=== variant=ia3 dtype=torch.float32 ===
    35:  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
    37:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    39:  loss[0]=1.0823  loss[-1]=0.1228  drop=88.7%
    42:=== variant=dora dtype=torch.float32 ===
    44:  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
    46:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    48:  loss[0]=1.0882  loss[-1]=0.3996  drop=63.3%
    51:=== variant=hra dtype=torch.float32 ===
    53:  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
    55:  save/load: max|y_loaded - y_adapt| = 0.000e+00
    57:  loss[0]=1.0882  loss[-1]=0.0305  drop=97.2%
    68:  Linear8bitLt   lora  : identity_err=0.000e+00 grad_nonzero=True
    69:  Linear8bitLt   delora: identity_err=0.000e+00 grad_nonzero=True
    70:  Linear8bitLt   ia3   : identity_err=0.000e+00 grad_nonzero=True
    71:  Linear8bitLt   hra   : identity_err=0.000e+00 grad_nonzero=True
    ... (9 more lines)

[tool] Search ^(variant|lora|pissa|delora|ia3|dora|hra)|ALL QWEN in qwen_probe.log (completed)
  kind: search
  input: {"call_id":"call_UybdTwAPyZghPblXqP44hq58","process_id":"59149","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"^(variant|lora|pissa|delora|ia3|dora|hra)|ALL QWEN\" logs/qwen_...
  output:
    15:variant  	  targets	  trainable	  id_err	  perturb	  loss0	  lossN	  drop%	   grad	   dθ	  reload	out
    16:lora     	        2	      20480	  0     	   0.2969	  5.25 	  2.509	  52.22	1.349  	5.205	       0	outputs/qwen_train_probe/lora_adapter.pt
    17:pissa    	        2	      20480	  0.3125	   0.75  	  5.25 	  2.958	  43.66	6.124  	5.909	       0	outputs/qwen_train_probe/pissa_adapter.pt
    18:delora   	        2	      20484	  0.7188	   0.4688	  5.236	  3.364	  35.75	2.143  	9.086	       0	outputs/qwen_train_probe/delora_adapter.pt
    19:ia3      	        2	       3072	  0     	   0.375 	  5.25 	  4.473	  14.79	0.463  	5.926	       0	outputs/qwen_train_probe/ia3_adapter.pt
    20:dora     	        2	      23552	  0     	   0.4062	  5.25 	  2.189	  58.3 	2.052  	7.734	       0	outputs/qwen_train_probe/dora_adapter.pt
    21:hra      	        2	       8194	  0     	   0.3594	  5.25 	  4.136	  21.23	0.09808	4.49 	       0	outputs/qwen_train_probe/hra_adapter.pt
    22:ALL QWEN PROBES PASS

[tool] Read smoke.log (completed)
  kind: read
  input: {"call_id":"call_7n1MahUfgq21v3LwRKTww9W4","process_id":"42160","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba logs/smoke.log | sed -n '1,120p'"],"cwd":"/media/wassname/SGIron...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/smoke.log
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
         1	/media/wassname/SGIronWolf/projects5/2026/lora-lite/.venv/lib/python3.13/site-packages/bitsandbytes/autograd/_functions.py:123: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
         2	  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
         3	/media/wassname/SGIronWolf/projects5/2026/lora-lite/.venv/lib/python3.13/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
         4	  torch._check_is_size(blocksize)
         5	
         6	=== variant=lora dtype=torch.float32 ===
         7	  attached 28 targets, trainable params=17408
         8	  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
         9	  SHOULD: err<4.6e-06. PASS.
        10	  save/load: max|y_loaded - y_adapt| = 0.000e+00
        11	  SHOULD: err2<4.6e-06. PASS.
        12	  loss[0]=1.0882  loss[-1]=1.0220  drop=6.1%
        13	  SHOULD: drop>5%. PASS.
        14	
        15	=== variant=pissa dtype=torch.float32 ===
        16	  attached 28 targets, trainable params=17408
        17	  t=0 identity: max|y_adapt - y_base| = 1.550e-06  (base scale 4.609e+00)
        18	  SHOULD: err<2.3e-03. PASS.
        19	  save/load: max|y_loaded - y_adapt| = 0.000e+00
        20	  SHOULD: err2<2.3e-03. PASS.
        21	  loss[0]=1.0823  loss[-1]=0.9583  drop=11.5%
        22	  SHOULD: drop>5%. PASS.
        23	
        24	=== variant=delora dtype=torch.float32 ===
        25	  attached 28 targets, trainable params=17436
        26	  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
        27	  SHOULD: err<4.6e-06. PASS.
    ... (54 more lines)

[tool] Read qwen_probe.log (completed)
  kind: read
  input: {"call_id":"call_w3TWFELhGDScRzTyE8wD25oI","process_id":"41693","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba logs/qwen_probe.log | sed -n '1,80p'"],"cwd":"/media/wassname/SG...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/logs/qwen_probe.log
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
         1	New task added (id 84).
         2	SHOULD: exact q_proj/v_proj layer-0 targets, lora-only grads, lossN<loss0, perturb>0, reload<tol. ELSE hook/target/train/save bug.
         3	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11183.76it/s]
         4	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10483.91it/s]
         5	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11231.62it/s]
         6	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11419.02it/s]
         7	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11501.48it/s]
         8	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10138.41it/s]
         9	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11177.81it/s]
        10	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10532.84it/s]
        11	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10290.29it/s]
        12	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 11146.68it/s]
        13	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10274.97it/s]
        14	Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 10701.60it/s]
        15	variant  	  targets	  trainable	  id_err	  perturb	  loss0	  lossN	  drop%	   grad...
One important pattern has emerged: the toy smoke and the Qwen probe do not agree for PiSSA and DeLoRA identity behavior. I’m checking the probe script/config now because that difference affects whether the log is a pass or a paper-faithfulness warning.

[tool] Search qwen|qwen_train_probe|LoraLiteConfig|lambda0|alpha|variant in . (running)
  input: {"call_id":"call_icH4JA6mT0tbZ9MkdbCU47eS","process_id":"97197","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"qwen|qwen_train_probe|LoraLiteConfig|lambda0|alpha|variant\" -S...

[tool] Search qwen|qwen_train_probe|LoraLiteConfig|lambda0|alpha|variant in . (completed)
  kind: search
  input: {"call_id":"call_icH4JA6mT0tbZ9MkdbCU47eS","process_id":"97197","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"qwen|qwen_train_probe|LoraLiteConfig|lambda0|alpha|variant\" -S...
  output:
    ./README.md:19:cfg = ll.LoraLiteConfig(variant="lora", r=8, alpha=16, dtype=torch.bfloat16)
    ./README.md:35:just qwen-probe  # Qwen/Qwen3-0.6B train/save-load probe
    ./README.md:57:Useful `LoraLiteConfig` fields:
    ./README.md:78:See [docs/developer_guide.md](docs/developer_guide.md) for the variant API, data-calibrated init, and adapter roadmap.
    ./docs/spec/20260426_lora_lite_plan.md:7:The core bet is that adapter variants should own the relationship between `(x, layer.weight, layer.lora_*)` and the layer output, while the library only handles targeting, parameter attachment, hooks, and save/load.
    ./docs/spec/20260426_lora_lite_plan.md:26:- Data-calibrated variants use `group_init(model, targets, cfg, calibration_data)`; dataloaders stay out of `cfg` so checkpoints are serializable.
    ./docs/spec/20260426_lora_lite_plan.md:32:| `LoraLiteConfig` | done | `src/lora_lite/config.py` |
    ./docs/spec/20260426_lora_lite_plan.md:33:| Variant registry + `ParamSpec` | done | `src/lora_lite/variant.py` |
    ./docs/spec/20260426_lora_lite_plan.md:36:| LoRA | done | `src/lora_lite/variants/lora.py` |
    ./docs/spec/20260426_lora_lite_plan.md:37:| PiSSA | done, fp-only | `src/lora_lite/variants/pissa.py` |
    ./docs/spec/20260426_lora_lite_plan.md:38:| DeLoRA | done | `src/lora_lite/variants/delora.py` |
    ./docs/spec/20260426_lora_lite_plan.md:39:| IA3 | done | `src/lora_lite/variants/ia3.py` |
    ./docs/spec/20260426_lora_lite_plan.md:40:| DoRA | done, fp-only | `src/lora_lite/variants/dora.py` |
    ./docs/spec/20260426_lora_lite_plan.md:41:| HRA | done | `src/lora_lite/variants/hra.py` (output-side Householder, hook-only -> bnb-compatible) |
    ./docs/spec/20260426_lora_lite_plan.md:73:- Pytest coverage for LoRA, PiSSA, and DeLoRA correctness invariants.
    ./docs/spec/20260426_lora_lite_plan.md:75:- A real `Qwen/Qwen3-0.6B` probe that trains each current variant on layer-0 `q_proj` and `v_proj`.
    ./docs/spec/20260426_lora_lite_plan.md:91:| R4: current variants train on tiny task | 28 TinyModel targets; ...

[tool] Read qwen_train_probe.py (running)
  input: {"call_id":"call_VBAAqoYfZcyMHtv4vJKvZDyq","process_id":"87533","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba scripts/qwen_train_probe.py | sed -n '20,45p'"],"cwd":"/media/wa...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py

[tool] Read qwen_train_probe.py (completed)
  kind: read
  input: {"call_id":"call_VBAAqoYfZcyMHtv4vJKvZDyq","process_id":"87533","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","nl -ba scripts/qwen_train_probe.py | sed -n '20,45p'"],"cwd":"/media/wa...
  files: /media/wassname/SGIronWolf/projects5/2026/lora-lite/scripts/qwen_train_probe.py
  output:
    sed: /tmp/.mount_vscodiOHg5sd/lib/x86_64-linux-gnu/libselinux.so.1: no version information available (required by sed)
        20	
        21	
        22	def cfg_for_variant(variant: str, dtype: torch.dtype, r: int, alpha: float) -> ll.LoraLiteConfig:
        23	    return ll.LoraLiteConfig(
        24	        variant=variant,
        25	        r=r,
        26	        alpha=r if variant == "pissa" else alpha,
        27	        dtype=dtype,
        28	        target_roles=(),
        29	        target_names=(r"model\.layers\.0\.self_attn\.(q_proj|v_proj)$",),
        30	        layers=(0,),
        31	        variant_kwargs={"lambda0": 0.1} if variant == "delora" else {},
        32	    )
        33	
        34	
        35	def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        36	    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
        37	
        38	
        39	def assert_only_lora_trainable(model: torch.nn.Module) -> None:
        40	    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        41	    assert trainable
        42	    assert all("lora_" in name for name in trainable), trainable[:20]
        43	
        44	
        45	def assert_no_base_grads(model: torch.nn.Module) -> None:

[tool] Search dtype in qwen_train_probe.py (running)
  input: {"call_id":"call_DKgY6gkmUZdWaTwYoZS4fkZq","process_id":"82966","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"dtype\" scripts/qwen_train_probe.py"],"cwd":"/media/wassname/SG...

[tool] Search dtype in qwen_train_probe.py (completed)
  kind: search
  input: {"call_id":"call_DKgY6gkmUZdWaTwYoZS4fkZq","process_id":"82966","turn_id":"019dc96e-e7cd-79f2-b04d-65a1c4d26637","command":["/usr/bin/zsh","-lc","rg -n \"dtype\" scripts/qwen_train_probe.py"],"cwd":"/media/wassname/SG...
  output:
    22:def cfg_for_variant(variant: str, dtype: torch.dtype, r: int, alpha: float) -> ll.LoraLiteConfig:
    27:        dtype=dtype,
    74:def load_model(model_id: str, dtype: torch.dtype, device: str):
    75:    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    80:def run_variant(args, variant: str, input_ids: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype):
    81:    model = load_model(args.model, dtype, args.device)
    83:    cfg = cfg_for_variant(variant, dtype, args.r, args.alpha)
    147:    loaded_model = load_model(args.model, dtype, args.device)
    183:    parser.add_argument("--torch-dtype", default="bfloat16")
    196:    dtype = getattr(torch, args.torch_dtype)
    202:    rows = [run_variant(args, variant, input_ids, labels, dtype) for variant in args.variants]
# Per-Variant Paper-Faithfulness Audit

## lora

1. **Paper claim summary**  
LoRA (`docs/papers/lora_2106.09685.txt` §4.1, eq.3): freeze `W0`, train low-rank `B,A`, and compute `h = W0x + BAx`; initialize `A` random and `B` zero so the adapter is identity at training start, then scale by `alpha/r`.

2. **What our code does** — `src/lora_lite/variants/lora.py:19`, `src/lora_lite/variants/lora.py:31`

```python
"lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
"lora_B": ParamSpec((d_out, cfg.r), init="zeros",  trainable=True),
scale = cfg.alpha / cfg.r
h = einsum(x, layer.lora_A, "... i, r i -> ... r")
return y + scale * delta
```

3. **Match? Y**  
Faithful. Kaiming is not literally “Gaussian”, but it is the standard LoRA-style nonzero `A`, zero `B` identity initialization and the scaling matches.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:6-12`:

```text
=== variant=lora dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0882  loss[-1]=1.0220  drop=6.1%
```

bnb rows from `logs/smoke.log:68,74`:

```text
Linear8bitLt   lora  : identity_err=0.000e+00 grad_nonzero=True
Linear4bit     lora  : identity_err=0.000e+00 grad_nonzero=True
```

Qwen row from `logs/qwen_probe.log:16`:

```text
lora     	        2	      20480	  0     	   0.2969	  5.25 	  2.509	  52.22	1.349  	5.205	       0	outputs/qwen_train_probe/lora_adapter.pt
```

Consistent with paper: identity at t=0, nonzero perturb after training, loss drops, reload exact.

5. **Bugs / concerns**  
No paper-faithfulness bugs found. Missing test: no explicit check that `A` itself receives zero gradient on the first step when `B=0`, then receives gradient after `B` moves, but this is expected LoRA behavior.

6. **Confidence**  
High — simple formula, direct implementation, logs exercise identity, gradient, training, bnb, and reload.

---

## pissa

1. **Paper claim summary**  
PiSSA (`docs/papers/pissa_2404.02948.txt` §3, eqs.2-5): SVD-decompose `W = U S V^T`, initialize the adapter from the top-`r` singular values/vectors, freeze the residual `W_res`, and preserve the pretrained function initially via `Y = X(W_res + AB)`.

2. **What our code does** — `src/lora_lite/variants/pissa.py:25`, `src/lora_lite/variants/pissa.py:31`

```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
B = (Ur * sqrtS).to(cfg.dtype)
A = (sqrtS[:, None] * Vhr).to(cfg.dtype)
layer.lora_B.data.copy_(B)
layer.weight.data.copy_((W - scale * BA).to(layer.weight.dtype))
```

3. **Match? Partial**  
Matches the SVD/residual idea only when `alpha == r` so `scale == 1`. This deviation is documented in the code comment at `src/lora_lite/variants/pissa.py:43`, but not in the top docstring. The implementation rejects bnb layers instead of implementing QPiSSA; that is documented in code and is acceptable for a small v1, but not the quantized PiSSA extension.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:15-21`:

```text
=== variant=pissa dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 1.550e-06  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0823  loss[-1]=0.9583  drop=11.5%
```

bnb rows from `logs/smoke.log:72,78`:

```text
Linear8bitLt   pissa : fail-loud OK (TypeError)
Linear4bit     pissa : fail-loud OK (TypeError)
```

Qwen row from `logs/qwen_probe.log:17`:

```text
pissa    	        2	      20480	  0.3125	   0.75  	  5.25 	  2.958	  43.66	6.124  	5.909	       0	outputs/qwen_train_probe/pissa_adapter.pt
```

Mixed. FP32 toy identity is consistent with paper. Qwen has `id_err=0.3125`, despite `scripts/qwen_train_probe.py:26` forcing `alpha=r`; likely bf16 residual/SVD roundoff, but the log itself is not an identity pass in paper terms. The probe’s pass condition accepts this.

5. **Bugs / concerns**  
Potentially material bf16 initialization error on real weights: `id_err=0.3125` is too large to treat as “paper identity” without a relative tolerance or logit-scale context. Save/load assumes loading into the same unmodified base model, because the residual is recomputed by `attach()` rather than saved. Missing test: PiSSA should explicitly check `alpha != r` fails or warns, since otherwise initial reconstruction is not faithful.

6. **Confidence**  
Medium — core math is right in FP32, but Qwen identity evidence is concerning and scaling fidelity depends on config.

---

## dora

1. **Paper claim summary**  
DoRA (`docs/papers/dora_2402.09353.txt` §4.1, eq.5): decompose weight into magnitude and direction, initialize `m = ||W0||c`, `V = W0`, freeze `V`, and learn magnitude plus LoRA directional update `W' = m (W0 + BA) / ||W0 + BA||c`; LoRA init makes `W' = W0` before fine-tuning.

2. **What our code does** — `src/lora_lite/variants/dora.py:23`, `src/lora_lite/variants/dora.py:32`, `src/lora_lite/variants/dora.py:44`

```python
"lora_m": ParamSpec((d_out,), init="zeros", trainable=True),
col_norm = W.norm(dim=1).to(layer.lora_m.dtype)
BA = einsum(layer.lora_B, layer.lora_A, "o r, r i -> o i")
V = layer.weight + scale * BA
return (layer.lora_m / v_norm) * combined
```

3. **Match? Y**  
Faithful for plain `nn.Linear`, modulo PyTorch’s row-oriented weight convention. The implementation explicitly documents and enforces no bnb support.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:42-48`:

```text
=== variant=dora dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0882  loss[-1]=0.3996  drop=63.3%
```

bnb rows from `logs/smoke.log:73,79`:

```text
Linear8bitLt   dora  : fail-loud OK (TypeError)
Linear4bit     dora  : fail-loud OK (TypeError)
```

Qwen row from `logs/qwen_probe.log:20`:

```text
dora     	        2	      23552	  0     	   0.4062	  5.25 	  2.189	  58.3 	2.052  	7.734	       0	outputs/qwen_train_probe/dora_adapter.pt
```

Consistent with paper: identity at init, learns, reloads.

5. **Bugs / concerns**  
No obvious gradient-flow bug. Concern: paper discusses a cost-saving / gradient adjustment around the denominator; this code differentiates through `v_norm`. I would not call that wrong for eq.5, but it may differ from the final reported DoRA variant. Missing test: verify `lora_m` equals the correct orientation-specific norm for a nonsquare linear layer.

6. **Confidence**  
High — formula and initialization line up, and both toy and Qwen evidence support it.

---

## ia3

1. **Paper claim summary**  
IA3 (`docs/papers/ia3_2205.05638.txt` §3): learn vectors that elementwise rescale activations, specifically keys, values, and FFN intermediate activations: `lk ⊙ K`, `lv ⊙ V`, and `(lff ⊙ gamma(W1x))W2`.

2. **What our code does** — `src/lora_lite/variants/ia3.py:31`, `src/lora_lite/variants/ia3.py:39`

```python
return {"lora_g": ParamSpec((d_out,), init="ones", trainable=True)}
return y * layer.lora_g
```

3. **Match? Partial**  
The operation is IA3-style output gating, but target semantics are not paper-faithful by default. The docstring explicitly documents the deviation at `src/lora_lite/variants/ia3.py:5-18`. Qwen probe targets `q_proj/v_proj`, while paper IA3 would use `k_proj/v_proj` plus FFN intermediate.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:33-39`:

```text
=== variant=ia3 dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0823  loss[-1]=0.1228  drop=88.7%
```

bnb rows from `logs/smoke.log:70,76`:

```text
Linear8bitLt   ia3   : identity_err=0.000e+00 grad_nonzero=True
Linear4bit     ia3   : identity_err=0.000e+00 grad_nonzero=True
```

Qwen row from `logs/qwen_probe.log:19`:

```text
ia3      	        2	       3072	  0     	   0.375 	  5.25 	  4.473	  14.79	0.463  	5.926	       0	outputs/qwen_train_probe/ia3_adapter.pt
```

Numbers are consistent with identity-at-init activation scaling, but the smoke/probe do not validate paper-correct IA3 placement.

5. **Bugs / concerns**  
Main issue is target placement, not arithmetic. Missing test: attach IA3 with the recommended `k_proj/v_proj/up_proj` pattern and assert those exact targets, not `q_proj/v_proj`.

6. **Confidence**  
Medium — implementation is correct for a documented IA3-like general gate, not full paper IA3.

---

## hra

1. **Paper claim summary**  
HRA (`docs/papers/hra_2405.17484.txt` §3.1, eqs.1-2): adapt a frozen matrix as `z = W (prod_i H_i) x`, where each Householder reflection is `H_i = I - 2 u_i u_i^T` with unit vectors. §3.4, eq.6 adds an orthogonality regularizer on the reflection vectors.

2. **What our code does** — `src/lora_lite/variants/hra.py:33`, `src/lora_lite/variants/hra.py:46`

```python
"lora_U": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
"lora_gate": ParamSpec((), init="zeros", trainable=True),
sq = (u * u).sum().clamp_min(1e-12)
Rx = Rx - coeff.unsqueeze(-1) * u
return x + layer.lora_gate * (Rx - x)
```

3. **Match? Partial**  
The Householder chain is recognizable and the omitted regularizer is documented in the docstring. But the scalar zero gate is not the paper’s forward: for `gate != 1`, `x + gate*(Rx-x)` is not a product of orthogonal Householder reflections. The identity-at-init deviation is documented, but it changes gradient flow.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:51-57`:

```text
=== variant=hra dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0882  loss[-1]=0.0305  drop=97.2%
```

bnb rows from `logs/smoke.log:71,77`:

```text
Linear8bitLt   hra   : identity_err=0.000e+00 grad_nonzero=True
Linear4bit     hra   : identity_err=2.447e-04 grad_nonzero=True
```

Qwen row from `logs/qwen_probe.log:21`:

```text
hra      	        2	       8194	  0     	   0.3594	  5.25 	  4.136	  21.23	0.09808	4.49 	       0	outputs/qwen_train_probe/hra_adapter.pt
```

Identity and learning pass. But `grad_nonzero=True` is insufficient: with `lora_gate=0`, gradients into `lora_U` are initially zero; the first update can only move the gate.

5. **Bugs / concerns**  
Gradient-flow concern: at initialization, `return x + gate*(Rx-x)` makes `d output / d U` proportional to `gate`, so `lora_U` has zero gradient when `lora_gate=0`. This is not caught by the logs. Paper regularizer is omitted and documented. The bnb 4-bit identity error `2.447e-04` passes a loose smoke tolerance but is not exact identity.

6. **Confidence**  
Medium-Low — smoke says it trains, but the zero-gate formulation is materially different from paper HRA and creates an initial dead-`U` step.

---

## delora

1. **Paper claim summary**  
DeLoRA (`docs/papers/delora_2503.18225.txt` §2.2, eq.15): learn normalized low-rank `BΞA` and scale `lambda * ||W|| / r`, decoupling angular learning from adaptation strength. The paper says initialization preserves the pretrained model by merging a frozen copy of the initialized adaptation matrices, not by simply zeroing the scale.

2. **What our code does** — `src/lora_lite/variants/delora.py:30`, `src/lora_lite/variants/delora.py:43`, `src/lora_lite/variants/delora.py:52`

```python
"lora_A": ParamSpec((cfg.r, d_in), init="kaiming", trainable=True),
"lora_B": ParamSpec((d_out, cfg.r), init="kaiming", trainable=True),
"lora_lambda": ParamSpec((), init=lambda t: t.fill_(lam0), trainable=True)
A = F.normalize(layer.lora_A, dim=1)
scale = layer.lora_lambda * layer.lora_wnorm / cfg.r
```

3. **Match? Partial**  
The normalized update and `||W||/r` scaling match eq.15. The initialization does not match the paper’s Eq.9-style pretrained-preserving frozen-copy construction; instead it defaults `lambda0=0` for identity. This is partially documented in the docstring, but the docstring understates the paper’s frozen-copy initialization.

4. **Smoke evidence**  
Toy row from `logs/smoke.log:24-30`:

```text
=== variant=delora dtype=torch.float32 ===
  t=0 identity: max|y_adapt - y_base| = 0.000e+00  (base scale 4.609e+00)
  save/load: max|y_loaded - y_adapt| = 0.000e+00
  loss[0]=1.0909  loss[-1]=0.0277  drop=97.5%
```

bnb rows from `logs/smoke.log:69,75`:

```text
Linear8bitLt   delora: identity_err=0.000e+00 grad_nonzero=True
Linear4bit     delora: identity_err=0.000e+00 grad_nonzero=True
```

Qwen row from `logs/qwen_probe.log:18`:

```text
delora   	        2	      20484	  0.7188	   0.4688	  5.236	  3.364	  35.75	2.143  	9.086	       0	outputs/qwen_train_probe/delora_adapter.pt
```

Not consistent with the stated expectation that DeLoRA should be identity at t=0: Qwen uses `variant_kwargs={"lambda0": 0.1}` in `scripts/qwen_train_probe.py:31`, so `id_err=0.7188` is expected from this implementation but not a paper-faithful pretrained initialization.

5. **Bugs / concerns**  
Major gradient-flow/design concern: with default `lambda0=0`, `A` and `B` receive zero gradient initially because the update is multiplied by zero; only `lambda` moves on the first step. With nonzero `lambda0`, training gradients flow, but identity is broken. The paper’s frozen-copy initialization is the way to get both nonzero scale and pretrained identity; this implementation lacks it. Missing tests: explicit per-parameter gradient check at `lambda0=0`, and identity check for the nonzero-lambda training config.

6. **Confidence**  
Medium — eq.15 forward is mostly present, but initialization is not paper-faithful and Qwen evidence exposes the tradeoff.

---

# Aggregate

| variant | paper match | smoke pass | qwen pass | bugs found | confidence |
|---|---|---:|---:|---|---|
| lora | Y | Y | Y | none material | High |
| pissa | Partial | Y | Partial | bf16/Qwen init error; config-sensitive `alpha==r`; residual not saved | Medium |
| dora | Y | Y | Y | possible denominator-gradient variant mismatch | High |
| ia3 | Partial | Y | Partial | target placement not paper IA3; missing exact-placement test | Medium |
| hra | Partial | Y | Partial | zero gate gives initial zero grad to `U`; not orthogonal for gate != 1 | Medium-Low |
| delora | Partial | Y | N/Partial | no Eq.9 frozen-copy init; zero-lambda dead A/B first step; nonzero-lambda breaks identity | Medium |

# Biggest Risks

- **Initialization vs gradient-flow tradeoffs are being hidden by coarse tests**: HRA’s `U` and DeLoRA’s `A/B` can be initially dead while `grad_nonzero=True` still passes.
- **Qwen probe pass criteria do not enforce paper identity**: PiSSA and DeLoRA show large `id_err` rows but the log still says pass.
- **Target semantics are under-tested**: IA3’s paper-specific `k/v/ffn` placement is documented as a deviation, but the real-model probe uses `q/v`, so it does not validate IA3 faithfulness.

[done] end_turn
