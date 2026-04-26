# Per-variant paper-faithfulness audit for lora-lite

You are reviewing a small from-scratch PEFT library (`lora-lite`) that re-implements
6 LoRA variants. Your job: independent paper-vs-implementation sign-off for each.

## Inputs available locally

- Code: `src/lora_lite/variants/{lora,pissa,dora,ia3,hra,delora}.py`
- Adapter base + plumbing: `src/lora_lite/{adapter.py,target.py,variant.py,config.py}`
- Papers (extracted text): `docs/papers/{lora,pissa,dora,ia3,hra,delora}_*.txt`
- Smoke log (toy + bnb): `logs/smoke.log`
- Real-model probe log (Qwen0.6B, 16 SGD steps): `logs/qwen_probe.log`
- Reference implementations (peft / antipasto3 / baukit): `docs/refs/*.py`

## What I want from you (per variant, all 6)

For each of `lora, pissa, dora, ia3, hra, delora` produce a section with:

1. **Paper claim summary (1-3 sentences)** — cite paper file + section/eq number.
   E.g. "PiSSA (docs/papers/pissa_2404.02948.txt §3.1, eq.4): A,B = top-r SVD of W,
   W_res = W - BA; trains A,B with W_res frozen."

2. **What our code does** — point to the function and key lines in
   `src/lora_lite/variants/<v>.py`. Quote ≤5 lines.

3. **Match? Y / Partial / N** — explicit verdict. If Partial, state the deviation
   and whether it is documented in the variant's docstring.

4. **Smoke evidence** — quote the exact row from `logs/smoke.log` (toy + bnb)
   and the row from `logs/qwen_probe.log`. State whether the numbers are
   consistent with paper expectations (e.g. PiSSA should have nonzero perturb at
   t=0 because B@A reconstructs W; LoRA/HRA/IA3/DeLoRA should be identity at t=0).

5. **Bugs / concerns** — anything actually wrong, especially:
   - Gradient flow issues
   - Wrong normalization / scaling
   - Wrong initialization (e.g. PiSSA without SVD, HRA without orthogonality)
   - Missing or wrong save/load handling
   - Numerical issues (dtype, in-place ops on grad-required tensors)

6. **Confidence** — High / Medium / Low, with one-line reason.

## Final aggregate

After the 6 sections, produce a Markdown table:

| variant | paper match | smoke pass | qwen pass | bugs found | confidence |

And a 3-bullet "biggest risks" summary.

## Rules

- Be skeptical. The previous audit found IA3, HRA, DeLoRA bugs that had been
  declared "OK". Assume nothing.
- If the smoke log does not include a check that you'd want to see, flag it as
  a missing test — don't infer correctness from absence.
- Quote evidence; do not paraphrase code.
- Use file links: `src/lora_lite/variants/lora.py:42` style.
- Do NOT edit code. Output is a verdict only.
- If you cannot determine something from the available files, say so explicitly
  rather than guessing.

Write the full review to stdout. I will redirect to a file.
