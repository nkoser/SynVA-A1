# SynVA Method Code

This directory contains the SynVA Stage-1 generative models and wrappers that
plug into the reference `vessel-mesh-editing-master` pipeline copied at the
repository root.

Main files:

```text
methods/W_cond_ghd_vae/model.py
methods/W_cond_ghd_vae/train.py
methods/_common/stage3_surrogate_loss.py
tools/run_strict_reference_stage3_batch.py
tools/generate_and_stitch_reference.py
tools/run_w_variant_test_generation_dual.py
tools/fair_reference_stitch_seams.py
```

The paper-facing comparison uses:

```text
W_ref
W_stage3surrogate
W_stage3nearest
```

