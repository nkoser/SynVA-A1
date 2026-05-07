# Method C — FSQ-VAE + Autoregressive Transformer Prior

**Idea**: Encode each GHD vector to T discrete tokens via Finite Scalar Quantization (Mentzer et al. 2023). Each token's coordinates are independently rounded to a small integer grid `levels=[8,8,5,5,5]`, giving a per-token vocabulary of `8·8·5·5·5 = 5000`. A tiny GPT prior models `p(z_1, …, z_T | cond)` autoregressively.

**Why FSQ over VQ-VAE for small data (~400 cases)**:
- No codebook collapse (no learned codebook)
- No commitment loss / EMA / dead-code resets
- Fewer hyperparameters
- Comparable downstream quality at this scale

**Two stages**:
1. FSQ-VAE: train encoder + decoder with reconstruction MSE only. Conditioner co-trained.
2. AR Transformer: freeze stage 1, encode train tokens, fit `p(tokens | cond)` with cross-entropy.

**Files**:
- [model.py](model.py) — `FSQuantizer`, `FSQVAE`
- [ar_prior.py](ar_prior.py) — `FSQARPrior` (small causal transformer, cond as additive prefix)
- [train.py](train.py) — both stages, single checkpoint
- [sample.py](sample.py) — load + sample

**Train**:
```bash
bash methods/C_fsq_ar/run.sh
```
