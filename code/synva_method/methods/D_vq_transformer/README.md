# Method D — VQ-VAE + Autoregressive Transformer

**Idea**: Like Method C but with a learned vector-quantization codebook (van den Oord et al. 2017) of `K=256` entries of dim `D_code=32`, EMA updates (Razavi et al. 2019), commitment loss (β=0.25), and dead-code reset every 50 epochs (replaces unused codes with random encoder outputs).

**Risk vs. Method C**: With only ~400 train cases, codebook collapse is plausible (most of `K=256` codes go unused). Dead-code reset mitigates this but does not eliminate the risk. FSQ (Method C) avoids the issue entirely; we still include VQ for completeness and as an upper bound on representational capacity per token.

**Files**:
- [model.py](model.py) — `VectorQuantizerEMA`, `VQVAE`
- [train.py](train.py) — both stages (reuses [ar_prior.py](../C_fsq_ar/ar_prior.py))
- [sample.py](sample.py) — load + sample

**Train**:
```bash
bash methods/D_vq_transformer/run.sh
```

**Watch**: stage-1 log prints `perp` (codebook perplexity). If perp stays < 30 the codebook is collapsing — reduce `--num_codes`, increase `--commitment_beta`, or prefer Method C.
