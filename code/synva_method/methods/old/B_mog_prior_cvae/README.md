# Method B — Mixture-of-Gaussians Conditional Prior CVAE

**Idea**: Replace the single-Gaussian conditional prior `p(z|c) = N(μ(c), σ(c))` of the v8 CVAE with a K-component MoG `p(z|c) = Σ π_k(c) N(μ_k(c), σ_k(c))`. The encoder, decoder, and FiLM injection stay identical.

Multimodality (e.g. "60% spherical, 30% lobed, 10% asymmetric" per ostium) is now representable explicitly, mitigating mode collapse without architectural overhead.

**ELBO**: KL is estimated by Monte-Carlo with one z-sample per case using `log q(z|x,c) - logsumexp_k(log π_k + log N(z; μ_k, σ_k))`.

**Files**:
- [model.py](model.py) — `VesselAwareCVAEMoGPrior` (subclass of v8) + `mog_kl_mc`
- Reuses the existing trainer with `--model_type v8_mog --mog_components K`

**Train**:
```bash
bash methods/B_mog_prior_cvae/run.sh
```
