# Method A — PCA95 + Whitened Flow Matching

**Idea**: Replace the Gaussian VAE prior with a continuous normalizing flow (rectified flow / flow matching) trained in a whitened PCA-95% space. The flow models a fully non-Gaussian conditional distribution `p(z | ostium)` and decoder is the linear PCA inverse — so the architecture cannot collapse to mean.

**Reuses**:
- `train_vessel_pca_flow_matching.py` (existing trainer, extended with `--pca_var`)
- `eval_vessel_pca_flow_matching.py`
- `models/vessel_aware_flow_matching.py` (FiLM velocity network)

**Train**:
```bash
bash methods/A_pca_flow_matching/run.sh
```
