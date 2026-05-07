# Methods — Architecture Comparison

Four conditional generators for `p(GHD | ostium)`, each in its own folder.

| Folder | Architecture | Latent | Conditioning | Two-stage? |
|---|---|---|---|---|
| `A_pca_flow_matching/` | Flow matching (FiLM velocity net) over PCA-95 whitened space | Continuous (~K=80) | FiLM cond | No |
| `B_mog_prior_cvae/` | v8 ResNet CVAE with K-component MoG conditional prior | Continuous (D=64) | FiLM + cond-prior | No |
| `C_fsq_ar/` | FSQ-VAE (T=8 tokens, levels=8,8,5,5,5) + tiny GPT prior | Discrete (vocab=5000^T) | cond as additive prefix | Yes |
| `D_vq_transformer/` | VQ-VAE EMA (K=256, T=8) + tiny GPT prior | Discrete (vocab=256^T) | cond as additive prefix | Yes |

All methods:
- use the bigdata split `checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429/`
- default to `--no_vessel_pts` (only ostium params, since vessel-pts contributed 0% in ablation)
- save to `checkpoints/methods/<X>/<meta>/`

Single-seed launchers:
```bash
bash methods/A_pca_flow_matching/run.sh
bash methods/B_mog_prior_cvae/run.sh 1
bash methods/C_fsq_ar/run.sh 1
bash methods/D_vq_transformer/run.sh 1
```

Methods B/C/D accept a seed arg as `$1` for ensemble training (loop 1..5).
