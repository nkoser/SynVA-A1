"""Sample from a trained Method D checkpoint."""
from __future__ import annotations
import os, sys, torch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, ROOT)

from methods.D_vq_transformer.model import VQVAE
from methods.C_fsq_ar.ar_prior import FSQARPrior
from models.vessel_conditioner import OstiumConditioner


def load(ckpt_path: str, device: torch.device):
    pl = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = pl["saved_args"]
    vq = VQVAE(
        input_dim=sa["input_dim"], cond_dim=sa["vessel_cond_dim"],
        hidden_dim=sa["hidden_dim"], num_tokens=sa["num_tokens"],
        code_dim=sa["code_dim"], num_codes=sa["num_codes"],
        encoder_blocks=sa["encoder_blocks"], decoder_blocks=sa["decoder_blocks"],
    ).to(device)
    vq.load_state_dict(pl["vqvae"])
    cond_net = OstiumConditioner(
        vessel_feat_dim=64, ostium_plane_dim=8,
        ostium_feat_dim=16, cond_out_dim=sa["vessel_cond_dim"],
    ).to(device)
    cond_net.load_state_dict(pl["conditioner"])
    ar = FSQARPrior(vocab_size=sa["num_codes"], num_tokens=sa["num_tokens"],
                    cond_dim=sa["vessel_cond_dim"], dim=sa["ar_dim"],
                    depth=sa["ar_depth"], heads=sa["ar_heads"]).to(device)
    ar.load_state_dict(pl["ar_prior"])
    return vq, cond_net, ar, pl, sa


@torch.no_grad()
def sample_ghd(vq, ar, cond, num_samples: int = 1, temperature: float = 1.0, top_k: int | None = None):
    cond_rep = cond.repeat_interleave(num_samples, dim=0)
    ids = ar.sample(cond_rep, temperature=temperature, top_k=top_k)
    return vq.decode_indices(ids, cond_rep)
