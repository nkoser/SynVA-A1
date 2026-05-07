"""Sample from a trained Method C checkpoint."""
from __future__ import annotations
import os, sys, torch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, ROOT)

from methods.C_fsq_ar.model import FSQVAE
from methods.C_fsq_ar.ar_prior import FSQARPrior
from models.vessel_conditioner import OstiumConditioner
from train_vessel_flow_matching import build_conditioner


def load(ckpt_path: str, device: torch.device):
    pl = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = pl["saved_args"]
    fsq = FSQVAE(
        input_dim=sa["input_dim"], cond_dim=sa["vessel_cond_dim"],
        hidden_dim=sa["hidden_dim"], num_tokens=sa["num_tokens"],
        levels=sa["levels"], encoder_blocks=sa["encoder_blocks"],
        decoder_blocks=sa["decoder_blocks"],
    ).to(device)
    fsq.load_state_dict(pl["fsqvae"])
    cond_net = build_conditioner(sa, device)
    cond_net.load_state_dict(pl["conditioner"])
    vocab = 1
    for L in sa["levels"]: vocab *= L
    ar = FSQARPrior(vocab_size=vocab, num_tokens=sa["num_tokens"],
                    cond_dim=sa["vessel_cond_dim"], dim=sa["ar_dim"],
                    depth=sa["ar_depth"], heads=sa["ar_heads"]).to(device)
    ar.load_state_dict(pl["ar_prior"])
    return fsq, cond_net, ar, pl, sa


@torch.no_grad()
def sample_ghd(fsq, ar, cond, num_samples: int = 1, temperature: float = 1.0, top_k: int | None = None):
    """cond: [B, cond_dim] -> ghd recon [B*num_samples, input_dim] (in normalized space)."""
    B = cond.size(0)
    cond_rep = cond.repeat_interleave(num_samples, dim=0)
    ids = ar.sample(cond_rep, temperature=temperature, top_k=top_k)
    z_q = fsq.quant.decode_indices(ids)  # [B*S, T, D_tok]
    return fsq.decode(z_q, cond_rep)
