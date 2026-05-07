#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj


class MeshAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Direct vertex-only overfit autoencoder for sanity checking Stage-1 data.")
    parser.add_argument("--ghd-root", type=Path, default=root / "checkpoint-v2" / "ghd_fitting")
    parser.add_argument("--cases", type=str, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=root / "checkpoint-v2" / "direct_vertex_ae_inspect")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_case_mesh(case_root: Path, device: torch.device):
    mesh_path = case_root / "vanilla" / "viz" / "warped_epoch_02999.obj"
    verts, faces, _ = load_obj(str(mesh_path))
    return verts.to(device), faces.verts_idx.to(device), mesh_path.name


def render_pair(output_path: Path, case_name: str, input_verts, recon_verts, faces, rmse: float, dpi: int):
    input_np = input_verts.detach().cpu().numpy()
    recon_np = recon_verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    all_vertices = torch.cat([input_verts, recon_verts], dim=0).detach().cpu().numpy()
    min_xyz = all_vertices.min(axis=0)
    max_xyz = all_vertices.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    extent = float((max_xyz - min_xyz).max())
    radius = max(0.5 * extent, 1e-3)

    fig = plt.figure(figsize=(10, 5.8), constrained_layout=True)
    fig.suptitle(f"Direct Vertex AE Overfit | {case_name} | Vertex RMSE: {rmse:.6f}", fontsize=14)
    for idx, (title, verts) in enumerate((("Input", input_np), ("Reconstruction", recon_np)), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces_np,
            color="#8eb3d3",
            edgecolor=(0.06, 0.12, 0.18, 0.09),
            linewidth=0.08,
            alpha=0.30,
            shade=True,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=17, azim=35)
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_data = []
    for case in args.cases:
        verts, faces, mesh_name = load_case_mesh(args.ghd_root / case, device)
        case_data.append((case, verts, faces, mesh_name))

    num_verts = case_data[0][1].shape[0]
    input_dim = int(num_verts * 3)
    targets = torch.stack([verts.reshape(-1) for _, verts, _, _ in case_data], dim=0)

    model = MeshAutoencoder(input_dim=input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        recon = model(targets)
        loss = F.mse_loss(recon, targets)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            rmse = torch.sqrt(torch.mean((recon - targets) ** 2)).item()
            history.append({"epoch": epoch, "rmse": rmse, "loss": float(loss.item())})
            print({"epoch": epoch, "loss": float(loss.item()), "rmse": rmse})

    with torch.no_grad():
        recon = model(targets).reshape(len(case_data), num_verts, 3)

    results = []
    for idx, (case, verts, faces, mesh_name) in enumerate(case_data):
        rmse = torch.sqrt(torch.mean((recon[idx] - verts) ** 2)).item()
        output_path = output_dir / f"{case.replace('/', '__')}_direct_vertex_ae.png"
        render_pair(output_path, case, verts, recon[idx], faces, rmse=rmse, dpi=args.dpi)
        results.append(
            {
                "case": case,
                "mesh_name": mesh_name,
                "rmse": rmse,
                "output_path": str(output_path),
            }
        )

    summary = {
        "cases": args.cases,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "history": history,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
