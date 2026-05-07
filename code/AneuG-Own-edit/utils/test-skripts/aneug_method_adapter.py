from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


class ExternalAneuGSampler:
    """Adapter for /path/to/SynVA-A1 method checkpoints.

    The official vessel-mesh-editing-master inference script expects raw target
    vectors in the fitted Stage-1 space: [GHD(432), scale(1)].  A/B/C/D/E/W
    checkpoints in /path/to/SynVA-A1 usually sample normalized vectors, so this
    adapter builds the matching AneuG dataset/statistics and denormalizes before
    returning to the reference mesh replay code.
    """

    def __init__(
        self,
        *,
        method_type: str,
        checkpoint: Path,
        aneug_root: Path,
        device: torch.device,
        temperature: float = 0.8,
        top_k: int = 0,
        flow_steps: int = 64,
        flow_sampler: str = "heun",
    ):
        self.method_type = str(method_type)
        self.checkpoint = Path(checkpoint)
        self.aneug_root = Path(aneug_root)
        self.device = device
        self.sample_args = SimpleNamespace(
            temperature=float(temperature),
            top_k=int(top_k),
            flow_steps=int(flow_steps),
            flow_sampler=str(flow_sampler),
        )
        if str(self.aneug_root) not in sys.path:
            sys.path.insert(0, str(self.aneug_root))
        # This script runs inside vessel-mesh-editing-master, which also has a
        # top-level ``models`` package. Drop those cached modules before loading
        # /path/to/SynVA-A1 so imports inside methods/eval_all resolve to our
        # method implementations. Already imported reference classes remain bound
        # in the caller.
        for module_name in list(sys.modules):
            if (
                module_name == "models"
                or module_name.startswith("models.")
                or module_name == "methods"
                or module_name.startswith("methods.")
                or module_name in {"first_stage_vessel_aware", "train_vessel_flow_matching"}
            ):
                sys.modules.pop(module_name, None)
        from methods.eval_all import METHOD_LOADERS, _build_dataset, _copy_stats
        from first_stage_vessel_aware import collate_fn
        from train_vessel_flow_matching import condition_from_batch

        self._build_dataset = _build_dataset
        self._copy_stats = _copy_stats
        self._collate_fn = collate_fn
        self._condition_from_batch = condition_from_batch
        self.ckpt, self.saved_args, self.cond_net, self.sampler = METHOD_LOADERS[self.method_type](
            str(self.checkpoint), self.device
        )
        self._dataset_cache = {}

    def _dataset_for_case(self, case_name: str):
        if case_name not in self._dataset_cache:
            ds = self._build_dataset([case_name], self.saved_args)
            self._copy_stats(ds, self.ckpt)
            if "orig_ghd_mean" in self.ckpt:
                ds.ghd_mean = self.ckpt["orig_ghd_mean"].cpu()
                ds.ghd_std = self.ckpt["orig_ghd_std"].cpu()
            self._dataset_cache[case_name] = ds
        return self._dataset_cache[case_name]

    def _denormalize(self, samples_norm: torch.Tensor, ds) -> torch.Tensor:
        mean = ds.ghd_mean.to(samples_norm.device)
        std = ds.ghd_std.to(samples_norm.device)
        dim = min(samples_norm.shape[-1], mean.shape[-1])
        raw = samples_norm[..., :dim] * std[..., :dim] + mean[..., :dim]
        if raw.shape[-1] == 433:
            return raw
        if raw.shape[-1] != 432:
            raise ValueError(f"Expected 432 or 433 sampled dims, got {raw.shape[-1]}")
        if hasattr(ds, "scale") and len(ds.scale) > 0:
            scale = ds.scale[0].reshape(1, 1).to(raw.device).float().abs()
        elif mean.shape[-1] > 432:
            scale = mean[..., 432:433].to(raw.device).float().abs()
        else:
            scale = torch.ones((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
        scale = scale.repeat(raw.shape[0], 1)
        return torch.cat([raw, scale], dim=-1)

    def sample_raw_for_case(self, case_name: str, num_samples: int) -> torch.Tensor:
        ds = self._dataset_for_case(case_name)
        item = ds[0]
        batch = self._collate_fn([item])
        cond = self._condition_from_batch(self.cond_net, batch, self.device)
        with torch.no_grad():
            samples_norm = self.sampler(cond, int(num_samples), self.sample_args)[:, 0, :]
            samples_raw = self._denormalize(samples_norm, ds)
        return samples_raw.detach().cpu()
