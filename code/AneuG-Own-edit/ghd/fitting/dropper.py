from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


class Do_Dropper:
    """Dynamically re-sample differentiable occupancy points.

    The original project used a lightweight helper to keep track of which DO
    points should participate in the loss.  The behaviour we need here is
    modest: optionally resample the indices according to an attention weight or
    uniformly at random.  When the dropper is disabled we simply keep all
    points.
    """

    def __init__(
        self,
        args: Any,
        weights_attention: Optional[torch.Tensor],
        num_queries: int,
        drop_num: int = 25,
        drop_rate: float = 0.75,
    ) -> None:
        self.device = torch.device(args.device)
        self.use_dropper = bool(getattr(args, "use_do_dropper", 0))
        self.interval = max(1, getattr(args, "do_drop_interval", getattr(args, "log_freq", 100)))
        self.drop_rate = float(max(0.0, min(1.0, drop_rate)))
        self.total_points = int(max(0, num_queries))
        self.drop_num = 0 if self.total_points <= 1 else min(drop_num, self.total_points - 1)
        self.keep_num = self.total_points - self.drop_num if self.total_points else 0
        self.current_index = torch.arange(self.total_points, device=self.device, dtype=torch.long)
        self.last_update_epoch = -1
        self.weights_attention = None
        if weights_attention is not None:
            self.weights_attention = weights_attention.detach().to(self.device)

    def _sample_indices(self) -> torch.Tensor:
        if self.weights_attention is not None and self.keep_num < self.total_points:
            weights = self.weights_attention.squeeze(0)
            weights = torch.clamp(weights, min=0)
            if torch.sum(weights) <= 0:
                weights = torch.ones_like(weights)
            prob = weights / torch.sum(weights)
            selected = torch.multinomial(prob, self.keep_num, replacement=False)
        else:
            selected = torch.randperm(self.total_points, device=self.device)[: self.keep_num]
        return torch.sort(selected)[0]

    def forward(self, epoch: int) -> Tuple[torch.Tensor, bool]:
        if not self.use_dropper or self.drop_num == 0 or self.total_points == 0:
            return self.current_index, False
        should_update = False
        if self.last_update_epoch < 0:
            should_update = True
        elif (epoch - self.last_update_epoch) >= self.interval:
            should_update = True
        if should_update and self.drop_rate < 1.0:
            rand_val = torch.rand(1, device=self.device).item()
            should_update = rand_val <= self.drop_rate
        if not should_update:
            return self.current_index, False
        self.current_index = self._sample_indices()
        self.last_update_epoch = epoch
        return self.current_index, True
