from __future__ import annotations

import copy
from typing import Dict, Any


class base_loss_weighter:
    """Utility class that provides per-epoch loss weights.

    Historically this logic lived in a standalone script.  The implementation
    here focuses on the pieces that the public repository relies on: keep a
    base dictionary of weights and expose an ``easy_weighting`` helper that can
    optionally modulate the weights over time.
    """

    def __init__(self, args: Any, glo_loss_weighting: Dict[str, float], style: str = "static") -> None:
        self.args = args
        self.base_weighting = copy.deepcopy(glo_loss_weighting or {})
        self.style = style or "static"
        self.epochs = getattr(args, "epochs", None)

    def easy_weighting(self, epoch: int) -> Dict[str, float]:
        """Return the loss weights for the current epoch."""
        weights = copy.deepcopy(self.base_weighting)
        if not weights:
            return {}
        style = self.style.lower()
        if style == "static":
            return weights
        if style == "strategy_v1_linear":
            warmup = getattr(self.args, "weighter_warmup", max(int(0.1 * (self.epochs or 1)), 1))
            scale = min(max(epoch, 0) / max(warmup, 1), 1.0)
            for key in weights:
                weights[key] = weights[key] * scale
            return weights
        if style == "exp_decay":
            gamma = float(getattr(self.args, "weighter_gamma", 0.995))
            scale = gamma ** max(epoch, 0)
            for key in weights:
                weights[key] = weights[key] * scale
            return weights
        if style == "milestone_decay":
            milestones = getattr(self.args, "weighter_milestones", []) or []
            decay = getattr(self.args, "weighter_decay", 0.5)
            multiplier = 1.0
            for milestone in milestones:
                if epoch >= milestone:
                    multiplier *= decay
            for key in weights:
                weights[key] = weights[key] * multiplier
            return weights
        # Fallback for unknown strategies.
        return weights
