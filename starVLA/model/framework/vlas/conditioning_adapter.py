from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BottleneckProjectionAdapter(nn.Module):
    """CLARE-style nonlinear side branch: up(ReLU(down(x))).

    The branch can map between different input/output dimensions, which makes it
    usable both for LaWAM's VLM->Flow projection (2048->768) and for AdaLN
    modulation projection (1024->2048).  The up projection is zero initialized by
    default, so enabling the adapter preserves the Base policy exactly at step 0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        bottleneck_dim = int(bottleneck_dim)
        if input_dim <= 0 or output_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError(
                f"Adapter dims must be positive, got in={input_dim}, out={output_dim}, bottleneck={bottleneck_dim}."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        if zero_init:
            nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.relu(self.down(x)))
