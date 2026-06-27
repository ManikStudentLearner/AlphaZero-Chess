"""
AlphaZero neural network: a Residual Convolutional Tower with
a Policy Head (4672-dim logits) and a Value Head (scalar in [-1, 1]).

Architecture mirrors the original DeepMind paper, scaled down for
a single-GPU / laptop training setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_RES_BLOCKS, NUM_FILTERS, INPUT_CHANNELS, POLICY_OUTPUT_SIZE


class ResBlock(nn.Module):
    """Two-conv residual block with batch norm and skip connection."""

    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    Dual-head network that maps a board state → (policy_logits, value).

    Args:
        num_res_blocks: Number of residual blocks in the tower.
        num_filters:    Channel width throughout the tower.
    """

    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()

        # ── Input convolution ──────────────────────────────
        self.conv_in = nn.Conv2d(INPUT_CHANNELS, num_filters,
                                 kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(num_filters)

        # ── Residual tower ─────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # ── Policy head ────────────────────────────────────
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, POLICY_OUTPUT_SIZE)

        # ── Value head ─────────────────────────────────────
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(32)
        self.value_fc1  = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 18, 8, 8) canonical board tensor.

        Returns:
            policy_logits: (batch, 4672) raw policy scores.
            value:         (batch, 1)    scalar in [-1, 1].
        """
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value