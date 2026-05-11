# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable direct-task seam and contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class DirectTaskProtocol(Protocol):
    """Protocol for differentiable direct environments.

    Establishes the shared task seam for trajectory initialization, checkpoint,
    and step semantic compatibility.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and detach from prior computation history."""
        ...

    def step(self, action: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute one environment step with rewarped-compatible terminal observation semantics."""
        ...

    def get_policy_observation(self, observations: dict[str, Any]) -> torch.Tensor:
        """Extract the policy observation tensor from an observation dictionary."""
        ...

    def get_terminal_policy_observation(self, extras: dict[str, Any]) -> torch.Tensor:
        """Extract the terminal policy observation tensor from step extras."""
        ...

    def initialize_trajectory_from_current_state(
        self, env_ids: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Start a new differentiable rollout from the current physical state."""
        ...

    def initialize_trajectory(self) -> torch.Tensor:
        """Start a new differentiable rollout and return the initial observation."""
        ...

    def export_differentiable_checkpoint(self) -> dict[str, dict[str, torch.Tensor]]:
        """Export the rollout checkpoint over model/state/control tensors."""
        ...

    def restore_differentiable_checkpoint(self, checkpoint: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore a rollout checkpoint without touching env bookkeeping buffers."""
        ...
