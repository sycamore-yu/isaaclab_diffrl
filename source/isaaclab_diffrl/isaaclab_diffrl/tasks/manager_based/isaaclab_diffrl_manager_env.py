# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for differentiable manager-based environments."""

from __future__ import annotations

from typing import Any
import torch
from isaaclab.envs import ManagerBasedRLEnv


class IsaaclabDiffrlManagerEnv(ManagerBasedRLEnv):
    """Base class for differentiable manager-based environments.

    Provides hooks for differentiable rollout and bridge integration.
    """

    def __init__(self, cfg: any, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def observe_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Compute observations from a differentiable state tensor.

        This method should be overridden by subclasses or use managers to compute
        observations from the provided state instead of self.robot.data.
        """
        raise NotImplementedError

    def reward_from_observation(self, observation: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Compute reward from a differentiable observation tensor."""
        raise NotImplementedError

    def terminal_flags_from_observation(
        self, observation: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute termination flags from a differentiable observation tensor."""
        raise NotImplementedError

    def apply_post_reset_rollout_state(
        self, env_ids: torch.Tensor, next_state: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply reset logic to a subset of environments in a differentiable rollout."""
        raise NotImplementedError

    def _get_checkpoint_bridge(self) -> any:
        """Return the active rollout bridge."""
        raise NotImplementedError
