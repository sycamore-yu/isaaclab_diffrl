# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed cartpole environment with trajectory-init and reset-detach semantics.

Localizes cartpole-specific observation, reward, and trajectory helpers into the task layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import warp as wp

from isaaclab.utils.math import sample_uniform
from isaaclab_diffrl.rollout import NewtonCartpoleAutogradBridge

from ..contracts import DirectTaskProtocol
from .cartpole_env import CartpoleEnv
from .cartpole_env_cfg import CartpoleEnvCfg


class CartpoleNewtonEnv(CartpoleEnv, DirectTaskProtocol):
    """Newton-backed cartpole environment for differentiable rollout validation.

    Extends the base cartpole env with explicit trajectory-initialisation and
    reset-detach semantics required for differentiable-rollout workflows.
    """

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        self._cached_obs_before_reset = None
        self._active_bridge: NewtonCartpoleAutogradBridge | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _zero_bridge_tape(self) -> None:
        """Zero any uncleared tape references on the active Newton bridge."""
        bridge = getattr(self, "_active_bridge", None)
        if bridge is not None:
            bridge._clear_grad_refs()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and detach from prior computation history."""
        self._zero_bridge_tape()

        if seed is not None:
            self.seed(seed)

        # Write default joint state (with pole angle randomisation) directly
        # into the Newton buffers that the bridge controls.
        default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).detach().clone()
        pole_angle_noise = sample_uniform(
            float(self.cfg.initial_pole_angle_range[0]),
            float(self.cfg.initial_pole_angle_range[1]),
            default_joint_pos[:, self._pole_dof_idx].shape,
            default_joint_pos.device,
        )
        default_joint_pos[:, self._pole_dof_idx] += pole_angle_noise
        default_joint_vel = wp.to_torch(self.robot.data.default_joint_vel).detach().clone()

        self.joint_pos[:] = default_joint_pos
        self.joint_vel[:] = default_joint_vel

        self.episode_length_buf[:] = 0
        self.reset_buf[:] = False
        self.reset_terminated[:] = False
        self.reset_time_outs[:] = False

        self.initialize_trajectory_from_current_state()

        return self._get_observations(), self.extras

    def initialize_trajectory_from_current_state(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Start a new differentiable rollout from the current physical state."""
        if env_ids is None:
            env_ids = slice(None)

        # Detach: robot data may be differentiable, but the env observability
        # tensors must stay grad-free for _reset_idx in-place writes.
        self.joint_pos = wp.to_torch(self.robot.data.joint_pos).detach()
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel).detach()

        q = self.joint_pos[env_ids].detach().clone()
        qd = self.joint_vel[env_ids].detach().clone()

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.reset_terminated[env_ids] = False
        self.reset_time_outs[env_ids] = False

        return q, qd

    def initialize_trajectory(self) -> torch.Tensor:
        """Start a new differentiable rollout and return the initial observation."""
        q, qd = self.initialize_trajectory_from_current_state()
        self._get_checkpoint_bridge().initialize_trajectory(q, qd)
        return self.get_policy_observation(self._get_observations())

    def get_terminal_policy_observation(self, extras: dict[str, Any]) -> torch.Tensor:
        return self.get_policy_observation(extras["obs_before_reset"])

    def _sample_reset_joint_state(self, done_env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reset_joint_q = wp.to_torch(self.robot.data.default_joint_pos).detach().clone()[done_env_ids]
        reset_joint_qd = wp.to_torch(self.robot.data.default_joint_vel).detach().clone()[done_env_ids]
        pole_angle_noise = sample_uniform(
            float(self.cfg.initial_pole_angle_range[0]),
            float(self.cfg.initial_pole_angle_range[1]),
            reset_joint_q[:, self._pole_dof_idx].shape,
            reset_joint_q.device,
        )
        reset_joint_q[:, self._pole_dof_idx] += pole_angle_noise
        return reset_joint_q, reset_joint_qd

    def apply_post_reset_rollout_state(
        self,
        done_env_ids: torch.Tensor,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        episode_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_q_view = joint_q.reshape(self.num_envs, -1).detach().clone()
        joint_qd_view = joint_qd.reshape(self.num_envs, -1).detach().clone()

        if done_env_ids.numel() > 0:
            reset_joint_q, reset_joint_qd = self._sample_reset_joint_state(done_env_ids)
            joint_q_view[done_env_ids] = reset_joint_q
            joint_qd_view[done_env_ids] = reset_joint_qd
            self.joint_pos[done_env_ids] = reset_joint_q
            self.joint_vel[done_env_ids] = reset_joint_qd
            episode_step[done_env_ids] = 0

        rollout_joint_q = joint_q_view.reshape_as(joint_q)
        rollout_joint_qd = joint_qd_view.reshape_as(joint_qd)
        obs = self.observe_joint_state(rollout_joint_q, rollout_joint_qd)
        return obs, rollout_joint_q, rollout_joint_qd

    def _get_checkpoint_bridge(self):
        """Return the active Newton rollout bridge, creating one on demand."""
        bridge = getattr(self, "_active_bridge", None)
        if bridge is None:
            bridge = NewtonCartpoleAutogradBridge(
                self,
                num_envs=self.num_envs,
                action_scale=float(self.cfg.action_scale),
                cart_dof_idx=int(self._cart_dof_idx[0]),
            )
        return bridge

    def export_differentiable_checkpoint(self) -> dict[str, dict[str, torch.Tensor]]:
        """Export the rollout checkpoint over Newton model/state/control tensors."""
        return self._get_checkpoint_bridge().export_checkpoint()

    def restore_differentiable_checkpoint(self, checkpoint: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore a rollout checkpoint without touching env bookkeeping buffers."""
        self._get_checkpoint_bridge().restore_checkpoint(checkpoint)

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset specified envs and capture terminal observation before state change."""
        self._zero_bridge_tape()
        if self._cached_obs_before_reset is None:
            self._cached_obs_before_reset = self._get_observations()
        super()._reset_idx(env_ids)

    def step(self, action: torch.Tensor):
        """Execute one environment step with rewarped-compatible terminal observation semantics."""
        self._cached_obs_before_reset = None
        obs, rew, terminated, truncated, extras = super().step(action)
        if self._cached_obs_before_reset is not None:
            extras["obs_before_reset"] = self._cached_obs_before_reset
        else:
            extras["obs_before_reset"] = obs
        return obs, rew, terminated, truncated, extras
