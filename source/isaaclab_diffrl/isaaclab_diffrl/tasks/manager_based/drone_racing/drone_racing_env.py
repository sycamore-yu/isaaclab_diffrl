# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone racing environment with differentiable BPTT support."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms, quat_rotate

from ..isaaclab_diffrl_manager_env import IsaaclabDiffrlManagerEnv
from .drone_racing_env_cfg import DroneRacingEnvCfg
from ....quadrotor import observations as quad_obs


class DroneRacingEnv(IsaaclabDiffrlManagerEnv):
    """Drone racing environment with gate progression.

    Task-level checkpoint payload: racing command (next_gate_idx) and progress state.
    Rollout-level checkpoint payload: differentiable physics resume state (13D).
    """

    cfg: DroneRacingEnvCfg

    def __init__(self, cfg: DroneRacingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._active_bridge = None

        # Task-level gate progression state
        self._next_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_gate_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Gate reward tracking
        self._gate_size = cfg.commands.target.gate_size
        self._num_gates = 4  # Track has 4 gates

    def _get_checkpoint_bridge(self):
        """Return the active rollout bridge, creating one on demand."""
        if self._active_bridge is None:
            from ....rollout.quadrotor_bridge import NewtonQuadrotorAutogradBridge
            self._active_bridge = NewtonQuadrotorAutogradBridge(
                self,
                num_envs=self.num_envs,
            )
        return self._active_bridge

    def observe_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Compute observations from 13D state [pos, quat, lin_vel, ang_vel].

        Shared phase-one observation frame: body-frame where applicable.
        """
        # state: (num_envs, 13)
        pos_w = state[:, 0:3]
        quat_w = state[:, 3:7]
        lin_vel_w = state[:, 7:10]
        ang_vel_w = state[:, 10:13]

        # Get next gate target from command manager
        target_pos_w = self.command_manager.get_term("target").command[:, :3]
        target_quat_w = self.command_manager.get_term("target").command[:, 3:7]

        # Convert to body frame
        pos_error_b, _ = subtract_frame_transforms(pos_w, quat_w, target_pos_w)
        lin_vel_b = quat_apply_inverse(quat_w, lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)

        # Observation: [target_pos_b (3), quat_w (4), lin_vel_b (3), ang_vel_b (3)]
        return torch.cat([pos_error_b, quat_w, lin_vel_b, ang_vel_b], dim=-1)

    def reward_from_observation(self, observation: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Compute reward from observation.

        Reward terms:
        - progress: gate index increment
        - gate_passed: discrete gate passage reward
        - ang_vel_l2: angular velocity penalty
        - velocity_l2: linear velocity penalty
        """
        # Get command state for reward computation
        cmd_term = self.command_manager.get_term("target")
        gate_passed = cmd_term.gate_passed if hasattr(cmd_term, 'gate_passed') else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_gate_idx = cmd_term.next_gate_idx if hasattr(cmd_term, 'next_gate_idx') else self._next_gate_idx.float()

        # Gate passage reward
        gate_reward = gate_passed.float() * 400.0

        # Progress reward (based on gate index)
        progress_reward = next_gate_idx.float() * 20.0

        # Penalty terms
        ang_vel_b = observation[:, 10:13]
        lin_vel_b = observation[:, 7:10]
        ang_vel_penalty = -0.001 * torch.sum(ang_vel_b ** 2, dim=-1)
        lin_vel_penalty = -0.01 * torch.sum(lin_vel_b ** 2, dim=-1)

        # Termination penalty
        term_penalty = terminated.float() * -500.0

        return gate_reward + progress_reward + ang_vel_penalty + lin_vel_penalty + term_penalty

    def terminal_flags_from_observation(
        self, observation: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute termination flags.

        - terminated: collision or crash (not used in current simple model)
        - truncated: episode length exceeded
        """
        # For this tracer bullet, we only truncate on episode length
        truncated = episode_step >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        done = terminated | truncated
        return terminated, truncated, done

    def initialize_trajectory_from_current_state(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Start a new differentiable rollout from the current physical state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        pos_w = wp.to_torch(self.robot.data.root_pos_w)[env_ids].detach().clone()
        quat_w = wp.to_torch(self.robot.data.root_quat_w)[env_ids].detach().clone()
        lin_vel_w = wp.to_torch(self.robot.data.root_lin_vel_w)[env_ids].detach().clone()
        ang_vel_w = wp.to_torch(self.robot.data.root_ang_vel_w)[env_ids].detach().clone()
        return torch.cat([pos_w, quat_w, lin_vel_w, ang_vel_w], dim=-1)

    def apply_post_reset_rollout_state(
        self, env_ids: torch.Tensor, next_state: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply reset logic for done environments."""
        next_state_detached = next_state.detach().clone()
        self._reset_idx(env_ids)
        reset_state = self.initialize_trajectory_from_current_state(env_ids)
        next_state_detached[env_ids] = reset_state
        episode_step[env_ids] = 0
        # Reset task-level gate progress state
        self._next_gate_idx[env_ids] = 0
        self._gate_passed[env_ids] = False
        self._episode_gate_count[env_ids] = 0
        obs = self.observe_from_state(next_state_detached)
        return obs, next_state_detached