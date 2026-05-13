# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

from ..isaaclab_diffrl_manager_env import IsaaclabDiffrlManagerEnv
from .drone_position_control_env_cfg import DronePositionControlEnvCfg
from ....quadrotor import observations as quad_obs


class DronePositionControlEnv(IsaaclabDiffrlManagerEnv):
    """Quadrotor position control environment."""

    cfg: DronePositionControlEnvCfg

    def __init__(self, cfg: DronePositionControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._active_bridge = None
        self.arrive_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.arrival_radius = 0.5
        self.wait_before_truncate = 0.5

    def _get_checkpoint_bridge(self):
        """Return the active rollout bridge, creating one on demand."""
        if self._active_bridge is None:
            # We'll implement QuadrotorNewtonAutogradBridge later
            from ....rollout.quadrotor_bridge import NewtonQuadrotorAutogradBridge
            self._active_bridge = NewtonQuadrotorAutogradBridge(
                self,
                num_envs=self.num_envs,
            )
        return self._active_bridge

    def observe_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Compute observations from 13D state [pos, quat, lin_vel, ang_vel]."""
        # state: (num_envs, 13)
        pos_w = state[:, 0:3]
        quat_w = state[:, 3:7]
        lin_vel_w = state[:, 7:10]
        ang_vel_w = state[:, 10:13]

        # Target position from command manager
        target_pos_w = self.command_manager.get_term("target_pos").command[:, :3]

        # Convert to body frame
        # pos_error_b
        pos_error_b, _ = subtract_frame_transforms(pos_w, quat_w, target_pos_w)

        # lin_vel_b (body frame)
        lin_vel_b = quat_apply_inverse(quat_w, lin_vel_w)

        # ang_vel_b (body frame)
        ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)

        # Observation: [pos_error_b (3), quat_w (4), lin_vel_b (3), ang_vel_b (3)]
        return torch.cat([pos_error_b, quat_w, lin_vel_b, ang_vel_b], dim=-1)

    def reward_from_observation(self, observation: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Compute reward from observation."""
        pos_error_b = observation[:, 0:3]
        lin_vel_b = observation[:, 7:10]
        ang_vel_b = observation[:, 10:13]

        pos_reward = -torch.norm(pos_error_b, dim=-1)
        vel_penalty = -0.01 * torch.norm(lin_vel_b, dim=-1)
        omega_penalty = -0.01 * torch.norm(ang_vel_b, dim=-1)

        return pos_reward + vel_penalty + omega_penalty

    def terminal_flags_from_observation(
        self, observation: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute termination flags."""
        pos_error_b = observation[:, 0:3]
        arrived = torch.linalg.norm(pos_error_b, dim=-1) < self.arrival_radius
        curr_time = episode_step.float() * (self.cfg.sim.dt * self.cfg.decimation)
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), curr_time, self.arrive_time))

        truncated = episode_step >= self.max_episode_length - 1
        truncated |= arrived & (curr_time > (self.arrive_time + self.wait_before_truncate))
        terminated = torch.zeros_like(truncated)
        done = terminated | truncated
        self.success.copy_(arrived & truncated)
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
        """Apply reset logic."""
        next_state_detached = next_state.detach().clone()
        self._reset_idx(env_ids)
        reset_state = self.initialize_trajectory_from_current_state(env_ids)
        next_state_detached[env_ids] = reset_state
        episode_step[env_ids] = 0
        self.arrive_time[env_ids] = 0.0
        self.success[env_ids] = False
        obs = self.observe_from_state(next_state_detached)
        return obs, next_state_detached
