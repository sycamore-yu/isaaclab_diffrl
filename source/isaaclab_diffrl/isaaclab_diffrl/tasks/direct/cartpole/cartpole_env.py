# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .rewards import compute_rewards

from .cartpole_env_cfg import CartpoleEnvCfg


class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(target=self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    @staticmethod
    def get_policy_observation(observations: dict[str, Any]) -> torch.Tensor:
        return observations["policy"]

    def observe_joint_state(self, joint_q: torch.Tensor, joint_qd: torch.Tensor) -> torch.Tensor:
        joint_q_view = joint_q.reshape(self.num_envs, -1)
        joint_qd_view = joint_qd.reshape(self.num_envs, -1)
        return torch.stack(
            (
                joint_q_view[:, self._pole_dof_idx[0]],
                joint_qd_view[:, self._pole_dof_idx[0]],
                joint_q_view[:, self._cart_dof_idx[0]],
                joint_qd_view[:, self._cart_dof_idx[0]],
            ),
            dim=-1,
        )

    def terminal_flags_from_observation(
        self, obs: torch.Tensor, episode_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terminated = (obs[:, 2].abs() > float(self.cfg.max_cart_pos)) | (obs[:, 0].abs() > math.pi / 2.0)
        truncated = episode_step >= self.max_episode_length
        done = terminated | truncated
        return terminated, truncated, done

    def reward_from_observation(self, obs: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        return compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            obs[:, 0],
            obs[:, 1],
            obs[:, 2],
            obs[:, 3],
            terminated,
        )

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = wp.to_torch(self.robot.data.joint_pos).detach()
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel).detach()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds |= torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2.0, dim=1)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        pole_angle_noise = sample_uniform(
            float(self.cfg.initial_pole_angle_range[0]),
            float(self.cfg.initial_pole_angle_range[1]),
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pole_dof_idx] += pole_angle_noise

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
