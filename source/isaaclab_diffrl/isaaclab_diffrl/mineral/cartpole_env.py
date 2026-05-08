# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin Mineral-facing adapter for the extension-owned Newton cartpole env.

This adapter keeps Mineral's APG/BPTT contract intact while delegating all
physics-state ownership to the IsaacLab DiffRL cartpole environment plus the
existing Newton autograd bridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..tasks.direct.isaaclab_diffrl.isaaclab_diffrl_env import compute_rewards
from ..tasks.direct.isaaclab_diffrl.newton_torch_autograd import NewtonCartpoleAutogradBridge


@dataclass
class TransitionGradReport:
    """Small report used by the tracer-bullet validator."""

    tracked_steps: int
    has_action_grad: bool
    max_abs_action_grad: float


class MineralCartpoleEnvAdapter:
    """Present the IsaacLab DiffRL cartpole env through Mineral's 4-value API.

    Mineral's BPTT agent expects a rewarped-style environment contract:
    ``reset() -> obs``, ``initialize_trajectory() -> obs``,
    ``step() -> (obs, reward, done, extras)``, plus attributes such as
    ``num_envs``, ``num_obs``, ``num_actions``, and ``episode_length``.

    The differentiable rollout path itself is driven by the existing
    ``NewtonCartpoleAutogradBridge`` so ``actor_loss.backward()`` propagates
    through the Newton physics transition.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.device = env.device
        self.bridge = NewtonCartpoleAutogradBridge(env)

        self.num_envs = 1
        self.num_obs = 4
        self.num_actions = 1
        self.episode_length = int(env.max_episode_length)

        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.get_initial_joint_state()
        self._episode_step = 0
        self._pending_reset = False
        self._tracked_actions: list[torch.Tensor] = []

    def close(self) -> None:
        self.env.close()

    def _obs_from_joint_state(self, joint_q: torch.Tensor, joint_qd: torch.Tensor) -> torch.Tensor:
        obs = torch.stack((joint_q[1], joint_qd[1], joint_q[0], joint_qd[0]))
        return obs.unsqueeze(0)

    def _terminal_flags(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cart_pos = obs[:, 2]
        pole_pos = obs[:, 0]
        terminated = (cart_pos.abs() > float(self.env.cfg.max_cart_pos)) | (pole_pos.abs() > math.pi / 2.0)
        truncated = torch.tensor([self._episode_step >= self.episode_length], device=self.device, dtype=torch.bool)
        done = terminated | truncated
        return terminated, truncated, done

    def _reward_from_obs(self, obs: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        return compute_rewards(
            float(self.env.cfg.rew_scale_alive),
            float(self.env.cfg.rew_scale_terminated),
            float(self.env.cfg.rew_scale_pole_pos),
            float(self.env.cfg.rew_scale_cart_vel),
            float(self.env.cfg.rew_scale_pole_vel),
            obs[:, 0],
            obs[:, 1],
            obs[:, 2],
            obs[:, 3],
            terminated,
        )

    def _detach_current_rollout_state(self) -> None:
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
        )

    def clear_grad(self, checkpoint=None) -> None:
        del checkpoint
        self._tracked_actions.clear()
        self._detach_current_rollout_state()

    def reset(self) -> torch.Tensor:
        self._episode_step = 0
        self._pending_reset = False
        self.env.reset()
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory()
        return self._obs_from_joint_state(self._trajectory_joint_q, self._trajectory_joint_qd)

    def initialize_trajectory(self) -> torch.Tensor:
        if self._pending_reset:
            return self.reset()
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
        )
        return self._obs_from_joint_state(self._trajectory_joint_q, self._trajectory_joint_qd)

    def step(self, actions: torch.Tensor):
        if self._pending_reset:
            obs = self.reset()
            extras = {
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "obs_before_reset": obs,
            }
            reward = torch.zeros(self.num_envs, dtype=obs.dtype, device=self.device)
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, reward, done, extras

        if actions.ndim == 2:
            action_vec = actions[0]
        else:
            action_vec = actions.reshape(-1)
        if action_vec.requires_grad:
            action_vec.retain_grad()
        self._tracked_actions.append(action_vec)

        next_joint_q, next_joint_qd = self.bridge.step(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
            action_vec,
        )
        self._trajectory_joint_q = next_joint_q
        self._trajectory_joint_qd = next_joint_qd
        self._episode_step += 1

        obs = self._obs_from_joint_state(next_joint_q, next_joint_qd)
        terminated, truncated, done = self._terminal_flags(obs)
        reward = self._reward_from_obs(obs, terminated)
        extras = {
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "obs_before_reset": obs,
        }
        if bool(done.any().item()):
            self._pending_reset = True
        return obs, reward, done, extras

    def transition_grad_report(self) -> TransitionGradReport:
        max_abs_action_grad = 0.0
        has_action_grad = False
        for action in self._tracked_actions:
            grad = action.grad
            if grad is None:
                continue
            finite_grad = grad.detach()
            if finite_grad.numel() == 0:
                continue
            max_abs_action_grad = max(max_abs_action_grad, float(finite_grad.abs().max().item()))
            has_action_grad = has_action_grad or bool(torch.isfinite(finite_grad).all().item() and finite_grad.abs().max().item() > 0.0)
        return TransitionGradReport(
            tracked_steps=len(self._tracked_actions),
            has_action_grad=has_action_grad,
            max_abs_action_grad=max_abs_action_grad,
        )
