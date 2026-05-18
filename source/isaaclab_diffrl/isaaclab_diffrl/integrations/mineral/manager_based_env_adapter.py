# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Mineral-facing adapter for differentiable manager-based environments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from ...tasks.direct.contracts import DirectTaskProtocol


@dataclass
class TransitionGradReport:
    """Small report used by the tracer-bullet validator."""

    tracked_steps: int
    has_action_grad: bool
    max_abs_action_grad: float


class MineralManagerBasedEnvAdapter:
    """Present a differentiable manager-based environment through Mineral's 4-value API.

    This adapter implements the DirectTaskProtocol by wrapping a ManagerBasedRLEnv.
    It expects the environment to have a rollout-capable bridge.
    """

    def __init__(self, env: any) -> None:
        self.env = env
        self.device = getattr(env, "device", torch.device("cuda:0"))

        self.num_envs = int(getattr(env, "num_envs", 1))
        policy_obs_dim = env.observation_manager.group_obs_dim["policy"]
        if isinstance(policy_obs_dim, list):
            self.num_obs = sum(math.prod(term_dim) for term_dim in policy_obs_dim)
        else:
            self.num_obs = math.prod(policy_obs_dim)
        self.num_actions = env.action_manager.total_action_dim
        self.episode_length = int(getattr(env, "max_episode_length", 100))
        self.max_episode_length = self.episode_length

        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        # The bridge must be provided by the environment or task layer
        # For this tracer bullet, we assume the environment provides _get_checkpoint_bridge()
        self.bridge = self.env._get_checkpoint_bridge()
        self._trajectory_state = self.bridge.get_initial_state()
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracked_actions: list[torch.Tensor] = []

    def close(self) -> None:
        self.env.close()

    @staticmethod
    def _obs_dict(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"obs": obs}

    def clear_grad(self, checkpoint=None) -> None:
        del checkpoint
        self._tracked_actions.clear()
        self._trajectory_state = self.bridge.initialize_trajectory(self._trajectory_state)

    def reset(self) -> torch.Tensor:
        _obs, _extras = self.env.reset()
        self._trajectory_state = self.bridge.initialize_trajectory()
        self._episode_step.zero_()
        return self.env.observe_from_state(self._trajectory_state)

    def initialize_trajectory(self) -> torch.Tensor:
        self._trajectory_state = self.bridge.initialize_trajectory(self._trajectory_state)
        return self.env.observe_from_state(self._trajectory_state)

    def step(self, actions: torch.Tensor):
        action_view = actions.reshape(self.num_envs, self.num_actions)
        if actions.requires_grad:
            actions.retain_grad()
            self._tracked_actions.append(actions)

        next_state = self.bridge.step(self._trajectory_state, action_view)
        self._episode_step += 1

        terminal_obs = self.env.observe_from_state(next_state)
        if hasattr(self.env, "terminal_flags_from_state"):
            terminated, truncated, done = self.env.terminal_flags_from_state(next_state, self._episode_step)
        else:
            terminated, truncated, done = self.env.terminal_flags_from_observation(terminal_obs, self._episode_step)
        reward = self.env.reward_from_observation(terminal_obs, terminated)

        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            obs, self._trajectory_state = self.env.apply_post_reset_rollout_state(
                done_env_ids,
                next_state,
                self._episode_step,
            )
        else:
            obs = terminal_obs
            self._trajectory_state = next_state

        return obs, reward, done, {
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "obs_before_reset": self._obs_dict(terminal_obs),
        }

    def transition_grad_report(self) -> TransitionGradReport:
        max_abs_action_grad = 0.0
        has_action_grad = False
        for action in self._tracked_actions:
            grad = action.grad
            if grad is None:
                continue
            grad_detached = grad.detach()
            if grad_detached.numel() == 0:
                continue
            max_abs_action_grad = max(max_abs_action_grad, float(grad_detached.abs().max().item()))
            has_action_grad = has_action_grad or bool(
                torch.isfinite(grad_detached).all().item() and grad_detached.abs().max().item() > 0.0
            )
        return TransitionGradReport(
            tracked_steps=len(self._tracked_actions),
            has_action_grad=has_action_grad,
            max_abs_action_grad=max_abs_action_grad,
        )
