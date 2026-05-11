# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Mineral-facing adapter for differentiable direct environments."""

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


class MineralDirectEnvAdapter:
    """Present a differentiable direct environment through Mineral's 4-value API."""

    def __init__(self, env: DirectTaskProtocol) -> None:
        self.env = env
        self.device = getattr(env, "device", torch.device("cuda:0"))

        self.num_envs = int(getattr(env, "num_envs", 1))
        env_cfg = getattr(env, "cfg", None)
        self.num_obs = int(getattr(env_cfg, "observation_space", 4))
        self.num_actions = int(getattr(env_cfg, "action_space", 1))
        self.episode_length = int(getattr(env, "max_episode_length", 100))
        self.max_episode_length = self.episode_length

        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        self.bridge = self.env._get_checkpoint_bridge()
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.get_initial_joint_state()
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
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
        )

    def reset(self) -> torch.Tensor:
        _obs, _extras = self.env.reset()
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory()
        self._episode_step.zero_()
        return self.env.observe_joint_state(self._trajectory_joint_q, self._trajectory_joint_qd)

    def initialize_trajectory(self) -> torch.Tensor:
        self._trajectory_joint_q, self._trajectory_joint_qd = self.bridge.initialize_trajectory(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
        )
        return self.env.observe_joint_state(self._trajectory_joint_q, self._trajectory_joint_qd)

    def step(self, actions: torch.Tensor):
        action_view = actions.reshape(self.num_envs, self.num_actions)
        if actions.requires_grad:
            actions.retain_grad()
            self._tracked_actions.append(actions)

        next_joint_q, next_joint_qd = self.bridge.step(
            self._trajectory_joint_q,
            self._trajectory_joint_qd,
            action_view,
        )
        self._episode_step += 1

        terminal_obs = self.env.observe_joint_state(next_joint_q, next_joint_qd)
        terminated, truncated, done = self.env.terminal_flags_from_observation(terminal_obs, self._episode_step)
        reward = self.env.reward_from_observation(terminal_obs, terminated)

        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            obs, self._trajectory_joint_q, self._trajectory_joint_qd = self.env.apply_post_reset_rollout_state(
                done_env_ids,
                next_joint_q,
                next_joint_qd,
                self._episode_step,
            )
        else:
            obs = self.env.observe_joint_state(next_joint_q, next_joint_qd)
            self._trajectory_joint_q = next_joint_q
            self._trajectory_joint_qd = next_joint_qd

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
