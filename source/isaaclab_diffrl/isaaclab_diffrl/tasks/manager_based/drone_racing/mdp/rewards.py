# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for drone racing."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminal reward for terminated episodes."""
    # For now, return zeros - termination is handled by truncations
    return torch.zeros(env.num_envs, device=env.device)


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time-based termination flag."""
    return env.episode_length_buf >= env.max_episode_length - 1


def progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward for making progress toward next gate."""
    command = env.command_manager.get_term(command_name)
    return command.next_gate_idx.float()


def gate_passed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward for passing through a gate."""
    command = env.command_manager.get_term(command_name)
    return command.gate_passed.float()


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: any = None) -> torch.Tensor:
    """Penalize angular velocity."""
    from isaaclab_diffrl.quadrotor.observations import root_ang_vel_b
    ang_vel = root_ang_vel_b(env)
    return torch.sum(ang_vel ** 2, dim=-1)


def velocity_l2(env: ManagerBasedRLEnv, asset_cfg: any = None) -> torch.Tensor:
    """Penalize linear velocity."""
    from isaaclab_diffrl.quadrotor.observations import root_lin_vel_b
    lin_vel = root_lin_vel_b(env)
    return torch.sum(lin_vel ** 2, dim=-1)


def position_l2_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Position error to gate."""
    from isaaclab_diffrl.quadrotor.observations import target_pos_b
    pos_error = target_pos_b(env, command_name)
    return torch.norm(pos_error, dim=-1)