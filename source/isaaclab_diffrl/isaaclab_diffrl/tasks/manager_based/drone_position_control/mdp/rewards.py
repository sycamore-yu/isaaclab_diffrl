# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions for quadrotor MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg: any = None) -> torch.Tensor:
    """Reward for being close to the target position."""
    # target_pos_b is already provided in body frame by our helper
    from isaaclab_diffrl.quadrotor.observations import target_pos_b
    pos_error = target_pos_b(env, command_name)
    return torch.norm(pos_error, dim=-1)

def velocity_l2(env: ManagerBasedRLEnv, asset_cfg: any = None) -> torch.Tensor:
    """Penalty for high velocity."""
    from isaaclab_diffrl.quadrotor.observations import root_lin_vel_b
    lin_vel = root_lin_vel_b(env)
    return torch.norm(lin_vel, dim=-1)

def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: any = None) -> torch.Tensor:
    """Penalty for high angular velocity."""
    from isaaclab_diffrl.quadrotor.observations import root_ang_vel_b
    ang_vel = root_ang_vel_b(env)
    return torch.norm(ang_vel, dim=-1)
