# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared observation logic for quadrotors."""

from __future__ import annotations

import torch
import warp as wp
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    data = asset.data.root_lin_vel_b
    return wp.to_torch(data) if isinstance(data, wp.array) else data


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    data = asset.data.root_ang_vel_b
    return wp.to_torch(data) if isinstance(data, wp.array) else data


def root_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    data = asset.data.root_quat_w
    return wp.to_torch(data) if isinstance(data, wp.array) else data


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    data = asset.data.root_pos_w
    return wp.to_torch(data) if isinstance(data, wp.array) else data


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_pos_w = env.command_manager.get_term(command_name).command[:, :3]

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    # Convert to torch if needed
    if isinstance(root_pos_w, wp.array):
        root_pos_w = wp.to_torch(root_pos_w)
    if isinstance(root_quat_w, wp.array):
        root_quat_w = wp.to_torch(root_quat_w)

    # Convert target position to body frame
    pos_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, target_pos_w
    )
    return pos_b
