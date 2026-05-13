# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for drone racing."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation


def reset_robot_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Reset robot to random position near the first gate."""
    asset: RigidObject | Articulation = env.scene["robot"]

    # Start position: in front of first gate
    pos_min = torch.tensor([-6.0, -0.5, 1.5], device=env.device)
    pos_max = torch.tensor([-4.0, 0.5, 2.5], device=env.device)
    positions = math_utils.sample_uniform(pos_min, pos_max, (len(env_ids), 3), device=env.device)
    positions += env.scene.env_origins[env_ids]

    # Identity quaternion
    orientations = torch.zeros((len(env_ids), 4), device=env.device)
    orientations[:, 0] = 1.0  # w=1 for identity

    # Zero velocity
    velocities = torch.zeros((len(env_ids), 6), device=env.device)

    # Set into physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_after_prev_gate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gate_pose: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    asset_cfg_name: str,
):
    """Reset robot position to be just before the gate it needs to fly through."""
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]

    # Extract position from gate_pose [pos(3), quat(4)]
    pos = gate_pose[:, :3].clone()
    # Add small offset to start before the gate
    from isaaclab.utils.math import quat_apply
    forward = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    gate_normal = quat_apply(gate_pose[:, 3:7], forward)
    pos = pos - gate_normal * 2.0  # Start 2m before gate

    # Add small random offset
    for i, key in enumerate(["x", "y", "z"]):
        if pose_range.get(key):
            pos[:, i] += torch.rand(len(env_ids), device=env.device) * (pose_range[key][1] - pose_range[key][0]) + pose_range[key][0]

    # Reset pose and velocity
    asset.write_root_pose_to_sim(
        torch.cat([pos, gate_pose[:, 3:7]], dim=1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device),
        env_ids=env_ids,
    )