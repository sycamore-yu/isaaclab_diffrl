# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the quadrotor task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation


def reset_root_custom_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    quat_range: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
    asset_cfg: SceneEntityCfg = None,
):
    """Reset the asset root state to a random position and orientation within a custom range."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # sample random positions
    pos_min = torch.tensor([pos_range[0][0], pos_range[1][0], pos_range[2][0]], device=asset.device)
    pos_max = torch.tensor([pos_range[0][1], pos_range[1][1], pos_range[2][1]], device=asset.device)
    positions = math_utils.sample_uniform(pos_min, pos_max, (len(env_ids), 3), device=asset.device)
    positions += env.scene.env_origins[env_ids]

    # sample random orientations
    quat_min = torch.tensor(quat_range[0], device=asset.device)
    quat_max = torch.tensor(quat_range[1], device=asset.device)
    orientations = math_utils.sample_uniform(quat_min, quat_max, (len(env_ids), 4), device=asset.device)
    orientations = torch.nn.functional.normalize(orientations, p=2, dim=-1)

    # velocities
    velocities = torch.zeros((len(env_ids), 6), device=asset.device)

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
