# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate progression command for drone racing."""

from __future__ import annotations

import math
import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Gate names in the scene and their initial poses
_GATE_NAMES = ["gate_0", "gate_1", "gate_2", "gate_3"]
_GATE_POSITIONS = [
    (5.0, 0.0, 2.0),
    (5.0, 5.0, 2.0),
    (0.0, 5.0, 2.0),
    (0.0, 0.0, 2.0),
]
_GATE_YAWS = [0.0, -3.14159 / 2, 3.14159, 3.14159 / 2]


def _euler_to_quat(yaw: float) -> torch.Tensor:
    """Convert euler yaw to quaternion (w, x, y, z)."""
    import math
    half_yaw = yaw / 2.0
    return torch.tensor([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=torch.float32)


class GateProgressionCommand(CommandTerm):
    """Command generator for racing gate progression.

    Manages racing command target (next gate pose) and task-specific gate progression logic.

    Task-level checkpoint payload: racing command (next_gate_idx) and progress state.
    """

    cfg: GateProgressionCommandCfg

    def __init__(self, cfg: GateProgressionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Access individual gates from scene
        self.gates = [env.scene[name] for name in _GATE_NAMES]
        self.robot = env.scene[cfg.asset_name]
        self.num_gates = len(self.gates)

        # Precompute gate poses (same for all envs in replicated physics)
        self._gate_poses = torch.zeros(self.num_gates, 7, device=self.device)
        for i in range(self.num_gates):
            self._gate_poses[i, 0:3] = torch.tensor(_GATE_POSITIONS[i], device=self.device)
            self._gate_poses[i, 3:7] = _euler_to_quat(_GATE_YAWS[i]).to(self.device)

        # Internal state - command for each env
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)

        # Racing semantics
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_robot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        return "GateProgressionCommand"

    @property
    def command(self) -> torch.Tensor:
        """Returns the next gate pose in world frame [pos (3), quat (4)]."""
        return self.next_gate_w

    def _resample_command(self, env_ids: torch.Tensor):
        """Resample commands for specified environments."""
        if self.cfg.randomise_start:
            self.next_gate_idx[env_ids] = torch.randint(
                0, self.num_gates, (len(env_ids),), device=self.device
            )
        else:
            self.next_gate_idx[env_ids] = 0

        self._update_gate_poses(env_ids)
        # Get robot position and convert from warp if needed
        robot_pos = self.robot.data.root_pos_w
        if isinstance(robot_pos, wp.array):
            robot_pos = wp.to_torch(robot_pos)
        self._prev_robot_pos_w[env_ids] = robot_pos[env_ids]

    def _update_metrics(self):
        """Update command metrics."""
        pass

    def _update_command(self):
        # Update current gate target poses
        self._update_gate_poses(torch.arange(self.num_envs, device=self.device))

        # Check if gate passed using plane-crossing logic from upstream
        next_gate_quat = self.next_gate_w[:, 3:7]
        forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        gate_normal = quat_apply(next_gate_quat, forward_vec)

        robot_pos_w = self.robot.data.root_pos_w
        if isinstance(robot_pos_w, wp.array):
            robot_pos_w = wp.to_torch(robot_pos_w)

        # Vector from gate to drone
        rel_pos_old = self._prev_robot_pos_w - self.next_gate_w[:, :3]
        rel_pos_new = robot_pos_w - self.next_gate_w[:, :3]

        # Projection onto normal
        proj_old = torch.bmm(rel_pos_old.view(self.num_envs, 1, 3), gate_normal.view(self.num_envs, 3, 1)).squeeze()
        proj_new = torch.bmm(rel_pos_new.view(self.num_envs, 1, 3), gate_normal.view(self.num_envs, 3, 1)).squeeze()

        # Passed if it crosses the plane
        passed_plane = (proj_old < 0) & (proj_new > 0)

        # Within aperture check
        within_aperture = torch.norm(robot_pos_w - self.next_gate_w[:, :3], dim=1) < self.cfg.gate_size

        self._gate_passed[:] = passed_plane & within_aperture

        # Increment gate index for those who passed
        passed_ids = self._gate_passed.nonzero().flatten()
        if len(passed_ids) > 0:
            self.next_gate_idx[passed_ids] = (self.next_gate_idx[passed_ids] + 1) % self.num_gates
            self._update_gate_poses(passed_ids)

        # Store current pos for next step
        self._prev_robot_pos_w[:] = robot_pos_w

    def _update_gate_poses(self, env_ids: torch.Tensor):
        """Update gate poses for specified envs."""
        for env_idx in env_ids:
            gate_idx = self.next_gate_idx[env_idx].item()
            self.next_gate_w[env_idx] = self._gate_poses[gate_idx]

    @property
    def gate_passed(self) -> torch.Tensor:
        """Whether each env passed its current gate this step."""
        return self._gate_passed


@configclass
class GateProgressionCommandCfg(CommandTermCfg):
    """Configuration for gate progression command generator."""
    class_type: type = GateProgressionCommand
    asset_name: str = "robot"
    track_name: str = ""  # Not used, gates are accessed individually
    gate_size: float = 1.5
    randomise_start: bool = False