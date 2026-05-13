# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared action logic for quadrotors."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class QuadrotorRateAction(ActionTerm):
    """Quadrotor rate control action term.

    The action contract is [normed_thrust, roll_rate, pitch_rate, yaw_rate].
    It applies a force and torque to the body frame.
    """

    cfg: QuadrotorRateActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: QuadrotorRateActionCfg, env: any) -> None:
        super().__init__(cfg, env)

        # Retrieve the asset
        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_idx = self._robot.find_bodies(self.cfg.body_name)[0]

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Controller parameters (default values from DiffAero or DroneRacer)
        self.mass = self.cfg.mass
        self.inertia = torch.tensor(self.cfg.inertia, device=self.device).reshape(3, 3)
        self.gravity = 9.81
        self.K_angvel = torch.tensor(self.cfg.K_angvel, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        # Action: [normed_thrust, roll_rate, pitch_rate, yaw_rate]
        # Body-frame angular velocity
        actual_angvel_b = self._robot.data.root_ang_vel_b

        desired_angvel_b = self._raw_actions[:, 1:]
        angvel_err = desired_angvel_b - actual_angvel_b

        # Torque = I * K * err + Ω × JΩ
        # Note: In IsaacLab, root_ang_vel_b is already in body frame.
        # Gyroscopic term
        J_omega = torch.matmul(self.inertia, actual_angvel_b.unsqueeze(-1)).squeeze(-1)
        gyroscopic_torque = torch.cross(actual_angvel_b, J_omega, dim=-1)

        # Proportional rate control
        angacc = self.K_angvel * angvel_err
        feedback_torque = torch.matmul(self.inertia, angacc.unsqueeze(-1)).squeeze(-1)

        # Total torque
        self._torques[:, 0, :] = feedback_torque + gyroscopic_torque

        # Thrust
        # thrust = normed_thrust * gravity * mass
        # In this tracer bullet, we assume action[0] is normed thrust where 1.0 = hover (gravity compensation)
        # or we follow DiffAero's convention where it's just scaled.
        # Let's use a simple scaling for now.
        thrust_val = self._raw_actions[:, 0] * self.gravity * self.mass
        self._forces[:, 0, 2] = thrust_val

        # Apply to body
        self._robot.set_external_force_and_torque(
            self._forces, self._torques, body_ids=self._body_idx
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._raw_actions[env_ids] = 0.0


@configclass
class QuadrotorRateActionCfg(ActionTermCfg):
    """Configuration for quadrotor rate action term."""

    class_type: type[ActionTerm] = QuadrotorRateAction
    asset_name: str = "robot"
    body_name: str = "body"
    mass: float = 1.0
    inertia: list[float] = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02]  # Flattened 3x3
    K_angvel: list[float] = [10.0, 10.0, 10.0]
