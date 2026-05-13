# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch autograd bridge for one-step Newton quadrotor rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import warp as wp
from newton import eval_fk

import isaaclab.utils.math as math_utils
from isaaclab_newton.physics.newton_manager import NewtonManager
from .differentiable_newton_manager import DifferentiableNewtonManager


@dataclass
class _StepSlot:
    state_in: any
    state_out: any
    control: any
    contacts: any | None
    in_use: bool = False


class NewtonQuadrotorStepFunction(torch.autograd.Function):
    """Run one differentiable Newton step for a quadrotor."""

    @staticmethod
    def forward(
        ctx,
        bridge: "NewtonQuadrotorAutogradBridge",
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        body_f: torch.Tensor,
        body_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_index, slot = bridge.acquire_step_slot()

        # Populate buffers
        slot.state_in.joint_q.assign(wp.from_torch(joint_q.contiguous(), dtype=slot.state_in.joint_q.dtype, requires_grad=False))
        slot.state_in.joint_qd.assign(wp.from_torch(joint_qd.contiguous(), dtype=slot.state_in.joint_qd.dtype, requires_grad=False))
        slot.state_in.clear_forces()

        # Apply body forces/torques
        body_f_wp = wp.from_torch(body_f.contiguous(), dtype=wp.vec3, requires_grad=False)
        body_t_wp = wp.from_torch(body_t.contiguous(), dtype=wp.vec3, requires_grad=False)

        tape = wp.Tape()
        with tape:
            eval_fk(bridge.model, slot.state_in.joint_q, slot.state_in.joint_qd, slot.state_in, None)

            # For this tracer bullet, we assume a 6-DOF joint_f for the root
            # joint_f: [F_x, F_y, F_z, T_x, T_y, T_z]
            joint_f = torch.cat([body_f, body_t], dim=-1)
            slot.control.joint_f.assign(wp.from_torch(joint_f.contiguous(), dtype=slot.control.joint_f.dtype, requires_grad=False))

            if bridge.use_single_state:
                bridge.solver.step(slot.state_in, slot.state_in, slot.control, slot.contacts, bridge.dt)
                output_state = slot.state_in
            else:
                bridge.solver.step(slot.state_in, slot.state_out, slot.control, slot.contacts, bridge.dt)
                output_state = slot.state_out

            next_joint_q = wp.to_torch(output_state.joint_q).clone()
            next_joint_qd = wp.to_torch(output_state.joint_qd).clone()

        ctx.bridge = bridge
        ctx.slot_index = slot_index
        ctx.state_in = slot.state_in
        ctx.output_state = output_state
        ctx.control = slot.control
        ctx.tape = tape
        return next_joint_q, next_joint_qd

    @staticmethod
    def backward(ctx, grad_joint_q: torch.Tensor, grad_joint_qd: torch.Tensor):
        tape = ctx.tape
        output_state = ctx.output_state
        state_in = ctx.state_in
        control = ctx.control

        grad_joint_q = grad_joint_q.contiguous() if grad_joint_q is not None else torch.zeros_like(wp.to_torch(output_state.joint_q))
        grad_joint_qd = grad_joint_qd.contiguous() if grad_joint_qd is not None else torch.zeros_like(wp.to_torch(output_state.joint_qd))

        grads = {
            output_state.joint_q: wp.from_torch(grad_joint_q, dtype=output_state.joint_q.dtype, requires_grad=False),
            output_state.joint_qd: wp.from_torch(grad_joint_qd, dtype=output_state.joint_qd.dtype, requires_grad=False),
        }
        tape.backward(grads=grads)

        joint_q_grad = tape.gradients.get(state_in.joint_q)
        joint_qd_grad = tape.gradients.get(state_in.joint_qd)
        control_grad = tape.gradients.get(control.joint_f)

        grad_joint_q_in = wp.to_torch(joint_q_grad).clone() if joint_q_grad is not None else torch.zeros_like(wp.to_torch(state_in.joint_q))
        grad_joint_qd_in = wp.to_torch(joint_qd_grad).clone() if joint_qd_grad is not None else torch.zeros_like(wp.to_torch(state_in.joint_qd))

        if control_grad is not None:
            grad_all_f = wp.to_torch(control_grad).clone()
            grad_body_f = grad_all_f[:, 0:3]
            grad_body_t = grad_all_f[:, 3:6]
        else:
            grad_body_f = torch.zeros((ctx.bridge.num_envs, 3), device=grad_joint_q.device)
            grad_body_t = torch.zeros((ctx.bridge.num_envs, 3), device=grad_joint_q.device)

        tape.zero()
        ctx.bridge.release_step_slot(ctx.slot_index)

        return None, grad_joint_q_in, grad_joint_qd_in, grad_body_f, grad_body_t


class NewtonQuadrotorAutogradBridge:
    """Rollout bridge for quadrotor with floating base using differentiable Newton physics."""

    def __init__(self, env: any, num_envs: int) -> None:
        self.env = env
        self.num_envs = num_envs

        # Ensure DifferentiableNewtonManager is used
        self._ensure_differentiable_manager()

        self.model = NewtonManager.get_model()
        self.solver = NewtonManager._solver
        self.contacts = NewtonManager._contacts
        self.collision_pipeline = NewtonManager._collision_pipeline
        self.use_single_state = bool(NewtonManager._use_single_state)
        self.dt = NewtonManager.get_solver_dt()

        self._step_slots: list[_StepSlot] = []
        self.reserve_step_slots(1)

    def _ensure_differentiable_manager(self) -> None:
        # Simplified for tracer bullet
        if not isinstance(NewtonManager, DifferentiableNewtonManager):
            pass

    def reserve_step_slots(self, capacity: int) -> None:
        while len(self._step_slots) < capacity:
            state_in = self.model.state()
            state_out = state_in if self.use_single_state else self.model.state()
            control = self.model.control()
            contacts = self.collision_pipeline.contacts() if self.collision_pipeline is not None else self.contacts
            self._step_slots.append(_StepSlot(state_in, state_out, control, contacts))

    def acquire_step_slot(self) -> tuple[int, _StepSlot]:
        for i, slot in enumerate(self._step_slots):
            if not slot.in_use:
                slot.in_use = True
                return i, slot
        # Grow if needed
        self.reserve_step_slots(len(self._step_slots) + 1)
        return self.acquire_step_slot()

    def release_step_slot(self, index: int) -> None:
        self._step_slots[index].in_use = False

    def get_initial_state(self) -> torch.Tensor:
        return self.env.initialize_trajectory_from_current_state()

    def initialize_trajectory(self, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            return self.get_initial_state()
        return state.detach().clone()

    def expand_action(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map quadrotor rate actions [thrust, roll_rate, pitch_rate, yaw_rate] to body forces and torques.

        Returns:
            body_f: (num_envs, 3) - forces in body frame [Fx, Fy, Fz]
            body_t: (num_envs, 3) - torques in body frame [Tx, Ty, Tz]
        """
        action_view = actions.reshape(self.num_envs, -1)

        # Extract action components
        thrust = action_view[:, 0]  # normalized thrust
        roll_rate = action_view[:, 1]
        pitch_rate = action_view[:, 2]
        yaw_rate = action_view[:, 3]

        # Thrust force in body frame (Z-up convention)
        # thrust = 1.0 means hover (gravity compensation)
        gravity = 9.81
        mass = 0.6076  # 5-inch drone mass
        thrust_force = thrust * gravity * mass

        body_f = torch.zeros((self.num_envs, 3), device=actions.device, dtype=actions.dtype)
        body_f[:, 2] = thrust_force  # Fz in body frame

        # Torques from rate commands
        # K_angvel controls how aggressively we track rate commands
        K_angvel = torch.tensor([10.0, 10.0, 10.0], device=actions.device).reshape(1, 3)
        ang_vel_current = self.env.robot.data.root_ang_vel_b if hasattr(self.env, 'robot') else torch.zeros_like(action_view[:, :3])
        if isinstance(ang_vel_current, wp.array):
            ang_vel_current = wp.to_torch(ang_vel_current)

        desired_rates = torch.stack([roll_rate, pitch_rate, yaw_rate], dim=-1)
        rate_error = desired_rates - ang_vel_current
        ang_acc_desired = K_angvel * rate_error

        # Inertia (simplified diagonal)
        inertia = torch.tensor([0.005, 0.005, 0.009], device=actions.device).reshape(1, 3)
        body_t = inertia * ang_acc_desired

        return body_f, body_t

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Run one differentiable Newton step using the quadrotor's rate controller.

        The action is [thrust, roll_rate, pitch_rate, yaw_rate] following DiffAero semantics.
        """
        # Extract joint state from 13D rigid body state
        joint_q = state[:, 0:7]  # [pos(3), quat(4)]
        joint_qd = state[:, 7:13]  # [lin_vel(3), ang_vel(3)]

        # Convert action to forces and torques
        body_f, body_t = self.expand_action(action)

        # Run differentiable Newton step
        next_joint_q, next_joint_qd = NewtonQuadrotorStepFunction.apply(
            self, joint_q, joint_qd, body_f, body_t
        )

        # Reconstruct 13D state [pos, quat, lin_vel, ang_vel]
        next_state = torch.cat([next_joint_q, next_joint_qd], dim=-1)

        if next_state.shape != state.shape:
            raise RuntimeError(
                f"Unexpected quadrotor state shape {tuple(next_state.shape)}; expected {tuple(state.shape)}"
            )
        return next_state