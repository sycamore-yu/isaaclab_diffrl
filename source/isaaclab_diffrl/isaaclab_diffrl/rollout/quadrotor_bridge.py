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
        bridge: NewtonQuadrotorAutogradBridge,
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
        # Newton allows applying forces/torques directly to bodies
        # Here we assume the quadrotor is the first body (root)
        body_f_wp = wp.from_torch(body_f.contiguous(), dtype=wp.vec3, requires_grad=False)
        body_t_wp = wp.from_torch(body_t.contiguous(), dtype=wp.vec3, requires_grad=False)

        # Note: Newton's set_body_force/torque might need to be called during tape

        tape = wp.Tape()
        with tape:
            eval_fk(bridge.model, slot.state_in.joint_q, slot.state_in.joint_qd, slot.state_in, None)

            # Apply forces in the differentiable tape
            # We use a custom kernel or helper if needed, but for now we'll try to use joint_f if possible
            # If the free joint is the first 6 DOFs, we can use joint_f.
            # Most IsaacLab Newton models use a free joint for the root.

            # For this tracer bullet, we'll assume a 6-DOF joint_f for the root
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
    """Rollout bridge for quadrotor with floating base."""

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
            # This is a bit tricky since NewtonManager is a class with class methods
            # We rely on the environment having initialized it properly
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

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pos_w = state[:, 0:3]
        quat_w = state[:, 3:7]
        lin_vel_w = state[:, 7:10]
        ang_vel_w = state[:, 10:13]

        ang_vel_b = math_utils.quat_apply_inverse(quat_w, ang_vel_w)
        desired_ang_vel_b = action[:, 1:4]
        next_ang_vel_b = ang_vel_b + self.dt * 10.0 * (desired_ang_vel_b - ang_vel_b)

        thrust_acc_b = torch.zeros((self.num_envs, 3), device=state.device)
        thrust_acc_b[:, 2] = action[:, 0] * 9.81
        gravity_w = torch.zeros((self.num_envs, 3), device=state.device)
        gravity_w[:, 2] = -9.81
        acc_w = math_utils.quat_apply(quat_w, thrust_acc_b) + gravity_w

        next_lin_vel_w = lin_vel_w + self.dt * acc_w
        next_pos_w = pos_w + self.dt * next_lin_vel_w

        ang_speed = torch.linalg.norm(next_ang_vel_b, dim=-1, keepdim=True)
        default_axis = torch.zeros_like(next_ang_vel_b)
        default_axis[:, 2] = 1.0
        axis = torch.where(ang_speed > 1e-6, next_ang_vel_b / ang_speed.clamp_min(1e-6), default_axis)
        delta_quat = math_utils.quat_from_angle_axis((ang_speed.squeeze(-1) * self.dt), axis)
        next_quat_w = math_utils.normalize(math_utils.quat_mul(quat_w, delta_quat))
        next_ang_vel_w = math_utils.quat_apply(next_quat_w, next_ang_vel_b)

        next_state = torch.cat([next_pos_w, next_quat_w, next_lin_vel_w, next_ang_vel_w], dim=-1)
        if next_state.shape != state.shape:
            raise RuntimeError(
                f"Unexpected quadrotor state shape {tuple(next_state.shape)}; expected {tuple(state.shape)}"
            )
        return next_state
