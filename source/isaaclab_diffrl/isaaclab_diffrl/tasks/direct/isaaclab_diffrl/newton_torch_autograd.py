# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch autograd bridge for one-step Newton cartpole rollouts."""

from __future__ import annotations

import torch
import warp as wp
from newton import eval_fk

from isaaclab_newton.physics.newton_manager import NewtonManager

from .differentiable_newton_manager import DifferentiableNewtonManager


class NewtonCartpoleStepFunction(torch.autograd.Function):
    """Run one differentiable Newton step from Torch inputs."""

    @staticmethod
    def forward(
        ctx,
        bridge: "NewtonCartpoleAutogradBridge",
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        joint_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bridge.prepare_inputs(joint_q, joint_qd, joint_f)

        tape = wp.Tape()
        with tape:
            eval_fk(bridge.model, bridge.state_in.joint_q, bridge.state_in.joint_qd, bridge.state_in, None)
            if bridge.collision_pipeline is not None and bridge.contacts is not None:
                bridge.collision_pipeline.collide(bridge.state_in, bridge.contacts)
            if bridge.use_single_state:
                bridge.solver.step(bridge.state_in, bridge.state_in, bridge.control, bridge.contacts, bridge.dt)
                next_joint_q = wp.to_torch(bridge.state_in.joint_q).clone()
                next_joint_qd = wp.to_torch(bridge.state_in.joint_qd).clone()
            else:
                bridge.solver.step(bridge.state_in, bridge.state_out, bridge.control, bridge.contacts, bridge.dt)
                next_joint_q = wp.to_torch(bridge.state_out.joint_q).clone()
                next_joint_qd = wp.to_torch(bridge.state_out.joint_qd).clone()

        ctx.bridge = bridge
        ctx.tape = tape
        return next_joint_q, next_joint_qd

    @staticmethod
    def backward(ctx, grad_joint_q: torch.Tensor, grad_joint_qd: torch.Tensor):
        bridge = ctx.bridge
        tape = ctx.tape

        output_state = bridge.state_in if bridge.use_single_state else bridge.state_out
        grads = {
            output_state.joint_q: wp.from_torch(grad_joint_q.contiguous(), dtype=output_state.joint_q.dtype),
            output_state.joint_qd: wp.from_torch(grad_joint_qd.contiguous(), dtype=output_state.joint_qd.dtype),
        }
        tape.backward(grads=grads)

        grad_joint_q_in = wp.to_torch(tape.gradients[bridge.state_in.joint_q]).clone()
        grad_joint_qd_in = wp.to_torch(tape.gradients[bridge.state_in.joint_qd]).clone()
        control_grad = tape.gradients.get(bridge.control.joint_f)
        if control_grad is None:
            grad_joint_f = torch.zeros_like(wp.to_torch(bridge.control.joint_f))
        else:
            grad_joint_f = wp.to_torch(control_grad).clone()

        tape.zero()
        bridge.clear_dynamic_grads()

        return None, grad_joint_q_in, grad_joint_qd_in, grad_joint_f


class NewtonCartpoleAutogradBridge:
    """Dedicated Torch-to-Newton single-step rollout bridge for cartpole."""

    _MANAGER_SYNC_ATTRS = (
        "_builder",
        "_num_envs",
        "_gravity_vector",
        "_up_axis",
        "_clone_physics_only",
        "_solver_dt",
        "_num_substeps",
        "_needs_collision_pipeline",
        "_use_single_state",
        "_model_changes",
        "_views",
    )

    def __init__(self, env) -> None:
        self.env = env
        self.cart_dof_idx = int(env._cart_dof_idx[0])
        self.action_scale = float(env.cfg.action_scale)

        self._ensure_differentiable_manager()

        self.model = NewtonManager.get_model()
        self.state_in = NewtonManager.get_state_0()
        self.state_out = NewtonManager.get_state_1()
        self.control = NewtonManager.get_control()
        self.solver = NewtonManager._solver
        self.contacts = NewtonManager._contacts
        self.collision_pipeline = NewtonManager._collision_pipeline
        self.use_single_state = bool(NewtonManager._use_single_state)
        self.dt = NewtonManager.get_solver_dt()

        self.initial_joint_q = wp.to_torch(self.state_in.joint_q).clone()
        self.initial_joint_qd = wp.to_torch(self.state_in.joint_qd).clone()
        self.control_template_joint_f = wp.to_torch(self.control.joint_f).clone()

    @staticmethod
    def _array_requires_grad(array) -> bool:
        """Return whether a Warp array participates in autodiff."""
        return bool(getattr(array, "requires_grad", False))

    def _ensure_differentiable_manager(self) -> None:
        """Rebuild the live Newton manager with differentiable model/state/control buffers."""
        source_state = NewtonManager.get_state_0()
        source_model = NewtonManager.get_model()
        source_control = NewtonManager.get_control()
        builder = NewtonManager._builder
        if builder is None or source_state is None or source_model is None or source_control is None:
            raise RuntimeError("Newton state is unavailable; create and reset the Newton cartpole environment first.")

        if self._array_requires_grad(source_state.joint_q) and self._array_requires_grad(source_control.joint_f):
            return

        initial_joint_q = wp.to_torch(source_state.joint_q).clone()
        initial_joint_qd = wp.to_torch(source_state.joint_qd).clone()

        diff_manager = DifferentiableNewtonManager
        for attr_name in self._MANAGER_SYNC_ATTRS:
            setattr(diff_manager, attr_name, getattr(NewtonManager, attr_name))

        diff_manager._model = None
        diff_manager._solver = None
        diff_manager._state_0 = None
        diff_manager._state_1 = None
        diff_manager._control = None
        diff_manager._contacts = None
        diff_manager._collision_pipeline = None
        diff_manager._graph = None

        diff_manager.start_simulation()
        diff_manager.initialize_solver()

        state_in = NewtonManager.get_state_0()
        state_out = NewtonManager.get_state_1()
        control = NewtonManager.get_control()
        state_in.joint_q.assign(wp.from_torch(initial_joint_q.contiguous(), dtype=state_in.joint_q.dtype))
        state_in.joint_qd.assign(wp.from_torch(initial_joint_qd.contiguous(), dtype=state_in.joint_qd.dtype))
        eval_fk(diff_manager.get_model(), state_in.joint_q, state_in.joint_qd, state_in, None)
        if state_out is not None:
            state_out.assign(state_in)
        if control is not None and control.joint_f is not None:
            control.joint_f.zero_()

        self.env.joint_pos = wp.to_torch(self.env.robot.data.joint_pos)
        self.env.joint_vel = wp.to_torch(self.env.robot.data.joint_vel)

    def get_initial_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current Newton joint state as Torch tensors."""
        return self.initial_joint_q.clone(), self.initial_joint_qd.clone()

    def expand_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Map Torch cart actions to a full Newton joint-force tensor."""
        joint_f = torch.zeros_like(self.control_template_joint_f)
        joint_f[self.cart_dof_idx] = actions[0] * self.action_scale
        return joint_f

    def prepare_inputs(self, joint_q: torch.Tensor, joint_qd: torch.Tensor, joint_f: torch.Tensor) -> None:
        """Copy Torch inputs into differentiable Newton state/control buffers."""
        self.control.joint_f.zero_()
        self.state_in.clear_forces()
        if not self.use_single_state:
            self.state_out.clear_forces()

        self.state_in.joint_q.assign(wp.from_torch(joint_q.contiguous(), dtype=self.state_in.joint_q.dtype))
        self.state_in.joint_qd.assign(wp.from_torch(joint_qd.contiguous(), dtype=self.state_in.joint_qd.dtype))
        self.control.joint_f.assign(wp.from_torch(joint_f.contiguous(), dtype=self.control.joint_f.dtype))
        self.clear_dynamic_grads()

    def clear_dynamic_grads(self) -> None:
        """Zero gradients on dynamic buffers reused across bridge calls."""
        for array in (self.state_in.joint_q, self.state_in.joint_qd, self.state_out.joint_q, self.state_out.joint_qd):
            if array.grad is not None:
                array.grad.zero_()
        if self.control.joint_f is not None and self.control.joint_f.grad is not None:
            self.control.joint_f.grad.zero_()

    def step(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one differentiable Newton step from Torch state and actions."""
        joint_f = self.expand_action(actions)
        return NewtonCartpoleStepFunction.apply(self, joint_q, joint_qd, joint_f)

    def rollout_observation(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run one step and return the cartpole observation tensor."""
        next_joint_q, next_joint_qd = self.step(joint_q, joint_qd, actions)
        return torch.stack(
            (
                next_joint_q[1],
                next_joint_qd[1],
                next_joint_q[0],
                next_joint_qd[0],
            )
        )
