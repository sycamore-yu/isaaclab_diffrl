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
    """Run one differentiable Newton step from Torch inputs.

    The tape is stored on the **bridge** (``bridge._step_tape``), not in
    ``ctx``.  This decouples the tape's lifetime from the autograd graph:
    the tape is zeroed and released as soon as ``backward()`` completes,
    or when the next forward's ``prepare_inputs()`` detects a stale tape.
    Without this decoupling, an uncleared tape pinned the state arrays,
    causing ``Warp CUDA error 700`` on subsequent ``assign()`` / kernel
    launches during trajectory re-initialisation.
    """

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
        bridge._step_tape = tape
        return next_joint_q, next_joint_qd

    @staticmethod
    def backward(ctx, grad_joint_q: torch.Tensor, grad_joint_qd: torch.Tensor):
        bridge = ctx.bridge
        tape = bridge._step_tape
        if tape is None:
            raise RuntimeError("NewtonCartpoleStepFunction.backward: no tape found on bridge.")

        output_state = bridge.state_in if bridge.use_single_state else bridge.state_out
        grads = {
            output_state.joint_q: wp.from_torch(grad_joint_q.contiguous(), dtype=output_state.joint_q.dtype),
            output_state.joint_qd: wp.from_torch(grad_joint_qd.contiguous(), dtype=output_state.joint_qd.dtype),
        }
        tape.backward(grads=grads)

        joint_q_grad = tape.gradients.get(bridge.state_in.joint_q)
        joint_qd_grad = tape.gradients.get(bridge.state_in.joint_qd)
        grad_joint_q_in = wp.to_torch(joint_q_grad).clone() if joint_q_grad is not None else torch.zeros_like(wp.to_torch(bridge.state_in.joint_q))
        grad_joint_qd_in = wp.to_torch(joint_qd_grad).clone() if joint_qd_grad is not None else torch.zeros_like(wp.to_torch(bridge.state_in.joint_qd))
        control_grad = tape.gradients.get(bridge.control.joint_f)
        if control_grad is None:
            grad_joint_f = torch.zeros_like(wp.to_torch(bridge.control.joint_f))
        else:
            grad_joint_f = wp.to_torch(control_grad).clone()

        tape.zero()
        bridge.clear_dynamic_grads()
        bridge._step_tape = None

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
    _CHECKPOINT_TENSOR_NAMES = {
        "model": (
            "joint_act",
            "joint_f",
            "joint_q",
            "joint_qd",
            "joint_target_pos",
            "joint_target_vel",
        ),
        "state_in": ("body_f", "body_q", "body_qd", "joint_q", "joint_qd"),
        "state_out": ("body_f", "body_q", "body_qd", "joint_q", "joint_qd"),
        "control": ("joint_act", "joint_f", "joint_target_pos", "joint_target_vel"),
    }

    def __init__(self, env) -> None:
        self.env = env
        # Register on the env so env.reset() can zero any uncleared tape
        # before touching the shared Newton state arrays.
        env._active_bridge = self
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
        self._step_tape = None
        self.initialize_trajectory()

    @staticmethod
    def _array_requires_grad(array) -> bool:
        """Return whether a Warp array participates in autodiff."""
        return bool(getattr(array, "requires_grad", False))

    def _ensure_differentiable_manager(self, force: bool = False) -> None:
        """Rebuild the live Newton manager with differentiable model/state/control buffers.

        Args:
            force: If True, rebuild even when arrays already require grad
                (bypasses the Warp 1.12 issue where a previous tape lifecycle
                leaves the shared arrays in a poisoned state).
        """
        source_state = NewtonManager.get_state_0()
        source_model = NewtonManager.get_model()
        source_control = NewtonManager.get_control()
        builder = NewtonManager._builder
        if builder is None or source_state is None or source_model is None or source_control is None:
            raise RuntimeError("Newton state is unavailable; create and reset the Newton cartpole environment first.")

        if not force and self._array_requires_grad(source_state.joint_q) and self._array_requires_grad(source_control.joint_f):
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

        # Detach so env lifecycle ops (_reset_idx in-place writes) can modify
        # these leaf tensors without triggering autograd errors.
        self.env.joint_pos = wp.to_torch(self.env.robot.data.joint_pos).detach()
        self.env.joint_vel = wp.to_torch(self.env.robot.data.joint_vel).detach()

    def _detach_joint_state(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clone joint tensors into a fresh rollout boundary."""
        joint_q = joint_q.detach().clone().reshape_as(self.initial_joint_q)
        joint_qd = joint_qd.detach().clone().reshape_as(self.initial_joint_qd)
        return joint_q, joint_qd

    def _sync_joint_state(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a detached joint state into differentiable Newton buffers.

        Copies joint position/velocity into both state buffers, runs FK, and
        clears forces and gradient history.  Avoids ``state_out.assign(state_in)``
        because the solver may attach extra arrays (e.g. ``body_f_ext``) to one
        state during a differentiated step, making a full assign fail.

        If a prior ``step()`` tape was never backwarded (its ``backward()``
        zeroes it), it is zeroed here to prevent stale tape references from
        causing ``Warp CUDA error 700`` when we modify the captured arrays.
        """
        # Drop any previous tape reference.  Do NOT call tape.zero() on it --
        # Warp 1.12 tape.zero() poisons the shared state/control arrays.
        if self._step_tape is not None:
            self._step_tape = None
            self._clear_grad_refs()

        joint_q, joint_qd = self._detach_joint_state(joint_q, joint_qd)

        # Create *fresh* state/control objects from the model and copy the
        # joint values into them.  We never touch the old state arrays again
        # (they are kept alive by the pending tape's C++ state and will be
        # freed when the tape is GC'd).  New arrays are fully independent.
        self.state_in = self.model.state()
        self.state_in.joint_q.assign(wp.from_torch(joint_q.contiguous(), dtype=self.state_in.joint_q.dtype))
        self.state_in.joint_qd.assign(wp.from_torch(joint_qd.contiguous(), dtype=self.state_in.joint_qd.dtype))
        if not self.use_single_state:
            self.state_out = self.model.state()
            self.state_out.joint_q.assign(wp.from_torch(joint_q.contiguous(), dtype=self.state_out.joint_q.dtype))
            self.state_out.joint_qd.assign(wp.from_torch(joint_qd.contiguous(), dtype=self.state_out.joint_qd.dtype))
        self.control = self.model.control()
        self.control.joint_f.zero_()

        # Sync class-level references so the articulation pipeline reads
        # from the fresh arrays.
        NewtonManager._state_0 = self.state_in
        NewtonManager._state_1 = self.state_out if not self.use_single_state else self.state_in
        NewtonManager._control = self.control

        self.initial_joint_q = joint_q
        self.initial_joint_qd = joint_qd
        self._sync_env_joint_buffers(joint_q, joint_qd)
        return self.get_initial_joint_state()

    def initialize_trajectory(
        self,
        joint_q: torch.Tensor | None = None,
        joint_qd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Start a new differentiable rollout from detached current state."""
        if joint_q is None or joint_qd is None:
            joint_q, joint_qd = self.env.initialize_trajectory_from_current_state()
        return self._sync_joint_state(joint_q, joint_qd)

    @staticmethod
    def _clone_warp_tensor(array) -> torch.Tensor:
        """Clone a Warp array into a detached Torch tensor."""
        return wp.to_torch(array).detach().clone()

    @staticmethod
    def _restore_warp_tensor(array, tensor: torch.Tensor) -> None:
        """Copy a Torch tensor back into a Warp array."""
        array.assign(wp.from_torch(tensor.contiguous(), dtype=array.dtype))

    def _capture_tensor_group(self, owner, group_name: str) -> dict[str, torch.Tensor]:
        """Capture selected rollout tensors from one Newton object."""
        payload: dict[str, torch.Tensor] = {}
        for tensor_name in self._CHECKPOINT_TENSOR_NAMES[group_name]:
            array = getattr(owner, tensor_name, None)
            if array is None:
                continue
            payload[tensor_name] = self._clone_warp_tensor(array)
        return payload

    def _sync_env_joint_buffers(self, joint_q: torch.Tensor, joint_qd: torch.Tensor) -> None:
        """Mirror rollout joint state onto the env-owned observability tensors."""
        self.env.joint_pos[:] = joint_q.reshape_as(self.env.joint_pos)
        self.env.joint_vel[:] = joint_qd.reshape_as(self.env.joint_vel)

    def _restore_tensor_group(self, owner, payload: dict[str, torch.Tensor]) -> None:
        """Restore selected rollout tensors onto one Newton object."""
        for tensor_name, tensor in payload.items():
            array = getattr(owner, tensor_name, None)
            if array is None:
                continue
            self._restore_warp_tensor(array, tensor)

    def export_checkpoint(self) -> dict[str, dict[str, torch.Tensor]]:
        """Export the differentiable rollout checkpoint payload.

        The payload contains only Newton model/state/control tensors needed to
        resume the physical trajectory. Episode counters and reset buffers stay
        on the env and remain outside the payload.
        """
        return {
            "model": self._capture_tensor_group(self.model, "model"),
            "state_in": self._capture_tensor_group(self.state_in, "state_in"),
            "state_out": self._capture_tensor_group(self.state_out, "state_out"),
            "control": self._capture_tensor_group(self.control, "control"),
        }

    def restore_checkpoint(self, checkpoint: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore a differentiable rollout checkpoint payload."""
        if self._step_tape is not None:
            self._step_tape = None
            self._clear_grad_refs()

        self._restore_tensor_group(self.model, checkpoint["model"])
        self._restore_tensor_group(self.state_in, checkpoint["state_in"])
        self._restore_tensor_group(self.state_out, checkpoint["state_out"])
        self._restore_tensor_group(self.control, checkpoint["control"])
        self.clear_dynamic_grads()

        self.initial_joint_q = checkpoint["state_in"]["joint_q"].detach().clone()
        self.initial_joint_qd = checkpoint["state_in"]["joint_qd"].detach().clone()
        self.control_template_joint_f = self._clone_warp_tensor(self.control.joint_f)
        self._sync_env_joint_buffers(self.initial_joint_q, self.initial_joint_qd)

    def get_initial_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current rollout-initialized Newton joint state."""
        return self.initial_joint_q.clone(), self.initial_joint_qd.clone()

    def expand_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Map Torch cart actions to a full Newton joint-force tensor."""
        joint_f = torch.zeros_like(self.control_template_joint_f)
        joint_f[self.cart_dof_idx] = actions[0] * self.action_scale
        return joint_f

    def prepare_inputs(self, joint_q: torch.Tensor, joint_qd: torch.Tensor, joint_f: torch.Tensor) -> None:
        """Copy Torch inputs into differentiable Newton state/control buffers.

        If a stale tape from a prior ``step()`` that was never backwarded
        still exists on this bridge, it is zeroed *before* we write to the
        buffers.  This prevents an uncleared tape from holding internal
        references to the arrays and causing ``Warp CUDA error 700`` on
        subsequent kernel launches.
        """
        if self._step_tape is not None:
            self._step_tape = None
            self._clear_grad_refs()

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

    def _clear_grad_refs(self) -> None:
        """Nullify grad references after tape.zero() to prevent use-after-free."""
        for array in (self.state_in.joint_q, self.state_in.joint_qd, self.state_out.joint_q, self.state_out.joint_qd):
            array.grad = None
        if self.control.joint_f is not None:
            self.control.joint_f.grad = None

    def step_forward(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Newton step **without** a ``wp.Tape`` (no autograd).

        Use this for forward-only computations (e.g. finite-difference
        validation) where gradients are not needed.  Skips the tape
        entirely, avoiding both the performance cost and the Warp 1.12
        limitation where an uncleared tape poisons the shared state/
        control arrays for subsequent kernel launches.
        """
        joint_f = self.expand_action(actions)
        self.prepare_inputs(joint_q, joint_qd, joint_f)
        eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in, None)
        if self.use_single_state:
            self.solver.step(self.state_in, self.state_in, self.control, self.contacts, self.dt)
            next_jq = wp.to_torch(self.state_in.joint_q).clone()
            next_jqd = wp.to_torch(self.state_in.joint_qd).clone()
        else:
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, self.dt)
            next_jq = wp.to_torch(self.state_out.joint_q).clone()
            next_jqd = wp.to_torch(self.state_out.joint_qd).clone()
        return next_jq, next_jqd

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
