# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton manager variant that finalizes a differentiable model."""

from __future__ import annotations

import logging

from newton import Axis, CollisionPipeline, eval_fk

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer
from isaaclab_newton.physics.newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class DifferentiableNewtonManager(NewtonManager):
    """Newton manager configured for differentiable single-step rollouts."""

    @classmethod
    def _sync_base_state(cls) -> None:
        """Mirror subclass-managed state onto :class:`NewtonManager` accessors."""
        NewtonManager._builder = cls._builder
        NewtonManager._model = cls._model
        NewtonManager._solver = cls._solver
        NewtonManager._state_0 = cls._state_0
        NewtonManager._state_1 = cls._state_1
        NewtonManager._control = cls._control
        NewtonManager._contacts = cls._contacts
        NewtonManager._collision_pipeline = cls._collision_pipeline
        NewtonManager._num_envs = cls._num_envs
        NewtonManager._solver_dt = cls._solver_dt
        NewtonManager._num_substeps = cls._num_substeps
        NewtonManager._needs_collision_pipeline = cls._needs_collision_pipeline
        NewtonManager._use_single_state = cls._use_single_state

    @classmethod
    def start_simulation(cls) -> None:
        """Finalize the Newton model with gradient tracking enabled."""
        logger.debug(f"Builder: {cls._builder}")

        if cls._builder is None:
            cls.instantiate_builder_from_stage()

        logger.info("Dispatching MODEL_INIT callbacks")
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)

        device = PhysicsManager._device
        logger.info(f"Finalizing differentiable model on device: {device}")
        cls._builder.up_axis = Axis.from_string(cls._up_axis)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:"):
            cls._model = cls._builder.finalize(device=device, requires_grad=True)
            cls._model.set_gravity(cls._gravity_vector)
            cls._model.num_envs = cls._num_envs

        cls._state_0 = cls._model.state()
        cls._state_1 = cls._model.state()
        cls._control = cls._model.control()
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
        cls._sync_base_state()

        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        if not cls._clone_physics_only:
            import usdrt

            body_paths = getattr(cls._model, "body_label", None) or getattr(cls._model, "body_key", None)
            if body_paths is None:
                raise RuntimeError("DifferentiableNewtonManager: model has no body_label/body_key for RTX sync.")
            cls._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(body_paths):
                prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(cls._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(cls._newton_index_attr).Set(i)
                xformable_prim = usdrt.Rt.Xformable(prim)
                if not xformable_prim.HasWorldXform():
                    xformable_prim.SetWorldXformFromUsd()

            cls.sync_transforms_to_usd()

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize solver state and keep base Newton accessors synchronized."""
        super().initialize_solver()
        cls._sync_base_state()

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Create a differentiable collision pipeline when Newton contacts are needed."""
        if cls._needs_collision_pipeline:
            if cls._collision_pipeline is None:
                cls._collision_pipeline = CollisionPipeline(cls._model, broad_phase="explicit", requires_grad=True)
            if cls._contacts is None:
                cls._contacts = cls._collision_pipeline.contacts()
        else:
            super()._initialize_contacts()
        cls._sync_base_state()
