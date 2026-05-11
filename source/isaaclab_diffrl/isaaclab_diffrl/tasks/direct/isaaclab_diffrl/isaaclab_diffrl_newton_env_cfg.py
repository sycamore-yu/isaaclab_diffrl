# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

from .isaaclab_diffrl_env_cfg import IsaaclabDiffrlEnvCfg


@configclass
class IsaaclabDiffrlNewtonEnvCfg(IsaaclabDiffrlEnvCfg):
    """Torch-observation cartpole task backed by Newton physics."""

    solver_cfg = FeatherstoneSolverCfg()

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2, physics=newton_cfg)
