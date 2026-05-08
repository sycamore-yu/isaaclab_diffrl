# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed cartpole environment with trajectory-init and reset-detach semantics.

Issue 03 layer:
- ``initialize_trajectory_from_current_state`` starts a new differentiable
  rollout from the current physical joint state. Returns detached tensors
  so the previous computation graph is cut.
- ``reset`` runs the full IsaacLab randomized reset, then re-initialises
  the trajectory state to the (now randomised) physical state, ensuring
  the reset boundary is a fresh graph root.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from .isaaclab_diffrl_env import IsaaclabDiffrlEnv


class IsaaclabDiffrlNewtonEnv(IsaaclabDiffrlEnv):
    """Newton-backed cartpole environment for differentiable rollout validation.

    Extends the base cartpole env with explicit trajectory-initialisation and
    reset-detach semantics required for differentiable-rollout workflows.
    """

    def _zero_bridge_tape(self) -> None:
        """Zero any uncleared tape on the active Newton bridge.

        A ``NewtonCartpoleAutogradBridge`` registers itself as
        ``self._active_bridge``.  If the bridge's last ``step()`` was never
        backwarded the tape still holds internal Warp references to the
        shared state arrays; launching any kernel (``clear_forces``,
        ``assign``, ``eval_fk``, solver internals) on those arrays would
        trigger ``CUDA error 700``.
        """
        bridge = getattr(self, "_active_bridge", None)
        if bridge is not None and bridge._step_tape is not None:
            bridge._step_tape = None
            bridge._clear_grad_refs()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment and detach from prior computation history.

        Bypasses ``DirectRLEnv.reset()`` entirely because IsaacLab's
        articulation pipeline (``scene.reset`` -> ``write_data_to_sim`` ->
        ``_apply_actuator_model``) wraps differentiable Newton buffers
        in Torch tensors and passes them to Warp kernel inputs, which
        Warp rejects.

        Instead we write a default (pole-randomised) joint state directly
        into the Newton buffers via the bridge's differentiable arrays,
        zero episode counters, and return observations.  FK will be
        evaluated by the next differentiable step's tape.
        """
        import warp as wp
        from isaaclab.utils.math import sample_uniform

        self._zero_bridge_tape()

        if seed is not None:
            self.seed(seed)

        # Write default joint state (with pole angle randomisation) directly
        # into the Newton buffers that the bridge controls.  Avoid the
        # IsaacLab articulation pipeline entirely.
        default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).detach().clone()
        pole_angle_noise = sample_uniform(
            -0.25 * math.pi, 0.25 * math.pi,
            default_joint_pos[:, self._pole_dof_idx].shape,
            default_joint_pos.device,
        )
        default_joint_pos[:, self._pole_dof_idx] += pole_angle_noise
        default_joint_vel = wp.to_torch(self.robot.data.default_joint_vel).detach().clone()

        # Write into the bridge's differentiable state via the env's robot data.
        # The bridge's state_in/state_out share these buffers with the env.
        self.joint_pos[:] = default_joint_pos
        self.joint_vel[:] = default_joint_vel

        self.episode_length_buf[:] = 0
        self.reset_buf[:] = False
        self.reset_terminated[:] = False
        self.reset_time_outs[:] = False

        # Re-read observability tensors (detached).
        self.initialize_trajectory_from_current_state()

        return self._get_observations(), self.extras

    def initialize_trajectory_from_current_state(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Start a new differentiable rollout from the current physical state.

        Unlike :meth:`reset`, this does **not** randomise; it captures the
        live joint position/velocity and returns them with ``.detach()``,
        cutting the graph edge to any preceding differentiable step.

        Args:
            env_ids: Environments to re-initialise. Defaults to all.

        Returns:
            Detached joint position and velocity tensors for each env.
        """
        return super().initialize_trajectory_from_current_state(env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset specified envs and capture terminal observation before state change.

        Overrides the base ``_reset_idx`` to store the pre-reset (terminal)
        observation in ``_cached_obs_before_reset``.  This cached observation
        is consumed by :meth:`step` to populate ``extras["obs_before_reset"]``
        per the rewarped-compatible step semantic (Issue 04).

        Clears any active bridge tape before reset-side writes can touch the
        shared Newton buffers, matching the guard in :meth:`reset`.
        """
        self._zero_bridge_tape()
        if self._cached_obs_before_reset is None:
            self._cached_obs_before_reset = self._get_observations()
        super()._reset_idx(env_ids)

    def step(self, action: torch.Tensor):
        """Execute one environment step with rewarped-compatible terminal observation semantics.

        The reward and done flags are computed from the pre-reset (terminal)
        physics state.  The terminal observation is captured before autoreset
        and stored as ``extras["obs_before_reset"]``, matching the rewarped
        convention.  After autoreset, the returned observation corresponds to
        the next post-reset state.

        Acceptance criteria (Issue 04):
        - AC1: reward and done come from the pre-reset terminal state.
        - AC2: ``extras["obs_before_reset"]`` holds the terminal observation.
        - AC3: the returned observation is from the post-reset state.
        """
        self._cached_obs_before_reset = None
        obs, rew, terminated, truncated, extras = super().step(action)
        if self._cached_obs_before_reset is not None:
            extras["obs_before_reset"] = self._cached_obs_before_reset
        else:
            extras["obs_before_reset"] = obs
        return obs, rew, terminated, truncated, extras
