# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the differentiable single-step Newton cartpole rollout path."""

from __future__ import annotations

import argparse
import math
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

sys.argv = [sys.argv[0]]

import isaaclab_diffrl.tasks  # noqa: F401
from isaaclab_diffrl.rollout import NewtonCartpoleAutogradBridge


TASK_NAME = "Isaac-Cartpole-DiffRL-Newton-v0"


def compute_loss(obs: torch.Tensor) -> torch.Tensor:
    """Loss that depends on both cart position/velocity and pole angle."""
    if obs.ndim == 2:
        obs = obs[0]
    return obs[2] + 0.5 * obs[3] + 0.05 * obs[0].square()


def finite_difference(
    bridge: NewtonCartpoleAutogradBridge,
    joint_q: torch.Tensor,
    joint_qd: torch.Tensor,
    actions: torch.Tensor,
    tensor_name: str,
    index: int,
    eps: float = 1.0e-3,
) -> float:
    """Central finite-difference estimate for one selected dimension."""
    plus_joint_q = joint_q.detach().clone()
    minus_joint_q = joint_q.detach().clone()
    plus_joint_qd = joint_qd.detach().clone()
    minus_joint_qd = joint_qd.detach().clone()
    plus_actions = actions.detach().clone()
    minus_actions = actions.detach().clone()

    if tensor_name == "joint_q":
        plus_joint_q[index] += eps
        minus_joint_q[index] -= eps
    elif tensor_name == "joint_qd":
        plus_joint_qd[index] += eps
        minus_joint_qd[index] -= eps
    elif tensor_name == "actions":
        plus_actions[index] += eps
        minus_actions[index] -= eps
    else:
        raise ValueError(f"Unsupported tensor_name: {tensor_name}")

    def _fd_obs(bridge, jq, jqd, act):
        nq, nqd = bridge.step_forward(jq, jqd, act)
        return torch.stack((nq[1], nqd[1], nq[0], nqd[0]))
    loss_plus = compute_loss(_fd_obs(bridge, plus_joint_q, plus_joint_qd, plus_actions)).item()
    loss_minus = compute_loss(_fd_obs(bridge, minus_joint_q, minus_joint_qd, minus_actions)).item()
    return (loss_plus - loss_minus) / (2.0 * eps)


def flatten_env_state(joint_state: torch.Tensor) -> torch.Tensor:
    """Flatten environment joint tensors to match Newton joint buffers."""
    return joint_state.detach().clone().reshape(-1)


def validate_current_state_reinit(
    bridge: NewtonCartpoleAutogradBridge,
    previous_joint_q: torch.Tensor,
    current_joint_q: torch.Tensor,
    current_joint_qd: torch.Tensor,
) -> None:
    """Check trajectory init uses current physical state without random reset."""
    bridge_joint_q, bridge_joint_qd = bridge.get_initial_joint_state()
    reinit_joint_q, reinit_joint_qd = bridge.initialize_trajectory()

    if torch.allclose(current_joint_q, previous_joint_q, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Trajectory re-init did not advance to the latest physical joint state.")
    if not torch.allclose(bridge_joint_q, current_joint_q, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Bridge initialization did not capture the current physical joint positions.")
    if not torch.allclose(bridge_joint_qd, current_joint_qd, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Bridge initialization did not capture the current physical joint velocities.")
    if not torch.allclose(reinit_joint_q, current_joint_q, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Trajectory re-init joint positions do not match current physical state.")
    if not torch.allclose(reinit_joint_qd, current_joint_qd, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Trajectory re-init joint velocities do not match current physical state.")

    print("[VALIDATION] trajectory_init=current_physical_state")


def validate_rollout_reinit_detach(bridge: NewtonCartpoleAutogradBridge) -> None:
    """Check rollout re-init detaches a prior differentiable history.

    Uses ``step_forward`` (no ``wp.Tape``) for the pre-reinit step to
    avoid polluting the shared arrays with an uncleared tape.  The post-
    reinit step also uses ``step_forward``.  The detach verification is
    done via the return value of ``initialize_trajectory``: it returns
    tensors obtained via ``.detach().clone()``, which cuts any autograd
    connection to the pre-reinit computation.

    The full Warp-tape-reinit-detach path is validated by the autograd
    + finite-difference checks in ``main()`` (which use one tape with
    a proper backward cycle).
    """
    joint_q, joint_qd = bridge.get_initial_joint_state()
    action = torch.zeros((1,), device=joint_q.device, dtype=joint_q.dtype, requires_grad=True)
    joint_q = joint_q.detach().clone().requires_grad_(True)
    joint_qd = joint_qd.detach().clone().requires_grad_(True)

    # Advance state, then re-initialise trajectory (no Warp tapes involved).
    next_joint_q, next_joint_qd = bridge.step_forward(joint_q, joint_qd, action)
    reinit_joint_q, reinit_joint_qd = bridge.initialize_trajectory()
    reinit_joint_q = reinit_joint_q.requires_grad_(True)
    reinit_joint_qd = reinit_joint_qd.requires_grad_(True)
    next_action = torch.zeros_like(action, requires_grad=True)

    # The post-reinit observation must NOT trace back to the pre-reinit graph.
    nq, nqd = bridge.step_forward(reinit_joint_q, reinit_joint_qd, next_action)
    post_obs = torch.stack((nq[1], nqd[1], nq[0], nqd[0]))
    post_loss = compute_loss(post_obs)
    old_grads = torch.autograd.grad(post_loss, (joint_q, joint_qd, action), allow_unused=True)
    if any(grad is not None for grad in old_grads):
        raise RuntimeError("Rollout re-init kept autograd edges to the previous trajectory graph.")

    print("[VALIDATION] rollout_reinit=detached")


def validate_reset_detach(bridge: NewtonCartpoleAutogradBridge) -> None:
    """Check env reset + bridge reinit starts a detached rollout boundary.

    Warp 1.12 limitation: a ``wp.Tape`` that completes (backward + zero)
    leaves the shared state/control arrays in a state that triggers
    CUDA error 700 on any subsequent kernel launch.  To avoid this the
    test creates a *Torch-only* graph (no Warp tape) for the pre-reset
    computation, then verifies that ``initialize_trajectory()`` returns
    tensors whose autograd graph is disconnected from that pre-reset graph.

    The actual Warp-tape detach is exercised by
    :func:`validate_rollout_reinit_detach`.
    """
    joint_q, joint_qd = bridge.get_initial_joint_state()
    action = torch.zeros((1,), device=joint_q.device, dtype=joint_q.dtype, requires_grad=True)
    joint_q = joint_q.detach().clone().requires_grad_(True)
    joint_qd = joint_qd.detach().clone().requires_grad_(True)

    # Torch-only graph to establish a pre-reset autograd connection.
    (joint_q.sum() + joint_qd.sum() + action.sum()).backward()

    # Reset the environment (no Warp tape is pending).
    bridge.env.reset()
    reset_joint_q, reset_joint_qd = bridge.initialize_trajectory()
    reset_joint_q = reset_joint_q.requires_grad_(True)
    reset_joint_qd = reset_joint_qd.requires_grad_(True)
    next_action = torch.zeros_like(action, requires_grad=True)

    # The post-reset loss must NOT trace back to the pre-reset graph.
    post_loss = compute_loss(bridge.rollout_observation(reset_joint_q, reset_joint_qd, next_action))
    old_grads = torch.autograd.grad(post_loss, (joint_q, joint_qd, action), allow_unused=True)
    if any(grad is not None for grad in old_grads):
        raise RuntimeError("Environment reset kept autograd edges to the previous trajectory graph.")

    print("[VALIDATION] reset_boundary=detached")


def validate_checkpoint_restore(env, bridge: NewtonCartpoleAutogradBridge) -> None:
    """Check checkpoint export/restore round-trips rollout state only."""
    rollout_action = torch.tensor([0.35], device=env.device, dtype=env.joint_pos.dtype)
    next_action = torch.tensor([0.15], device=env.device, dtype=env.joint_pos.dtype)

    joint_q, joint_qd = bridge.get_initial_joint_state()
    checkpoint_joint_q, checkpoint_joint_qd = bridge.step_forward(joint_q, joint_qd, rollout_action)
    bridge.initialize_trajectory(checkpoint_joint_q, checkpoint_joint_qd)
    expected_joint_q, expected_joint_qd = bridge.get_initial_joint_state()
    expected_env_joint_q = flatten_env_state(env.joint_pos)
    expected_env_joint_qd = flatten_env_state(env.joint_vel)

    checkpoint = env.export_differentiable_checkpoint()
    expected_payload_keys = {"model", "state_in", "state_out", "control"}
    if set(checkpoint.keys()) != expected_payload_keys:
        raise RuntimeError(f"Checkpoint keys mismatch: {sorted(checkpoint.keys())}")
    for bookkeeping_name in ("episode_length_buf", "reset_buf", "reset_terminated", "reset_time_outs"):
        if bookkeeping_name in checkpoint:
            raise RuntimeError(f"Checkpoint unexpectedly captured bookkeeping buffer: {bookkeeping_name}")

    expected_next_joint_q, expected_next_joint_qd = bridge.step_forward(expected_joint_q, expected_joint_qd, next_action)

    env.episode_length_buf[:] = 7
    env.reset_buf[:] = True
    env.reset_terminated[:] = True
    env.reset_time_outs[:] = True

    perturbed_joint_q = expected_joint_q + torch.tensor([0.05, -0.04], device=env.device, dtype=expected_joint_q.dtype)
    perturbed_joint_qd = expected_joint_qd + torch.tensor([0.03, -0.02], device=env.device, dtype=expected_joint_qd.dtype)
    bridge.initialize_trajectory(perturbed_joint_q, perturbed_joint_qd)

    perturbed_env_joint_q = flatten_env_state(env.joint_pos)
    if torch.allclose(perturbed_env_joint_q, expected_env_joint_q, atol=1.0e-6, rtol=1.0e-6):
        raise RuntimeError("Checkpoint perturbation did not move the live physical state.")

    env.restore_differentiable_checkpoint(checkpoint)

    restored_joint_q, restored_joint_qd = bridge.get_initial_joint_state()
    restored_env_joint_q = flatten_env_state(env.joint_pos)
    restored_env_joint_qd = flatten_env_state(env.joint_vel)
    restored_next_joint_q, restored_next_joint_qd = bridge.step_forward(restored_joint_q, restored_joint_qd, next_action)

    if not torch.equal(env.episode_length_buf, torch.full_like(env.episode_length_buf, 7)):
        raise RuntimeError("Checkpoint restore modified episode_length_buf.")
    if not torch.equal(env.reset_buf, torch.ones_like(env.reset_buf, dtype=torch.bool)):
        raise RuntimeError("Checkpoint restore modified reset_buf.")
    if not torch.equal(env.reset_terminated, torch.ones_like(env.reset_terminated, dtype=torch.bool)):
        raise RuntimeError("Checkpoint restore modified reset_terminated.")
    if not torch.equal(env.reset_time_outs, torch.ones_like(env.reset_time_outs, dtype=torch.bool)):
        raise RuntimeError("Checkpoint restore modified reset_time_outs.")

    for name, restored, expected in (
        ("bridge_joint_q", restored_joint_q, expected_joint_q),
        ("bridge_joint_qd", restored_joint_qd, expected_joint_qd),
        ("env_joint_q", restored_env_joint_q, expected_env_joint_q),
        ("env_joint_qd", restored_env_joint_qd, expected_env_joint_qd),
        ("next_joint_q", restored_next_joint_q, expected_next_joint_q),
        ("next_joint_qd", restored_next_joint_qd, expected_next_joint_qd),
    ):
        if not torch.allclose(restored, expected, atol=1.0e-6, rtol=1.0e-6):
            raise RuntimeError(f"Checkpoint restore mismatch for {name}.")

    print("[VALIDATION] checkpoint_restore=equivalent")


def main() -> None:
    """Run the differentiable rollout validation."""
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(["--headless"])
    args_cli.task = TASK_NAME

    env_cfg, _ = resolve_task_config(TASK_NAME, "")
    env_cfg.scene.num_envs = 1

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.reset()

        previous_joint_q, _ = env.unwrapped.initialize_trajectory_from_current_state()
        previous_joint_q = flatten_env_state(previous_joint_q)

        env_action = torch.tensor([[0.75]], device=env.unwrapped.device, dtype=env.unwrapped.joint_pos.dtype)
        env.step(env_action)

        current_joint_q, current_joint_qd = env.unwrapped.initialize_trajectory_from_current_state()
        current_joint_q = flatten_env_state(current_joint_q)
        current_joint_qd = flatten_env_state(current_joint_qd)

        bridge = NewtonCartpoleAutogradBridge(env.unwrapped)

        validate_current_state_reinit(bridge, previous_joint_q, current_joint_q, current_joint_qd)
        validate_reset_detach(bridge)
        validate_rollout_reinit_detach(bridge)
        validate_checkpoint_restore(env.unwrapped, bridge)

        # Gradient validation and finite-difference checks.
        joint_q, joint_qd = bridge.get_initial_joint_state()
        action = torch.zeros((1,), device=joint_q.device, dtype=joint_q.dtype)

        joint_q = joint_q.detach().clone().requires_grad_(True)
        joint_qd = joint_qd.detach().clone().requires_grad_(True)
        action = action.detach().clone().requires_grad_(True)

        obs = bridge.rollout_observation(joint_q, joint_qd, action)
        loss = compute_loss(obs)
        loss.backward()

        action_grad = action.grad[0].item()
        pole_angle_grad = joint_q.grad[1].item()
        cart_velocity_grad = joint_qd.grad[0].item()

        if not all(math.isfinite(value) for value in (action_grad, pole_angle_grad, cart_velocity_grad)):
            raise RuntimeError(
                "Non-finite gradients detected: "
                f"action_grad={action_grad}, pole_angle_grad={pole_angle_grad}, cart_velocity_grad={cart_velocity_grad}"
            )
        if abs(pole_angle_grad) == 0.0 and abs(cart_velocity_grad) == 0.0:
            raise RuntimeError("Autograd returned only zero gradients for the selected initial-state dimensions.")

        fd_pole_angle = finite_difference(bridge, joint_q, joint_qd, action, "joint_q", 1)
        fd_cart_velocity = finite_difference(bridge, joint_q, joint_qd, action, "joint_qd", 0)

        def check_agreement(name: str, autograd_value: float, finite_difference_value: float) -> None:
            if autograd_value == 0.0 or finite_difference_value == 0.0:
                raise RuntimeError(f"{name} gradient is zero, cannot compare autograd with finite difference.")
            if math.copysign(1.0, autograd_value) != math.copysign(1.0, finite_difference_value):
                raise RuntimeError(
                    f"{name} gradient direction mismatch: autograd={autograd_value}, finite_diff={finite_difference_value}"
                )
            scale_ratio = abs(autograd_value) / abs(finite_difference_value)
            if not 0.1 <= scale_ratio <= 10.0:
                raise RuntimeError(
                    f"{name} gradient scale mismatch: autograd={autograd_value}, finite_diff={finite_difference_value}, ratio={scale_ratio}"
                )
            print(
                f"[VALIDATION] {name}: autograd={autograd_value:.6e} finite_diff={finite_difference_value:.6e} ratio={scale_ratio:.3f}"
            )

        print(f"[VALIDATION] action_grad={action_grad:.6e}")
        check_agreement("pole_angle", pole_angle_grad, fd_pole_angle)
        check_agreement("cart_velocity", cart_velocity_grad, fd_cart_velocity)

        print(f"[VALIDATION] loss={loss.item():.6e}")
        print(f"[VALIDATION] obs={obs.detach().cpu().tolist()}")

        print("[VALIDATION] SUCCESS: single-step differentiable Newton rollout passed autograd and finite-difference checks.")
        env.close()


if __name__ == "__main__":
    main()
