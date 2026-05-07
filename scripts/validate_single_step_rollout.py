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
from isaaclab_diffrl.tasks.direct.isaaclab_diffrl.newton_torch_autograd import NewtonCartpoleAutogradBridge


TASK_NAME = "Isaac-Cartpole-DiffRL-Newton-v0"


def compute_loss(obs: torch.Tensor) -> torch.Tensor:
    """Loss that depends on both cart position/velocity and pole angle."""
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

    loss_plus = compute_loss(bridge.rollout_observation(plus_joint_q, plus_joint_qd, plus_actions)).item()
    loss_minus = compute_loss(bridge.rollout_observation(minus_joint_q, minus_joint_qd, minus_actions)).item()
    return (loss_plus - loss_minus) / (2.0 * eps)


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
        bridge = NewtonCartpoleAutogradBridge(env.unwrapped)

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
