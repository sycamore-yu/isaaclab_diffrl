#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Short BPTT tracer-bullet validation for drone racing."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source" / "isaaclab_diffrl"
MINERAL_ROOT = REPO_ROOT.parent / "mineral"
for import_root in (SOURCE_ROOT, MINERAL_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import torch
import gymnasium as gym

from isaaclab_diffrl.tasks.manager_based.drone_racing import drone_racing_env_cfg
from isaaclab_diffrl.integrations.mineral import MineralManagerBasedEnvAdapter


def test_drone_racing_bptt():
    """Run a short headless BPTT tracer-bullet test."""
    print("[TEST] Creating DroneRacingEnvCfg...")
    env_cfg = drone_racing_env_cfg.DroneRacingEnvCfg()
    env_cfg.scene.num_envs = 8  # Small number for quick test
    env_cfg.episode_length_s = 5.0  # Short episodes
    env_cfg.decimation = 2

    print("[TEST] Making gym environment...")
    env = gym.make("Isaac-Drone-Racing-DiffRL-v0", cfg=env_cfg)
    print(f"[TEST] Environment created: {env}")

    print("[TEST] Creating MineralManagerBasedEnvAdapter...")
    adapter = MineralManagerBasedEnvAdapter(env.unwrapped)
    print(f"[TEST] Adapter created: {adapter}")
    print(f"[TEST]   num_obs: {adapter.num_obs}")
    print(f"[TEST]   num_actions: {adapter.num_actions}")
    print(f"[TEST]   episode_length: {adapter.episode_length}")

    print("[TEST] Resetting adapter...")
    obs = adapter.reset()
    print(f"[TEST] Initial observation shape: {obs.shape}")

    # Test a few steps with a zero action first
    print("[TEST] Running 5 steps with zero actions...")
    zero_action = torch.zeros(adapter.num_envs, adapter.num_actions, device=adapter.device)

    for i in range(5):
        obs, reward, done, info = adapter.step(zero_action)
        print(f"  Step {i+1}: obs shape={obs.shape}, reward sum={reward.sum().item():.3f}, done={done.any().item()}")

    # Test gradient flow with multi-step accumulation
    print("\n[TEST] Testing gradient flow with multi-step BPTT...")
    adapter.clear_grad()

    # Accumulate gradients over multiple steps (truncated BPTT)
    num_bptt_steps = 5
    accumulated_loss = 0.0
    tracked_actions = []
    step_losses = []

    for i in range(num_bptt_steps):
        action = torch.randn(adapter.num_envs, adapter.num_actions, device=adapter.device, requires_grad=True)
        tracked_actions.append(action)
        obs, reward, done, info = adapter.step(action)

        step_loss = reward.mean()
        step_losses.append(step_loss)
        accumulated_loss += step_loss.item()
        print(f"  Step {i+1}: loss={step_loss.item():.6f}, done={done.any().item()}")

    total_loss = torch.stack(step_losses).sum()
    total_loss.backward()

    print(f"\n[TEST] Accumulated loss: {accumulated_loss:.6f}")
    print("[TEST] Per-step action gradients:")
    for i, action in enumerate(tracked_actions, start=1):
        print(f"  Step {i}: action.grad is None={action.grad is None}")
        if action.grad is not None:
            print(f"    grad abs mean: {action.grad.abs().mean().item():.8f}")
            print(f"    grad abs max: {action.grad.abs().max().item():.8f}")

    # Get gradient report
    report = adapter.transition_grad_report()
    print(f"\n[TEST] Gradient report:")
    print(f"  tracked_steps: {report.tracked_steps}")
    print(f"  has_action_grad: {report.has_action_grad}")
    print(f"  max_abs_action_grad: {report.max_abs_action_grad:.8f}")

    if report.has_action_grad and report.max_abs_action_grad > 1e-9:
        print("\n[TEST] RESULT: Finite gradients confirmed through bridge!")
    else:
        print("\n[TEST] RESULT: No finite gradients detected")

    adapter.close()
    print("[TEST] Test completed successfully!")
    return report.has_action_grad and report.max_abs_action_grad > 1e-9


if __name__ == "__main__":
    success = test_drone_racing_bptt()
    sys.exit(0 if success else 1)