# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Headless smoke test for the isaaclab_diffrl cartpole task.

Instantiates the registered ``Isaac-Cartpole-DiffRL-v0`` environment through
the extension entry point and runs a short simulation loop to verify the
registration and environment creation path.
"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# pass remaining args to Hydra (none by default)
sys.argv = [sys.argv[0]]

# Import extension tasks to trigger gym registration
import isaaclab_diffrl.tasks  # noqa: F401


def main():
    """Run a smoke test for the cartpole diffrl environment."""
    task_name = "Isaac-Cartpole-DiffRL-v0"
    torch.manual_seed(42)

    # Build CLI args namespace for headless launch
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(["--headless"])
    args_cli.task = task_name

    print(f"[SMOKE] Creating environment: {task_name}")

    # resolve environment configuration via Hydra
    env_cfg, _ = resolve_task_config(task_name, "")

    with launch_simulation(env_cfg, args_cli):
        # create environment with resolved config
        env = gym.make(task_name, cfg=env_cfg)

        print(f"[SMOKE] Observation space: {env.observation_space}")
        print(f"[SMOKE] Action space: {env.action_space}")

        # reset and run a few steps
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        num_steps = 10
        for i in range(num_steps):
            obs, rew, terminated, truncated, info = env.step(actions)
            print(f"[SMOKE] Step {i + 1}/{num_steps}  reward={rew.mean().item():.4f}  "
                  f"terminated={terminated.any().item()}")

        print(f"[SMOKE] SUCCESS: {task_name} instantiated and stepped through {num_steps} frames.")
        env.close()


if __name__ == "__main__":
    main()
