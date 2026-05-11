# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a tiny Mineral APG/BPTT tracer bullet against the extension cartpole env."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import torch
import wandb
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "isaaclab_diffrl"
MINERAL_ROOT = REPO_ROOT.parent / "mineral"
for import_root in (SOURCE_ROOT, MINERAL_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config  # noqa: E402

sys.argv = [sys.argv[0]]

import isaaclab_diffrl.tasks  # noqa: F401,E402
from isaaclab_diffrl.mineral import MineralCartpoleEnvAdapter  # noqa: E402
from mineral.agents.diffrl.bptt import BPTT  # noqa: E402


TASK_NAME = "Isaac-Cartpole-DiffRL-Newton-v0"
LOGDIR = REPO_ROOT / "outputs" / "mineral_bptt_tracer_bullet"


def build_cfg(rl_device: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
            "multi_gpu": False,
            "rl_device": rl_device,
            "env_render": False,
            "task": {"name": "IsaacLabDiffRLCartpole", "env_autoresets": True},
            "agent": {
                "algo": "BPTT",
                "print_every": 1,
                "ckpt_every": -1,
                "network": {
                    "normalize_input": False,
                    "encoder": None,
                    "actor": "Actor",
                    "actor_kwargs": {
                        "mlp_kwargs": {
                            "units": [32, 32],
                            "norm_type": "LayerNorm",
                            "act_type": "ELU",
                        }
                    },
                },
                "bptt": {
                    "num_actors": 1,
                    "reward_shaper": {"fn": "scale", "scale": 1.0},
                    "max_agent_steps": 1,
                    "horizon_len": 1,
                    "max_epochs": 1,
                    "optim_type": "Adam",
                    "actor_optim_kwargs": {"lr": 1.0e-3, "betas": [0.7, 0.95]},
                    "lr_schedule": "constant",
                    "max_grad_norm": 1.0,
                    "truncate_grads": True,
                    "gamma": 0.99,
                    "normalize_ret": False,
                },
            },
        }
    )


def main() -> None:
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(["--headless"])
    args_cli.task = TASK_NAME

    env_cfg, _ = resolve_task_config(TASK_NAME, "")
    env_cfg.scene.num_envs = 1

    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.reset()

        adapter = MineralCartpoleEnvAdapter(env.unwrapped)
        cfg = build_cfg(str(env.unwrapped.device))

        wandb.init(mode="disabled")
        agent = BPTT(cfg, logdir=str(LOGDIR), env=adapter)
        agent.train()
        grad_report = adapter.transition_grad_report()
        wandb.finish()

        if grad_report.tracked_steps < 1:
            raise RuntimeError("AC2/AC3 FAIL: tracer run did not record a differentiable rollout step.")
        if not grad_report.has_action_grad:
            raise RuntimeError(
                "AC3 FAIL: actor_loss.backward() did not propagate a finite non-zero action gradient through physics. "
                f"max_abs_action_grad={grad_report.max_abs_action_grad:.6e}"
            )

        final_ckpt = LOGDIR / "ckpt" / "final.pth"
        if not final_ckpt.is_file():
            raise RuntimeError(f"AC2 FAIL: expected training artifact missing: {final_ckpt}")

        print(f"[TRACER] task={TASK_NAME}")
        print(f"[TRACER] logdir={LOGDIR}")
        print(f"[TRACER] tracked_steps={grad_report.tracked_steps}")
        print(f"[TRACER] max_abs_action_grad={grad_report.max_abs_action_grad:.6e}")
        print("[TRACER] AC1 PASS: Mineral BPTT used the adapter without core library edits.")
        print("[TRACER] AC2 PASS: short headless cartpole training run completed.")
        print("[TRACER] AC3 PASS: backward propagated through the differentiable Newton transition.")

        adapter.close()


if __name__ == "__main__":
    main()
