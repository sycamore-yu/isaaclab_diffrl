# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Mineral BPTT or SHAC against a registered IsaacLab DiffRL task."""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

import gymnasium as gym
import torch
import wandb
from isaaclab.utils.io import dump_yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "isaaclab_diffrl"
MINERAL_ROOT = REPO_ROOT.parent / "mineral"
for import_root in (SOURCE_ROOT, MINERAL_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config  # noqa: E402

import isaaclab_diffrl.tasks  # noqa: F401,E402
from isaaclab_diffrl.mineral import MineralCartpoleEnvAdapter  # noqa: E402
from mineral.agents.diffrl.bptt import BPTT  # noqa: E402
from mineral.agents.diffrl.shac import SHAC  # noqa: E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401,E402

SUPPORTED_TASKS = {
    "Isaac-Cartpole-DiffRL-Newton-v0": MineralCartpoleEnvAdapter,
}


def resolve_learning_rate(args_cli: argparse.Namespace) -> float:
    if args_cli.learning_rate is not None:
        return args_cli.learning_rate
    return 2.0e-3 if args_cli.algo == "shac" else 1.0e-3


def resolve_critic_learning_rate(args_cli: argparse.Namespace) -> float:
    if args_cli.critic_learning_rate is not None:
        return args_cli.critic_learning_rate
    if args_cli.algo == "shac":
        return 5.0e-4
    return resolve_learning_rate(args_cli)


def resolve_max_agent_steps(args_cli: argparse.Namespace) -> int:
    rollout_budget = args_cli.num_envs * args_cli.horizon_len * max(args_cli.max_epochs, 1)
    if args_cli.max_agent_steps is None:
        return rollout_budget
    return max(args_cli.max_agent_steps, rollout_budget)


def resolve_num_critic_batches(args_cli: argparse.Namespace) -> int:
    rollout_samples = max(args_cli.num_envs * args_cli.horizon_len, 1)
    if args_cli.num_critic_batches is not None:
        return max(1, min(args_cli.num_critic_batches, rollout_samples))
    if args_cli.algo == "shac":
        return min(4, rollout_samples)
    return 1


def resolve_critic_iterations(args_cli: argparse.Namespace) -> int:
    if args_cli.critic_iterations is not None:
        return args_cli.critic_iterations
    return 16 if args_cli.algo == "shac" else 1


def resolve_lr_schedule(args_cli: argparse.Namespace) -> str:
    if args_cli.lr_schedule is not None:
        return args_cli.lr_schedule
    return "linear" if args_cli.algo == "shac" else "constant"


def resolve_normalize_input(args_cli: argparse.Namespace) -> bool:
    if args_cli.normalize_input is not None:
        return args_cli.normalize_input
    return args_cli.algo == "shac"


def resolve_tanh_clamp(args_cli: argparse.Namespace) -> bool:
    if args_cli.tanh_clamp is not None:
        return args_cli.tanh_clamp
    return args_cli.algo == "shac"


def resolve_launch_metadata(args_cli: argparse.Namespace, env_cfg) -> dict:
    return {
        "task": args_cli.task,
        "algo": args_cli.algo,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "action_scale": float(env_cfg.action_scale),
        "episode_length": args_cli.episode_length,
        "horizon_len": args_cli.horizon_len,
        "max_epochs": args_cli.max_epochs,
        "max_agent_steps": resolve_max_agent_steps(args_cli),
        "actor_learning_rate": resolve_learning_rate(args_cli),
        "critic_learning_rate": resolve_critic_learning_rate(args_cli),
        "critic_method": args_cli.critic_method,
        "td_lambda": args_cli.td_lambda,
        "target_critic_alpha": args_cli.target_critic_alpha,
        "normalize_input": resolve_normalize_input(args_cli),
        "tanh_clamp": resolve_tanh_clamp(args_cli),
    }


def build_cfg(args_cli: argparse.Namespace, rl_device: str) -> OmegaConf:
    learning_rate = resolve_learning_rate(args_cli)
    critic_learning_rate = resolve_critic_learning_rate(args_cli)
    max_agent_steps = resolve_max_agent_steps(args_cli)
    num_critic_batches = resolve_num_critic_batches(args_cli)
    critic_iterations = resolve_critic_iterations(args_cli)
    lr_schedule = resolve_lr_schedule(args_cli)
    network_cfg = {
        "normalize_input": resolve_normalize_input(args_cli),
        "encoder": None,
        "actor": "Actor",
        "actor_kwargs": {
            "mlp_kwargs": {
                "units": [32, 32],
                "norm_type": "LayerNorm",
                "act_type": "ELU",
            }
        },
        "tanh_clamp": resolve_tanh_clamp(args_cli),
    }
    agent_cfg = {
        "algo": args_cli.algo.upper(),
        "print_every": args_cli.print_every,
        "ckpt_every": args_cli.ckpt_every,
        "network": network_cfg,
    }
    if args_cli.algo == "bptt":
        agent_cfg["bptt"] = {
            "num_actors": args_cli.num_envs,
            "reward_shaper": {"fn": "scale", "scale": 1.0},
            "max_agent_steps": max_agent_steps,
            "horizon_len": args_cli.horizon_len,
            "max_epochs": args_cli.max_epochs,
            "optim_type": "Adam",
            "actor_optim_kwargs": {"lr": learning_rate, "betas": [0.7, 0.95]},
            "lr_schedule": lr_schedule,
            "max_grad_norm": 1.0,
            "truncate_grads": True,
            "gamma": 0.99,
            "normalize_ret": False,
        }
    else:
        agent_cfg["network"]["critic"] = "Critic"
        agent_cfg["network"]["critic_kwargs"] = {
            "mlp_kwargs": {
                "units": [32, 32],
                "norm_type": "LayerNorm",
                "act_type": "ELU",
            }
        }
        agent_cfg["shac"] = {
            "num_actors": args_cli.num_envs,
            "reward_shaper": {"fn": "scale", "scale": 1.0},
            "max_agent_steps": max_agent_steps,
            "horizon_len": args_cli.horizon_len,
            "max_epochs": args_cli.max_epochs,
            "num_critic_batches": num_critic_batches,
            "critic_iterations": critic_iterations,
            "optim_type": "Adam",
            "actor_optim_kwargs": {"lr": learning_rate, "betas": [0.7, 0.95]},
            "critic_optim_kwargs": {"lr": critic_learning_rate, "betas": [0.7, 0.95]},
            "lr_schedule": lr_schedule,
            "target_critic_alpha": args_cli.target_critic_alpha,
            "gamma": 0.99,
            "critic_method": args_cli.critic_method,
            "normalize_ret": False,
            "no_target_critic": False,
            "share_encoder": True,
            "max_grad_norm": 1.0,
            "truncate_grads": True,
        }
        if args_cli.critic_method == "td-lambda":
            agent_cfg["shac"]["lambda"] = args_cli.td_lambda
    return OmegaConf.create(
        {
            "seed": args_cli.seed,
            "multi_gpu": False,
            "rl_device": rl_device,
            "env_render": False,
            "task": {"name": args_cli.task, "env_autoresets": True},
            "agent": agent_cfg,
        }
    )


def make_logdir(args_cli: argparse.Namespace) -> Path:
    if args_cli.logdir is not None:
        return Path(args_cli.logdir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"_{args_cli.run_name}" if args_cli.run_name else ""
    return (REPO_ROOT / "logs" / "mineral" / args_cli.algo / f"{stamp}{run_name}").resolve()


def build_agent(args_cli: argparse.Namespace, cfg: OmegaConf, logdir: Path, adapter: MineralCartpoleEnvAdapter):
    if args_cli.algo == "bptt":
        return BPTT(cfg, logdir=str(logdir), env=adapter)
    return SHAC(cfg, logdir=str(logdir), env=adapter)


parser = argparse.ArgumentParser(description="Train a Mineral agent with an IsaacLab DiffRL task.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-DiffRL-Newton-v0", help="Name of the task.")
parser.add_argument("--algo", type=str, default="bptt", choices=("bptt", "shac"), help="Mineral algorithm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment and agent.")
parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="Maximum agent steps per training run.")
parser.add_argument("--horizon_len", type=int, default=1, help="Rollout horizon length.")
parser.add_argument("--learning_rate", type=float, default=None, help="Actor learning rate for Adam optimizers.")
parser.add_argument("--critic_learning_rate", type=float, default=None, help="Optional critic learning rate override.")
parser.add_argument("--print_every", type=int, default=1, help="Logging interval in epochs.")
parser.add_argument("--ckpt_every", type=int, default=-1, help="Checkpoint interval in epochs.")
parser.add_argument("--critic_iterations", type=int, default=None, help="Number of SHAC critic iterations.")
parser.add_argument("--num_critic_batches", type=int, default=None, help="Number of SHAC critic batches.")
parser.add_argument("--critic_method", type=str, default="td-lambda", choices=("one-step", "td-lambda"), help="SHAC critic target method.")
parser.add_argument("--td_lambda", type=float, default=0.95, help="Lambda used when critic_method is td-lambda.")
parser.add_argument("--target_critic_alpha", type=float, default=0.2, help="Target critic update coefficient.")
parser.add_argument("--lr_schedule", type=str, default=None, choices=("constant", "linear"), help="Optional learning-rate schedule override.")
parser.add_argument("--normalize_input", type=lambda x: bool(strtobool(x)), default=None, nargs="?", const=True, help="Optional observation normalization override.")
parser.add_argument("--tanh_clamp", type=lambda x: bool(strtobool(x)), default=None, nargs="?", const=True, help="Optional actor tanh clamp override.")
parser.add_argument("--episode_length", type=int, default=None, help="Optional adapter episode length override.")
parser.add_argument("--action_scale", type=float, default=None, help="Optional environment action scale override.")
parser.add_argument("--wandb_mode", type=str, default="disabled", help="Weights & Biases mode.")
parser.add_argument("--logdir", type=str, default=None, help="Optional explicit log directory.")
parser.add_argument("--run_name", type=str, default=None, help="Optional log directory suffix.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    if args_cli.task not in SUPPORTED_TASKS:
        supported_tasks = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported Mineral task '{args_cli.task}'. Supported tasks: {supported_tasks}")

    torch.manual_seed(args_cli.seed)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.action_scale is not None:
        env_cfg.action_scale = args_cli.action_scale

    logdir = make_logdir(args_cli)
    logdir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args_cli, str(env_cfg.sim.device))
    launch_metadata = resolve_launch_metadata(args_cli, env_cfg)
    dump_yaml(str(logdir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(logdir / "params" / "agent.yaml"), OmegaConf.to_container(cfg, resolve=True))
    dump_yaml(str(logdir / "params" / "launch.yaml"), launch_metadata)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] algo={args_cli.algo}")
    print(f"[INFO] num_envs={args_cli.num_envs}")
    print(f"[INFO] action_scale={env_cfg.action_scale}")
    print(f"[INFO] actor_learning_rate={resolve_learning_rate(args_cli)}")
    print(f"[INFO] critic_learning_rate={resolve_critic_learning_rate(args_cli)}")
    print(f"[INFO] max_agent_steps={resolve_max_agent_steps(args_cli)}")
    print(f"[INFO] logdir={logdir}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        adapter_class = SUPPORTED_TASKS[args_cli.task]
        adapter = adapter_class(env.unwrapped)
        if args_cli.episode_length is not None:
            adapter.episode_length = args_cli.episode_length
            adapter.max_episode_length = args_cli.episode_length

        wandb.init(mode=args_cli.wandb_mode)
        try:
            agent = build_agent(args_cli, cfg, logdir, adapter)
            agent.train()
        finally:
            wandb.finish()
            adapter.close()

    final_ckpt = logdir / "ckpt" / "final.pth"
    if not final_ckpt.is_file():
        raise RuntimeError(f"Expected training artifact missing: {final_ckpt}")
    print(f"[INFO] final_checkpoint={final_ckpt}")
    print("[INFO] final_checkpoint_exists=True")


if __name__ == "__main__":
    main()
