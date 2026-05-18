# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Mineral BPTT or SHAC policy on an IsaacLab DiffRL task."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "isaaclab_diffrl"
MINERAL_ROOT = REPO_ROOT.parent / "mineral"
for import_root in (SOURCE_ROOT, MINERAL_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, resolve_task_config  # noqa: E402

import isaaclab_diffrl.tasks  # noqa: F401,E402
from isaaclab_diffrl.integrations.mineral import MineralDirectEnvAdapter, MineralManagerBasedEnvAdapter  # noqa: E402
from mineral.agents.diffrl.bptt import BPTT  # noqa: E402
from mineral.agents.diffrl.shac import SHAC  # noqa: E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401,E402

SUPPORTED_TASKS = {
    "Isaac-Cartpole-DiffRL-Newton-v0": "cartpole",
    "Isaac-Drone-Position-Control-DiffRL-v0": "manager",
    "Isaac-Drone-Racing-DiffRL-v0": "manager",
}


@dataclass
class PolicyEnvSpec:
    num_envs: int
    num_obs: int
    num_actions: int
    episode_length: int
    max_episode_length: int
    observation_space: object
    action_space: object


def build_agent(args_cli: argparse.Namespace, cfg: OmegaConf, logdir: Path, env_spec: PolicyEnvSpec):
    if args_cli.algo == "bptt":
        return BPTT(cfg, logdir=str(logdir), env=env_spec)
    return SHAC(cfg, logdir=str(logdir), env=env_spec)


def _build_fallback_cfg(args_cli: argparse.Namespace, rl_device: str) -> OmegaConf:
    """Build minimal agent config for inference when agent.yaml is missing."""
    network_cfg = {
        "normalize_input": True,
        "encoder": None,
        "actor": "Actor",
        "actor_kwargs": {"mlp_kwargs": {"units": [32, 32], "norm_type": "LayerNorm", "act_type": "ELU"}},
        "tanh_clamp": True,
    }
    agent_cfg = {"algo": args_cli.algo.upper(), "print_every": 0, "ckpt_every": 0, "network": network_cfg}
    optim_kwargs = {"lr": 1e-3, "betas": [0.7, 0.95]}
    if args_cli.algo == "bptt":
        agent_cfg["bptt"] = {
            "num_actors": args_cli.num_envs, "max_agent_steps": 1, "horizon_len": 1, "max_epochs": 1,
            "gamma": 0.99, "optim_type": "Adam", "actor_optim_kwargs": optim_kwargs,
            "reward_shaper": {"fn": "scale", "scale": 1.0},
        }
    else:
        network_cfg["critic"] = "Critic"
        network_cfg["critic_kwargs"] = {"mlp_kwargs": {"units": [32, 32], "norm_type": "LayerNorm", "act_type": "ELU"}}
        agent_cfg["shac"] = {
            "num_actors": args_cli.num_envs, "max_agent_steps": 1, "horizon_len": 1, "max_epochs": 1,
            "gamma": 0.99, "critic_method": "td-lambda",
            "optim_type": "Adam", "actor_optim_kwargs": optim_kwargs, "critic_optim_kwargs": optim_kwargs,
            "no_target_critic": False, "reward_shaper": {"fn": "scale", "scale": 1.0},
        }
    return OmegaConf.create({
        "seed": args_cli.seed, "multi_gpu": False, "rl_device": rl_device, "env_render": False,
        "task": {"name": args_cli.task, "env_autoresets": True}, "agent": agent_cfg,
    })


def _apply_play_overrides(agent_cfg: OmegaConf, args_cli: argparse.Namespace) -> None:
    algo_cfg = OmegaConf.select(agent_cfg, f"agent.{args_cli.algo}")
    if algo_cfg is None:
        return
    checkpoint_num_actors = OmegaConf.select(algo_cfg, "num_actors")
    if checkpoint_num_actors != args_cli.num_envs:
        print(f"[INFO] overriding checkpoint num_actors from {checkpoint_num_actors} to {args_cli.num_envs} for play")
    algo_cfg.num_actors = args_cli.num_envs


def extract_policy_obs(obs) -> dict[str, torch.Tensor]:
    if isinstance(obs, dict):
        if "obs" in obs:
            return {"obs": obs["obs"]}
        if "policy" in obs:
            return {"obs": obs["policy"]}
    return {"obs": obs}


def strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


parser = argparse.ArgumentParser(description="Play a Mineral checkpoint on an IsaacLab DiffRL task.")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-DiffRL-Newton-v0", help="Name of the task.")
parser.add_argument("--algo", type=str, default="shac", choices=("bptt", "shac"), help="Mineral algorithm.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint path. Defaults to the latest final checkpoint for the selected algorithm.",
)
parser.add_argument(
    "--num-envs",
    "--num_envs",
    dest="num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate.",
)
parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes to play before exiting.")
parser.add_argument(
    "--sample-actions",
    action="store_true",
    default=False,
    help="Sample from the action distribution instead of using deterministic means.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Sleep to match the environment step time.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video-length", type=int, default=200, help="Recorded video length in environment steps.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and agent.")
parser.add_argument("--action-scale", type=float, default=None, help="Optional environment action scale override.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    if args_cli.task not in SUPPORTED_TASKS:
        supported_tasks = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported Mineral task '{args_cli.task}'. Supported tasks: {supported_tasks}")

    if args_cli.seed is None:
        args_cli.seed = 42
    torch.manual_seed(args_cli.seed)

    checkpoint_path, agent_metadata = resolve_checkpoint_and_metadata(args_cli)
    log_dir = checkpoint_path.parent.parent

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.action_scale is not None:
        env_cfg.action_scale = args_cli.action_scale

    if agent_metadata is None:
        print(f"[INFO] agent.yaml not found, using default config")
        agent_cfg = _build_fallback_cfg(args_cli, str(env_cfg.sim.device))
    else:
        agent_cfg = OmegaConf.create(agent_metadata)
        # Auto-detect algorithm from metadata
        if "agent" in agent_cfg and "algo" in agent_cfg.agent:
            trained_algo = agent_cfg.agent.algo.lower()
            if trained_algo != args_cli.algo:
                print(f"[INFO] Auto-detecting algorithm from checkpoint: {trained_algo} (overriding {args_cli.algo})")
                args_cli.algo = trained_algo
    _apply_play_overrides(agent_cfg, args_cli)
    print(f"[INFO] task={args_cli.task}, algo={args_cli.algo}")
    print(f"[INFO] checkpoint={checkpoint_path}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        task_type = SUPPORTED_TASKS[args_cli.task]
        adapter = None
        policy_episode_length = env.unwrapped.max_episode_length
        if task_type == "cartpole":
            policy_env = PolicyEnvSpec(
                num_envs=args_cli.num_envs,
                num_obs=4,
                num_actions=1,
                episode_length=policy_episode_length,
                max_episode_length=policy_episode_length,
                observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            )
        else:
            adapter = MineralManagerBasedEnvAdapter(env.unwrapped)
            policy_env = PolicyEnvSpec(
                num_envs=args_cli.num_envs,
                num_obs=adapter.num_obs,
                num_actions=adapter.num_actions,
                episode_length=adapter.episode_length,
                max_episode_length=adapter.max_episode_length,
                observation_space=adapter.observation_space,
                action_space=adapter.action_space,
            )

        agent = build_agent(args_cli, agent_cfg, log_dir, policy_env)
        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        agent.load(str(checkpoint_path), ckpt_keys=".*")
        agent.set_eval()

        dt = env.unwrapped.step_dt
        tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard" / "play"))

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during playback.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        with torch.inference_mode():
            if adapter is None:
                raw_obs, _ = env.reset()
            else:
                raw_obs = adapter.reset()
        obs = agent._convert_obs(extract_policy_obs(raw_obs))
        episode_rewards = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=env.unwrapped.device)
        episode_lengths = torch.zeros(args_cli.num_envs, dtype=torch.int32, device=env.unwrapped.device)
        played_episodes = 0
        timestep = 0

        try:
            while played_episodes < args_cli.num_episodes:
                start_time = time.time()
                policy_obs = obs
                if agent.obs_rms is not None:
                    policy_obs = {key: agent.obs_rms[key].normalize(value) for key, value in obs.items()}
                with torch.inference_mode():
                    actions = agent.get_actions(policy_obs, sample=args_cli.sample_actions).detach()
                    if adapter is None:
                        obs, reward, terminated, truncated, _ = env.step(actions)
                        done = terminated | truncated
                    else:
                        obs, reward, done, _ = adapter.step(actions)
                        # Push diff-RL state to physics and explicitly render for visualizers
                        state = adapter._trajectory_state
                        if hasattr(env.unwrapped, "robot"):
                            env.unwrapped.robot.write_root_state_to_sim(state)
                            env.unwrapped.robot.write_data_to_sim()
                        if hasattr(env.unwrapped.sim, "render"):
                            env.unwrapped.sim.render()
                obs = agent._convert_obs(extract_policy_obs(obs))

                episode_rewards += reward
                episode_lengths += 1
                done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_env_ids.numel() > 0:
                    for done_env_id in done_env_ids.tolist():
                        rew = episode_rewards[done_env_id].item()
                        length = int(episode_lengths[done_env_id].item())
                        print(f"[PLAY] episode={played_episodes + 1} env={done_env_id} reward={rew:.2f} length={length}")
                        tb_writer.add_scalar("play/episode_reward", rew, played_episodes)
                        tb_writer.add_scalar("play/episode_length", length, played_episodes)
                        episode_rewards[done_env_id] = 0.0
                        episode_lengths[done_env_id] = 0
                        played_episodes += 1
                        if played_episodes >= args_cli.num_episodes:
                            break
                    if played_episodes >= args_cli.num_episodes:
                        break

                if args_cli.video:
                    timestep += 1
                    if timestep >= args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            tb_writer.close()
            env.close()


def resolve_checkpoint_and_metadata(args_cli: argparse.Namespace) -> tuple[Path, dict | None]:
    if args_cli.checkpoint is not None:
        checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    else:
        log_root = (REPO_ROOT / "logs" / "mineral" / args_cli.algo).resolve()
        checkpoint_path = Path(get_checkpoint_path(str(log_root), other_dirs=["ckpt"], checkpoint="final\\.pth")).resolve()

    run_dir = checkpoint_path.parent.parent
    agent_yaml = run_dir / "params" / "agent.yaml"
    agent_metadata = load_yaml(str(agent_yaml)) if agent_yaml.is_file() else None
    return checkpoint_path, agent_metadata


if __name__ == "__main__":
    main()
