# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a tiny Mineral SHAC tracer bullet against the extension cartpole env."""

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
from mineral.agents.diffrl.shac import SHAC  # noqa: E402


TASK_NAME = "Isaac-Cartpole-DiffRL-Newton-v0"
LOGDIR = REPO_ROOT / "outputs" / "mineral_shac_tracer_bullet"


class TracedCartpoleEnvAdapter(MineralCartpoleEnvAdapter):
    """Capture timeout terminal observations for SHAC bootstrap validation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.timeout_terminal_obs: dict[str, torch.Tensor] | None = None
        self.timeout_terminal_obs_hits = 0

    def step(self, actions: torch.Tensor):
        obs, rew, done, extras = super().step(actions)
        if bool(extras["truncated"].any().item()):
            self.timeout_terminal_obs = extras["obs_before_reset"]
            self.timeout_terminal_obs_hits += 1
        return obs, rew, done, extras


class TracingSHAC(SHAC):
    """Keep the last actor and critic stats for tracer-bullet assertions."""

    def __init__(self, full_cfg, **kwargs):
        self.bootstrap_obs_matches = 0
        self.last_actor_results = None
        self.last_critic_results = None
        super().__init__(full_cfg, **kwargs)

    def update_actor(self):
        results = super().update_actor()
        self.last_actor_results = results
        return results

    def update_critic(self, dataset):
        results = super().update_critic(dataset)
        self.last_critic_results = results
        return results


def build_cfg(rl_device: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
            "multi_gpu": False,
            "rl_device": rl_device,
            "env_render": False,
            "task": {"name": "IsaacLabDiffRLCartpole", "env_autoresets": True},
            "agent": {
                "algo": "SHAC",
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
                    "critic": "Critic",
                    "critic_kwargs": {
                        "mlp_kwargs": {
                            "units": [32, 32],
                            "norm_type": "LayerNorm",
                            "act_type": "ELU",
                        }
                    },
                },
                "shac": {
                    "num_actors": 1,
                    "reward_shaper": {"fn": "scale", "scale": 1.0},
                    "max_agent_steps": 1,
                    "horizon_len": 1,
                    "max_epochs": 1,
                    "num_critic_batches": 1,
                    "critic_iterations": 1,
                    "optim_type": "Adam",
                    "actor_optim_kwargs": {"lr": 1.0e-3, "betas": [0.7, 0.95]},
                    "critic_optim_kwargs": {"lr": 1.0e-3, "betas": [0.7, 0.95]},
                    "lr_schedule": "constant",
                    "target_critic_alpha": 0.2,
                    "gamma": 0.99,
                    "critic_method": "one-step",
                    "normalize_ret": False,
                    "no_target_critic": False,
                    "share_encoder": True,
                    "max_grad_norm": 1.0,
                    "truncate_grads": True,
                },
            },
        }
    )


def extract_scalar(results: dict[str, list[torch.Tensor]], key: str) -> float:
    value = results[key][0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"Expected scalar tensor for {key}, got shape={tuple(value.shape)}")
        return float(value.detach().cpu().item())
    return float(value)


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

        adapter = TracedCartpoleEnvAdapter(env.unwrapped)
        adapter.episode_length = 1
        adapter.max_episode_length = 1

        timeout_probe_obs = adapter.reset()
        timeout_probe_action = torch.zeros((1, adapter.num_actions), device=adapter.device)
        timeout_probe_next_obs, _, timeout_probe_done, timeout_probe_extras = adapter.step(timeout_probe_action)
        timeout_probe_terminal_obs = timeout_probe_extras["obs_before_reset"]
        if not bool(timeout_probe_done.item()):
            raise RuntimeError("AC2 FAIL: timeout probe did not terminate on the configured one-step horizon.")
        if not bool(timeout_probe_extras["truncated"].item()):
            raise RuntimeError("AC2 FAIL: timeout probe did not report a truncated terminal step.")
        if set(timeout_probe_terminal_obs.keys()) != {"obs"}:
            raise RuntimeError(f"AC2 FAIL: terminal observation dict keys={tuple(timeout_probe_terminal_obs.keys())}")
        if torch.allclose(timeout_probe_terminal_obs["obs"], timeout_probe_next_obs):
            raise RuntimeError("AC2 FAIL: returned observation did not advance to the post-reset state.")
        if not torch.isfinite(timeout_probe_terminal_obs["obs"]).all():
            raise RuntimeError("AC2 FAIL: timeout probe terminal observation contains non-finite values.")
        if not torch.isfinite(timeout_probe_next_obs).all():
            raise RuntimeError("AC2 FAIL: timeout probe post-reset observation contains non-finite values.")

        adapter.clear_grad()
        adapter.reset()

        cfg = build_cfg(str(env.unwrapped.device))

        wandb.init(mode="disabled")
        agent = TracingSHAC(cfg, logdir=str(LOGDIR), env=adapter)

        original_encoder_target_forward = agent.encoder_target.forward

        def traced_encoder_target_forward(obs):
            terminal_obs = adapter.timeout_terminal_obs
            if terminal_obs is not None and isinstance(obs, dict) and "obs" in obs:
                if torch.allclose(obs["obs"], terminal_obs["obs"]):
                    agent.bootstrap_obs_matches += 1
            return original_encoder_target_forward(obs)

        agent.encoder_target.forward = traced_encoder_target_forward
        agent.train()
        wandb.finish()

        if adapter.timeout_terminal_obs_hits < 1:
            raise RuntimeError("AC2 FAIL: SHAC run never hit a timeout terminal step with obs_before_reset.")
        if agent.bootstrap_obs_matches < 1:
            raise RuntimeError("AC2 FAIL: SHAC bootstrap path never consumed obs_before_reset.")

        if agent.last_actor_results is None or agent.last_critic_results is None:
            raise RuntimeError("AC3 FAIL: SHAC tracer did not capture actor and critic results.")

        actor_loss = extract_scalar(agent.last_actor_results, "actor_loss")
        critic_loss = extract_scalar(agent.last_critic_results, "value_loss")
        if not torch.isfinite(torch.tensor(actor_loss)):
            raise RuntimeError(f"AC3 FAIL: actor loss is non-finite: {actor_loss}")
        if not torch.isfinite(torch.tensor(critic_loss)):
            raise RuntimeError(f"AC3 FAIL: critic loss is non-finite: {critic_loss}")

        final_ckpt = LOGDIR / "ckpt" / "final.pth"
        if not final_ckpt.is_file():
            raise RuntimeError(f"AC1/AC3 FAIL: expected training artifact missing: {final_ckpt}")

        print(f"[TRACER] task={TASK_NAME}")
        print(f"[TRACER] logdir={LOGDIR}")
        print(f"[TRACER] timeout_probe_obs_shape={tuple(timeout_probe_obs.shape)}")
        print(f"[TRACER] timeout_terminal_obs_hits={adapter.timeout_terminal_obs_hits}")
        print(f"[TRACER] bootstrap_obs_matches={agent.bootstrap_obs_matches}")
        print(f"[TRACER] actor_loss={actor_loss:.6e}")
        print(f"[TRACER] critic_loss={critic_loss:.6e}")
        print("[TRACER] AC1 PASS: Mineral SHAC used the extension-owned adapter without core library edits.")
        print("[TRACER] AC2 PASS: terminal bootstrap consumed obs_before_reset on a timeout terminal step.")
        print("[TRACER] AC3 PASS: short headless cartpole SHAC run completed with finite actor and critic losses.")

        adapter.close()


if __name__ == "__main__":
    main()
