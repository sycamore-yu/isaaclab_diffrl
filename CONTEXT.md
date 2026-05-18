# Context

## Terms

### Differentiable direct environment
A Direct workflow IsaacLab environment that preserves the action → physics → reward path inside the computation graph and exposes the trajectory-control semantics required by differentiable RL algorithms.

### Trajectory initialization
Starting a new optimization rollout from the current physical state after detaching history from the previous rollout. This operation does not perform a fresh randomized reset.

### Checkpoint
A snapshot of the differentiable physical trajectory state, organized as model, state, and control tensors. Bookkeeping buffers are not part of the checkpoint payload.

### Terminal observation
The full observation dictionary computed from the pre-reset terminal state of an environment step. It is returned through `extra_info["obs_before_reset"]` and is used for SHAC terminal bootstrap.

### Step semantic compatibility
The environment computes reward, done flags, and terminal observation from the pre-reset state, then auto-resets terminated environments and returns the post-reset observation for the next step.

### Differentiable rollout path
The differentiable training rollout uses Newton's official low-level diffsim pattern inside the environment: low-level model/state/solver stepping under `wp.Tape`, while the surrounding task, registration, and lifecycle remain in IsaacLab's direct environment structure.

## Architecture decision

### Layering
- `tasks/direct/` holds shared direct-task protocol and task registration structure.
- `tasks/direct/<task>/` holds task-specific environment code, task semantics, and task-local trajectory helpers.
- `rollout/` holds Newton/Warp/Torch differentiable rollout mechanics.
- `integrations/mineral/` holds Mineral contract translation.
- `scripts/<library>/` holds CLI, launcher, config assembly, and runner startup.

### Decision reasons
- This split keeps task semantics, rollout mechanics, and algorithm integration in separate modules with strong locality.
- This split matches the IsaacLab project template: the task package owns environment registration and task code, while train/play orchestration stays in scripts.
- This split gives future direct tasks a stable seam: add a new task package, reuse shared direct-task protocol, reuse rollout mechanics, and add an integration adapter only when a library needs one.

### Architecture principles
- Name modules by scope: task names live in the task layer, backend names live in the rollout layer, and library names live in the integration layer.
- Put shared direct-task protocol in `tasks/direct/contracts.py`.
- Put cartpole task registration, environment code, and task-local helpers in `tasks/direct/cartpole/`.
- Put cartpole reward semantics in `tasks/direct/cartpole/rewards.py`.
- Keep cartpole observation extraction, termination logic, reset sampling, and post-reset rollout state handling in the cartpole task package.
- Put Newton autograd bridge, manager sync, and checkpoint handling in `rollout/`.
- Put Mineral environment adaptation in `integrations/mineral/direct_env_adapter.py`.
- Keep `mineral/cartpole_env.py` as a compatibility alias only.
- Keep `scripts/mineral/` in the script layer so the repository follows IsaacLab RL script layout.

## Validation commands

### Environment
- Run these commands inside `env_isaaclab`.

### Commands
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/play.py --checkpoint /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/logs/mineral/bptt/2026-05-11_13-59-48/ckpt/final.pth --task Isaac-Cartpole-DiffRL-Newton-v0 --num-envs 8 --viz viser`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/train.py --task Isaac-Cartpole-DiffRL-Newton-v0 --algo bptt --viz viser --num_envs 256 --max_epochs 500`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/train.py --task Isaac-Cartpole-DiffRL-Newton-v0 --algo shac --viz viser --num_envs 256 --max_epochs 500`

### Validation result
- On `2026-05-11`, the play command completed task resolution, environment setup, Viser startup, and policy loading before the external timeout ended the process.
- On `2026-05-11`, the BPTT train command entered active training epochs before the external timeout ended the process.
- On `2026-05-11`, the SHAC train command entered active training epochs before the external timeout ended the process.

## Quadrotor family

### Architecture decision
- `source/isaaclab_diffrl/isaaclab_diffrl/assets/` holds shared quadrotor and obstacle assets.
- `source/isaaclab_diffrl/isaaclab_diffrl/quadrotor/` holds shared quadrotor action and observation helpers.
- `source/isaaclab_diffrl/isaaclab_diffrl/tasks/manager_based/<task>/` holds task-owned manager-based semantics for `drone_position_control` and `drone_racing`.
- `source/isaaclab_diffrl/isaaclab_diffrl/rollout/quadrotor_bridge.py` holds the shared 13D differentiable rollout bridge for the first-wave quadrotor tasks.
- `source/isaaclab_diffrl/isaaclab_diffrl/integrations/mineral/manager_based_env_adapter.py` is the shared Mineral adapter seam for manager-based quadrotor tasks.
- `scripts/mineral/train.py` uses the shared manager-based adapter for `Isaac-Drone-Position-Control-DiffRL-v0` and `Isaac-Drone-Racing-DiffRL-v0`.
- `scripts/mineral/play.py` uses the same shared manager-based adapter and rollout bridge for quadrotor play so evaluation stays on the same semantics as training.

### Validation commands
- Run these commands inside `env_isaaclab`.

### Commands
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/test_drone_racing_bptt.py`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/test_drone_racing_shac.py`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/train.py --task Isaac-Drone-Racing-DiffRL-v0 --algo bptt --num_envs 8 --max_epochs 1 --horizon_len 8 --headless --wandb_mode disabled`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/train.py --task Isaac-Drone-Racing-DiffRL-v0 --algo shac --num_envs 8 --max_epochs 1 --horizon_len 8 --headless --wandb_mode disabled`
- `python /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/scripts/mineral/play.py --task Isaac-Drone-Racing-DiffRL-v0 --algo shac --checkpoint /home/tong/tongworkspace/isaac_develop/isaaclab_diffrl/logs/mineral/shac/2026-05-13_22-53-58/ckpt/final.pth --num-envs 1 --num-episodes 1 --headless`

### Validation result
- On `2026-05-13`, `test_drone_racing_bptt.py` completed with `tracked_steps=5`, `has_action_grad=True`, and finite `max_abs_action_grad`.
- On `2026-05-13`, `test_drone_racing_shac.py` completed with finite actor and critic tracer-bullet losses and `has_action_grad=True`.
- On `2026-05-13`, the short BPTT train command completed one epoch with finite actor loss and wrote `logs/mineral/bptt/2026-05-13_23-55-13/ckpt/final.pth`.
- On `2026-05-13`, the short SHAC train command completed one epoch with finite `value_loss=0.0027`, finite actor loss, and wrote `logs/mineral/shac/2026-05-13_22-53-58/ckpt/final.pth`.
- On `2026-05-13`, the SHAC play command completed one episode on the shared manager-based adapter path without runtime errors.
