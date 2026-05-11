# IsaacLab DiffRL

`isaaclab_diffrl` 是一个基于 Isaac Lab 的外部扩展，目标是把 **Differentiable RL** 任务、Newton/Warp 可微 rollout、以及 Mineral 算法接到同一条可运行链路上。

当前主线是 `Isaac-Cartpole-DiffRL-Newton-v0`：
- task 层提供 differentiable direct environment
- rollout 层提供 Newton/Warp/Torch 的 differentiable rollout mechanics
- integration 层把 direct environment 翻译成 Mineral 需要的训练接口
- `scripts/mineral/` 保持 IsaacLab 风格的脚本层装配

## Current layout

```text
source/isaaclab_diffrl/isaaclab_diffrl/
├── tasks/
│   └── direct/
│       ├── contracts.py
│       └── cartpole/
│           ├── __init__.py
│           ├── agents/
│           ├── cartpole_env.py
│           ├── cartpole_env_cfg.py
│           ├── cartpole_newton_env.py
│           ├── cartpole_newton_env_cfg.py
│           └── rewards.py
├── rollout/
│   ├── __init__.py
│   ├── differentiable_newton_manager.py
│   └── newton_torch_autograd.py
├── integrations/
│   └── mineral/
│       ├── __init__.py
│       └── direct_env_adapter.py
└── mineral/
    ├── __init__.py
    └── cartpole_env.py   # compatibility alias
```

## Environment

本仓库当前在 `env_isaaclab` 环境下验证。
如果你使用 IsaacLab wrapper，也可以把下面的 `python ...` 换成 `isaaclab.sh -p ...`。

安装扩展：

```bash
python -m pip install -e source/isaaclab_diffrl
```

列出任务：

```bash
python scripts/list_envs.py
```

## Main commands

### Mineral play

```bash
python scripts/mineral/play.py \
  --checkpoint logs/mineral/bptt/<run>/ckpt/final.pth \
  --task Isaac-Cartpole-DiffRL-Newton-v0 \
  --num-envs 8 \
  --viz viser
```

### Mineral BPTT train

```bash
python scripts/mineral/train.py \
  --task Isaac-Cartpole-DiffRL-Newton-v0 \
  --algo bptt \
  --viz viser \
  --num_envs 256 \
  --max_epochs 500
```

### Mineral SHAC train

```bash
python scripts/mineral/train.py \
  --task Isaac-Cartpole-DiffRL-Newton-v0 \
  --algo shac \
  --viz viser \
  --num_envs 256 \
  --max_epochs 500
```

## Validation status

在 `2026-05-11`，下列链路完成了运行期验证：
- `scripts/mineral/play.py` 完成环境创建、Viser 启动和策略加载
- `scripts/mineral/train.py --algo bptt` 进入连续训练 epoch
- `scripts/mineral/train.py --algo shac` 进入连续训练 epoch

这些验证命令的项目内记录见 `CONTEXT.md`。

## Graphify

仓库内维护 `graphify-out/`：
- `graphify-out/graph.json`
- `graphify-out/graph.html`
- `graphify-out/GRAPH_REPORT.md`

代码结构改动后可运行：

```bash
graphify update /absolute/path/to/isaaclab_diffrl --force
```

## Notes

- canonical task 路径：`isaaclab_diffrl.tasks.direct.cartpole.*`
- canonical rollout 路径：`isaaclab_diffrl.rollout.*`
- canonical Mineral integration 路径：`isaaclab_diffrl.integrations.mineral.*`
- `isaaclab_diffrl.mineral.cartpole_env` 仍保留兼容导出，方便旧调用方过渡
