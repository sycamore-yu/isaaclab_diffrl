# Graph Report - isaaclab_diffrl  (2026-05-11)

## Corpus Check
- 55 files · ~39,469 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 387 nodes · 490 edges · 45 communities (22 shown, 23 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `NewtonCartpoleAutogradBridge` - 29 edges
2. `CartpoleNewtonEnv` - 20 edges
3. `DirectTaskProtocol` - 15 edges
4. `IsaaclabDiffrlNewtonEnv` - 15 edges
5. `CartpoleEnv` - 15 edges
6. `IsaaclabDiffrlEnv` - 14 edges
7. `MineralDirectEnvAdapter` - 11 edges
8. `main()` - 10 edges
9. `build_cfg()` - 10 edges
10. `Installation` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `NewtonCartpoleAutogradBridge`  [INFERRED]
  scripts/validate_single_step_rollout.py → source/isaaclab_diffrl/isaaclab_diffrl/rollout/newton_torch_autograd.py
- `IsaaclabDiffrlNewtonEnv` --uses--> `DirectTaskProtocol`  [INFERRED]
  source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/isaaclab_diffrl/isaaclab_diffrl_newton_env.py → source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/contracts.py
- `CartpoleNewtonEnv` --uses--> `DirectTaskProtocol`  [INFERRED]
  source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/cartpole/cartpole_newton_env.py → source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/contracts.py
- `IsaaclabDiffrlNewtonEnv` --uses--> `IsaaclabDiffrlEnv`  [INFERRED]
  source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/isaaclab_diffrl/isaaclab_diffrl_newton_env.py → source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/isaaclab_diffrl/isaaclab_diffrl_env.py
- `CartpoleNewtonEnv` --uses--> `CartpoleEnv`  [INFERRED]
  source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/cartpole/cartpole_newton_env.py → source/isaaclab_diffrl/isaaclab_diffrl/tasks/direct/cartpole/cartpole_env.py

## Communities (45 total, 23 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (39): NewtonManager, DifferentiableNewtonManager, _initialize_contacts(), initialize_solver(), Newton manager configured for differentiable single-step rollouts., start_simulation(), _sync_base_state(), _array_requires_grad() (+31 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (19): DirectTaskProtocol, Protocol for differentiable direct environments.      Establishes the shared tas, Reset the environment and detach from prior computation history., Execute one environment step with rewarped-compatible terminal observation seman, Extract the policy observation tensor from an observation dictionary., Extract the terminal policy observation tensor from step extras., Start a new differentiable rollout from the current physical state., Start a new differentiable rollout and return the initial observation. (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (7): CartpoleEnv, CartpoleEnvCfg, CartpoleNewtonEnvCfg, Torch-observation cartpole task backed by Newton physics., compute_rewards(), CartpoleEnvCfg, DirectRLEnvCfg

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (12): CartpoleNewtonEnv, Start a new differentiable rollout and return the initial observation., Return the active Newton rollout bridge, creating one on demand., Export the rollout checkpoint over Newton model/state/control tensors., Restore a rollout checkpoint without touching env bookkeeping buffers., Reset specified envs and capture terminal observation before state change., Execute one environment step with rewarped-compatible terminal observation seman, Newton-backed cartpole environment for differentiable rollout validation.      E (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (21): code:text (source/isaaclab_diffrl/isaaclab_diffrl/), code:bash (python -m pip install -e source/isaaclab_diffrl), code:bash (python scripts/list_envs.py), code:bash (python scripts/mineral/play.py \), code:bash (python scripts/mineral/train.py \), code:bash (python scripts/mineral/train.py \), code:bash (graphify update /absolute/path/to/isaaclab_diffrl --force), code:block8 (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (8): DirectRLEnv, IsaaclabDiffrlEnvCfg, IsaaclabDiffrlEnv, Start a new rollout from the current physical state without random reset., IsaaclabDiffrlNewtonEnvCfg, Torch-observation cartpole task backed by Newton physics., IsaaclabDiffrlEnvCfg, ManagerBasedRLEnvCfg

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (12): DirectTaskProtocol, IsaaclabDiffrlNewtonEnv, Start a new differentiable rollout from the current physical state.          Unl, Return the active Newton rollout bridge, creating one on demand., Export the rollout checkpoint over Newton model/state/control tensors., Restore a rollout checkpoint without touching env bookkeeping buffers., Reset specified envs and capture terminal observation before state change., Execute one environment step with rewarped-compatible terminal observation seman (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (16): InteractiveSceneCfg, ActionsCfg, EventCfg, IsaaclabDiffrlSceneCfg, ObservationsCfg, PolicyCfg, Reward terms for the MDP., Termination terms for the MDP. (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.2
Nodes (16): compute_loss(), finite_difference(), flatten_env_state(), main(), Check rollout re-init detaches a prior differentiable history.      Uses ``step_, Check env reset + bridge reinit starts a detached rollout boundary.      Warp 1., Check checkpoint export/restore round-trips rollout state only., Run the differentiable rollout validation. (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (16): Architecture decision, Architecture principles, Checkpoint, Commands, Context, Decision reasons, Differentiable direct environment, Differentiable rollout path (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (14): build_agent(), build_cfg(), main(), make_logdir(), resolve_critic_iterations(), resolve_critic_learning_rate(), resolve_launch_metadata(), resolve_learning_rate() (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (8): MineralCartpoleEnvAdapter, build_cfg(), extract_scalar(), main(), Capture timeout terminal observations for SHAC bootstrap validation., Keep the last actor and critic stats for tracer-bullet assertions., TracedCartpoleEnvAdapter, TracingSHAC

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (8): _apply_play_overrides(), build_agent(), _build_fallback_cfg(), extract_policy_obs(), main(), PolicyEnvSpec, Build minimal agent config for inference when agent.yaml is missing., resolve_checkpoint_and_metadata()

### Community 13 - "Community 13"
Cohesion: 0.33
Nodes (6): add_rsl_rl_args(), parse_rsl_rl_cfg(), Add RSL-RL arguments to the parser.      Args:         parser: The parser to add, Parse configuration for RSL-RL agent based on inputs.      Args:         task_na, Update configuration for RSL-RL agent based on inputs.      Args:         agent_, update_rsl_rl_cfg()

## Knowledge Gaps
- **112 isolated node(s):** `# NOTE: Add dependencies`, `Small report used by the tracer-bullet validator.`, `Present a differentiable direct environment through Mineral's 4-value API.`, `Run one differentiable Newton step from Torch inputs.`, `Dedicated Torch-to-Newton single-step rollout bridge for cartpole.` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NewtonCartpoleAutogradBridge` connect `Community 0` to `Community 8`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `IsaaclabDiffrlNewtonEnv` connect `Community 6` to `Community 1`, `Community 5`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `CartpoleNewtonEnv` connect `Community 3` to `Community 1`, `Community 2`, `Community 6`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `NewtonCartpoleAutogradBridge` (e.g. with `DifferentiableNewtonManager` and `._get_checkpoint_bridge()`) actually correct?**
  _`NewtonCartpoleAutogradBridge` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `CartpoleNewtonEnv` (e.g. with `DirectTaskProtocol` and `CartpoleEnv`) actually correct?**
  _`CartpoleNewtonEnv` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `DirectTaskProtocol` (e.g. with `TransitionGradReport` and `MineralDirectEnvAdapter`) actually correct?**
  _`DirectTaskProtocol` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `IsaaclabDiffrlNewtonEnv` (e.g. with `DirectTaskProtocol` and `IsaaclabDiffrlEnv`) actually correct?**
  _`IsaaclabDiffrlNewtonEnv` has 2 INFERRED edges - model-reasoned connections that need verification._