Changelog
---------

0.1.4 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added rewarped-compatible step semantics for the differentiable Newton
  cartpole task, including terminal ``obs_before_reset`` capture and
  validation for post-reset observation behavior.

0.1.3 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added trajectory initialization and detach-safe reset semantics for the
  differentiable Newton cartpole rollout path, along with regression validation
  for reset-boundary graph isolation.

0.1.2 (2026-05-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`isaaclab_diffrl.tasks.direct.isaaclab_diffrl.newton_torch_autograd.NewtonCartpoleAutogradBridge`
  and the ``Isaac-Cartpole-DiffRL-Newton-v0`` task for differentiable single-step
  Newton cartpole rollouts.

0.1.1 (2026-05-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the ``Isaac-Cartpole-DiffRL-v0`` task registration and a headless smoke-test
  entry point for the :mod:`isaaclab_diffrl` extension.

0.1.0 (2026-05-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab