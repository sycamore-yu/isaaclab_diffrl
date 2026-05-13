# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym

from . import mdp

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Drone-Racing-DiffRL-v0",
    entry_point=f"{__name__}.drone_racing_env:DroneRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racing_env_cfg:DroneRacingEnvCfg",
    },
)