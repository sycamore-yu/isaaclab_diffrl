# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for gate obstacle assets."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

##
# Configuration
##

# Absolute path to the gate USD file from the reference project.
_GATE_USD = "/home/tong/tongworkspace/isaac_develop/isaac_drone_racer/assets/gate/gate.usd"

GATE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_GATE_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
"""Configuration for a racing gate."""
