# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for quadrotor assets."""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

##
# Configuration
##

# Absolute path to the 5_in_drone USD file from the reference project.
# In a production environment, this should be moved to a shared asset location.
_FIVE_IN_DRONE_USD = "/home/tong/tongworkspace/isaac_develop/isaac_drone_racer/assets/5_in_drone/5_in_drone.usd"

FIVE_IN_DRONE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FIVE_IN_DRONE_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the 5-inch drone quadrotor."""
