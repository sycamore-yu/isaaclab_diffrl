# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

from ....assets.quadrotor import FIVE_IN_DRONE_CFG
from ....quadrotor.actions import QuadrotorRateActionCfg
from ....quadrotor import observations as quad_obs
from . import mdp

##
# Scene definition
##

@configclass
class DroneSceneCfg(InteractiveSceneCfg):
    """Configuration for a quadrotor scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FIVE_IN_DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Fixed target position command
    # In IsaacLab, we can use a UniformTerm or similar, but for a "fixed target"
    # we can set the range to zero or use a custom term.
    # For tracer bullet, we'll use a fixed position at (0, 0, 1)
    target_pos = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="body",
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(1.0, 1.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Action: [thrust, roll_rate, pitch_rate, yaw_rate]
    body_rate = QuadrotorRateActionCfg(
        asset_name="robot",
        body_name="body",
        mass=0.6076, # mass of 5-in drone
        inertia=[0.005, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.009], # estimate
        K_angvel=[10.0, 10.0, 10.0],
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 13D state: [pos_error (3), quat (4), lin_vel (3), ang_vel (3)]
        target_pos_b = ObsTerm(func=quad_obs.target_pos_b, params={"command_name": "target_pos"})
        root_quat_w = ObsTerm(func=quad_obs.root_quat_w)
        root_lin_vel_b = ObsTerm(func=quad_obs.root_lin_vel_b)
        root_ang_vel_b = ObsTerm(func=quad_obs.root_ang_vel_b)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    # Randomized initial position
    reset_robot_position = EventTerm(
        func=mdp.reset_root_custom_range,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_range": ((-1.0, 1.0), (-1.0, 1.0), (0.5, 1.5)),
            "quat_range": ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)), # identity
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Primary task: reach target
    pos_error = RewTerm(func=mdp.position_l2, weight=-1.0, params={"command_name": "target_pos"})
    # Effort penalties
    lin_vel = RewTerm(func=mdp.velocity_l2, weight=-0.01)
    ang_vel = RewTerm(func=mdp.ang_vel_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class DronePositionControlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneSceneCfg = DroneSceneCfg(num_envs=64, env_spacing=4.0)
    # Basic settings
    action_scale: float = 1.0
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=FeatherstoneSolverCfg(),
            num_substeps=1,
            use_cuda_graph=False,
        )
