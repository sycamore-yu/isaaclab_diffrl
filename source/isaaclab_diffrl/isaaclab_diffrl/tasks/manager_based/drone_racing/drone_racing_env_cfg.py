# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone racing environment configuration."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg
from isaaclab.utils.math import quat_from_euler_xyz

from ....assets.obstacles.gate import GATE_CFG
from ....assets.quadrotor import FIVE_IN_DRONE_CFG
from ....quadrotor.actions import QuadrotorRateActionCfg
from ....quadrotor import observations as quad_obs
from . import mdp


# Gate configurations for the track
_GATE_CONFIGS = [
    {"pos": (5.0, 0.0, 2.0), "yaw": 0.0},
    {"pos": (5.0, 5.0, 2.0), "yaw": -3.14159 / 2},
    {"pos": (0.0, 5.0, 2.0), "yaw": 3.14159},
    {"pos": (0.0, 0.0, 2.0), "yaw": 3.14159 / 2},
]


def _make_gate_cfg(idx: int, pos: tuple, yaw: float) -> RigidObjectCfg:
    """Create a gate configuration for the given index."""
    quat = quat_from_euler_xyz(
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(yaw),
    ).tolist()
    return GATE_CFG.replace(
        prim_path=f"{{ENV_REGEX_NS}}/Gate_{idx}",
        init_state=GATE_CFG.init_state.replace(pos=pos, rot=quat),
    )


##
# Scene definition
##

@configclass
class DroneRacingSceneCfg(InteractiveSceneCfg):
    """Configuration for a drone racing scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # track - individual gates (prim path matches IsaacLab naming convention)
    gate_0: RigidObjectCfg = _make_gate_cfg(0, _GATE_CONFIGS[0]["pos"], _GATE_CONFIGS[0]["yaw"])
    gate_1: RigidObjectCfg = _make_gate_cfg(1, _GATE_CONFIGS[1]["pos"], _GATE_CONFIGS[1]["yaw"])
    gate_2: RigidObjectCfg = _make_gate_cfg(2, _GATE_CONFIGS[2]["pos"], _GATE_CONFIGS[2]["yaw"])
    gate_3: RigidObjectCfg = _make_gate_cfg(3, _GATE_CONFIGS[3]["pos"], _GATE_CONFIGS[3]["yaw"])

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

    target = mdp.GateProgressionCommandCfg(
        asset_name="robot",
        track_name="track",
        gate_size=1.5,
        randomise_start=False,
        resampling_time_range=(1e9, 1e9),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Action: [thrust, roll_rate, pitch_rate, yaw_rate]
    body_rate = QuadrotorRateActionCfg(
        asset_name="robot",
        body_name="body",
        mass=0.6076,
        inertia=[0.005, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.009],
        K_angvel=[10.0, 10.0, 10.0],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Body-frame observations + gate targeting
        target_pos_b = ObsTerm(func=quad_obs.target_pos_b, params={"command_name": "target"})
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
    reset_robot = EventTerm(
        func=mdp.reset_robot_random,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) Primary task: reach target gate (Continuous Pulling)
    pos_error = RewTerm(func=mdp.position_l2_error, weight=-3.0, params={"command_name": "target"})
    # (2) Progress along the track (Dense Reward)
    progress = RewTerm(func=mdp.progress, weight=20.0, params={"command_name": "target"})
    # (3) Gate passage incentive (Discrete Bonus - Reduced to prevent gradient shock)
    gate_passed = RewTerm(func=mdp.gate_passed, weight=100.0, params={"command_name": "target"})
    # (4) Stability and effort penalties (Differentiable loss)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.1)
    lin_vel = RewTerm(func=mdp.velocity_l2, weight=-0.01)
    # (5) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)


import isaaclab.envs.mdp as isaaclab_mdp

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=isaaclab_mdp.bad_orientation, params={"limit_angle": 1.57}) # 90 degrees
    root_height_below_minimum = DoneTerm(func=isaaclab_mdp.root_height_below_minimum, params={"minimum_height": 0.1})



@configclass
class DroneRacingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacingSceneCfg = DroneRacingSceneCfg(num_envs=64, env_spacing=8.0)
    # Basic settings
    action_scale: float = 1.0
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=FeatherstoneSolverCfg(),
            num_substeps=1,
            use_cuda_graph=False,
        )