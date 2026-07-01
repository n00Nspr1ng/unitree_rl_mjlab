"""WBC low-level task configuration.

Observation layout matches the IsaacSim-trained G1LowLevelEnv actor obs exactly:
  actor (106): ang_vel(3) | proj_gravity(3) | vel_cmd(3) | wb_cmd(10) | q(29) | dq(29) | a(29)

Critic uses the same terms as actor (sim2sim only evaluates the actor).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.velocity.mdp import UniformVelocityCommandCfg
import src.tasks.wbc_lowlevel.mdp as mdp


def make_wbc_lowlevel_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base WBC low-level task configuration (flat terrain, no height scan)."""

    ##
    # Observations — ordering must match G1LowLevelEnv.compute_current_observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        # 10-dim whole-body command: [l_ee(3) | r_ee(3) | rpy(3) | height(1)]
        # Fixed constant for sim2sim; override params in robot-specific cfg to adjust.
        "wb_command": ObservationTermCfg(
            func=mdp.fixed_wb_command,
            params={
                "left_ee_pos": (0.20, 0.25, 0.10),
                "right_ee_pos": (0.20, -0.25, 0.10),
                "body_rpy": (0.0, 0.0, 0.0),
                "height": 0.75,
            },
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=dict(actor_terms),
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    ##
    # Actions — uniform scale=0.25 matches IsaacSim training (action_scale=0.25)
    ##

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }

    ##
    # Commands — velocity only; wb_command is injected as a fixed obs term above
    ##

    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.2,
            heading_command=False,
            heading_control_stiffness=1.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                # lin_vel_x=(-0.6, 1.0),
                lin_vel_y=(0.0, 0.0),
                # lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(0.0, 0.0),
                # ang_vel_z=(-1.0, 1.0),
            ),
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # "push_robot": EventTermCfg(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(10.0, 15.0),
        #     params={
        #         "velocity_range": {
        #             "x": (-1.0, 1.0),
        #             "y": (-1.0, 1.0),
        #         },
        #     },
        # ),
    }

    ##
    # Rewards — minimal set; focused on eval metrics, not training signal
    ##

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot
        ),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
