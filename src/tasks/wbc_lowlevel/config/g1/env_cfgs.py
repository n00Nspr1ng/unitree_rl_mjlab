"""Unitree G1 WBC low-level environment configuration."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots import get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import (
    ACTUATOR_4010,
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    G1_ARTICULATION,
)
from src.tasks.wbc_lowlevel.wbc_lowlevel_env_cfg import make_wbc_lowlevel_env_cfg


ISAAC_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.8),
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_shoulder_pitch_joint": 0.35,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "right_shoulder_roll_joint": -0.18,
    },
    joint_vel={".*": 0.0},
)

# Shared effort limits and armature values (from mjlab hardware motor model, unchanged).
_E5020    = ACTUATOR_5020.effort_limit     # 25 Nm
_E7520_14 = ACTUATOR_7520_14.effort_limit  # 88 Nm
_E7520_22 = ACTUATOR_7520_22.effort_limit  # 139 Nm
_E4010    = ACTUATOR_4010.effort_limit     # 5 Nm
_A5020    = ARMATURE_5020
_A7520_14 = ARMATURE_7520_14
_A7520_22 = ARMATURE_7520_22
_A4010    = ARMATURE_4010

# mjlab default gains (velocity/tracking tasks, g1_constants.py G1_ARTICULATION).
# Physics-based kp/kd same formula as SONIC, but hip_pitch → 7520_14 (kp≈40)
# instead of 7520_22 (kp≈99). Unitree later upgraded the hip_pitch motor.
MJLAB_ARTICULATION = G1_ARTICULATION

# IsaacSim training gains -- exact match of assets/unitree/unitree.py G1_CFG
# (ImplicitActuatorCfg stiffness/damping/effort_limit_sim/armature, verified joint-by-joint
# against the actually-imported file, not the dead assets/unitree.py). IsaacLab used a flat
# armature=0.01 on every joint (not the physically-derived per-motor reflected inertia), and
# the ankle/waist-roll/waist-pitch parallel-linkage joints used effort_limit_sim=35 (not
# 2x the single-motor limit). Use this when testing policy_v7.pt under the same gains it was
# trained with.
_TRAIN_ARMATURE = 0.01
ISAAC_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint",),
            stiffness=200.0, damping=5.0,
            effort_limit=88.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_yaw_joint",),
            stiffness=150.0, damping=5.0,
            effort_limit=88.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_roll_joint",),
            stiffness=150.0, damping=5.0,
            effort_limit=139.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_joint",),
            stiffness=200.0, damping=5.0,
            effort_limit=139.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            stiffness=20.0, damping=2.0,
            effort_limit=35.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("waist_yaw_joint",),
            stiffness=200.0, damping=5.0,
            effort_limit=88.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
            stiffness=200.0, damping=5.0,
            effort_limit=35.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
            stiffness=100.0, damping=2.0,
            effort_limit=25.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_shoulder_yaw_joint", ".*_elbow_joint"),
            stiffness=50.0, damping=2.0,
            effort_limit=25.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_wrist_roll_joint",),
            stiffness=40.0, damping=2.0,
            effort_limit=25.0, armature=_TRAIN_ARMATURE,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            stiffness=40.0, damping=2.0,
            effort_limit=5.0, armature=_TRAIN_ARMATURE,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)

# SONIC hardware deployment PD gains (gear_sonic_deploy/policy_parameters.hpp).
# kp = armature × ω²,  kd = 2ζ × armature × ω,  ω=10Hz×2π, ζ=2.
# In the SONIC sim loop the C++ binary sends these kp/kd per joint via DDS,
# so the sim applies exactly these gains — matching what runs on real hardware.
# Ankle and waist roll/pitch use 2× because of the parallel linkage (2 actuators/joint).
SONIC_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # hip pitch / roll / knee → 7520_22
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
            stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22,
            effort_limit=_E7520_22, armature=_A7520_22,
        ),
        # hip yaw / waist yaw → 7520_14
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
            stiffness=STIFFNESS_7520_14, damping=DAMPING_7520_14,
            effort_limit=_E7520_14, armature=_A7520_14,
        ),
        # ankle pitch/roll and waist roll/pitch → 2× 5020 parallel linkage
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                "waist_roll_joint", "waist_pitch_joint",
            ),
            stiffness=2.0 * STIFFNESS_5020, damping=2.0 * DAMPING_5020,
            effort_limit=_E5020 * 2, armature=_A5020 * 2,
        ),
        # shoulder pitch/roll/yaw, elbow, wrist roll → 5020
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint", ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            stiffness=STIFFNESS_5020, damping=DAMPING_5020,
            effort_limit=_E5020, armature=_A5020,
        ),
        # wrist pitch / yaw → 4010
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            stiffness=STIFFNESS_4010, damping=DAMPING_4010,
            effort_limit=_E4010, armature=_A4010,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def unitree_g1_wbc_lowlevel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 WBC low-level environment configuration."""
    cfg = make_wbc_lowlevel_env_cfg()

    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state = ISAAC_KEYFRAME
    robot_cfg.articulation = ISAAC_ARTICULATION #SONIC_ARTICULATION
    cfg.scene.entities = {"robot": robot_cfg}

    # Fill in robot-specific body name for body_orientation reward
    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg
