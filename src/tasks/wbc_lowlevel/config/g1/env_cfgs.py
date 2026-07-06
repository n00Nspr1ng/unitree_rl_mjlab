"""Unitree G1 WBC low-level environment configuration."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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


# SONIC MuJoCo sim2sim plant, reproducing gear_sonic's actual run_sim_loop.py + deploy.sh sim
# path EXACTLY (verified against gear_sonic/utils/mujoco_sim + the stock Unitree MJCF
# g1_29dof_with_hand.xml loaded via scene_43dof.xml). This is NOT the training model:
#   - kp/kd: the deploy binary sends STIFFNESS_*/DAMPING_* over DDS (same as SONIC_ARTICULATION).
#   - armature: FLAT 0.01 on every joint (stock MJCF default classes), NOT per-motor. SONIC's
#     sim2sim never edits joint inertia; its fidelity comes from the PD law, not armature.
#   - passive joint damping: the MJCF adds damping=0.05 on every joint on top of the PD kd.
#     BuiltinPositionActuatorCfg has no separate joint-damping knob, but its `damping` maps to
#     the actuator's -damping*qvel term, so we fold the 0.05 in (exact for the dq_des=0 case).
#   - frictionloss: 0.2 on leg/ankle/torso/arm-class joints, 0.1 on wrist_pitch/yaw (wrist_motor).
#     Note wrist_roll is arm_motor (0.2), grouped with shoulders/elbow.
#   - effort clamp (actuatorfrcrange / motor_effort_limit_list): hip_pitch/roll=88 (NOT 139;
#     only knee=139), hip_yaw/waist_yaw=88, ankle+waist_roll/pitch=50, arms+wrist_roll=25,
#     wrist_pitch/yaw=5.
# Select this as the plant to A/B a trained policy against SONIC's real sim2sim dynamics; keep
# the action scale sourced from the TRAINING articulation (see unitree_g1_wbc_lowlevel_env_cfg).
_SIM2SIM_ARMATURE = 0.01
_PASSIVE_DAMPING = 0.05  # MJCF per-joint passive damping, folded into the actuator kd term.
SONIC_SIM2SIM_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # hip pitch / roll → 7520_22 kp/kd, but clamped at 88 Nm (old 7520_14 ceiling).
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
            stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22 + _PASSIVE_DAMPING,
            effort_limit=88.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.2,
        ),
        # knee → 7520_22, clamped at 139 Nm.
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_joint",),
            stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22 + _PASSIVE_DAMPING,
            effort_limit=139.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.2,
        ),
        # hip yaw / waist yaw → 7520_14, clamped at 88 Nm.
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
            stiffness=STIFFNESS_7520_14, damping=DAMPING_7520_14 + _PASSIVE_DAMPING,
            effort_limit=88.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.2,
        ),
        # ankle pitch/roll and waist roll/pitch → 2× 5020, clamped at 50 Nm.
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                "waist_roll_joint", "waist_pitch_joint",
            ),
            stiffness=2.0 * STIFFNESS_5020, damping=2.0 * DAMPING_5020 + _PASSIVE_DAMPING,
            effort_limit=50.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.2,
        ),
        # shoulder pitch/roll/yaw, elbow, wrist roll → 5020, clamped at 25 Nm (arm_motor, fl 0.2).
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint", ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            stiffness=STIFFNESS_5020, damping=DAMPING_5020 + _PASSIVE_DAMPING,
            effort_limit=25.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.2,
        ),
        # wrist pitch / yaw → 4010, clamped at 5 Nm (wrist_motor, fl 0.1).
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            stiffness=STIFFNESS_4010, damping=DAMPING_4010 + _PASSIVE_DAMPING,
            effort_limit=5.0, armature=_SIM2SIM_ARMATURE, frictionloss=0.1,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def _action_scale_from_articulation(
    articulation: EntityArticulationInfoCfg,
) -> dict[str, float]:
    """Build action scale as 0.25 * effort_limit / stiffness for each actuator."""
    action_scale: dict[str, float] = {}
    for actuator in articulation.actuators:
        assert isinstance(actuator, BuiltinPositionActuatorCfg)
        effort_limit = actuator.effort_limit
        stiffness = actuator.stiffness
        assert effort_limit is not None
        assert stiffness is not None
        for name in actuator.target_names_expr:
            action_scale[name] = 0.25 * effort_limit / stiffness
    return action_scale


def unitree_g1_wbc_lowlevel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 WBC low-level environment configuration."""
    cfg = make_wbc_lowlevel_env_cfg()

    robot_cfg = get_g1_robot_cfg()
    # Plant dynamics the sim integrates. Swap to SONIC_SIM2SIM_ARTICULATION to A/B a trained
    # policy against SONIC's real MuJoCo sim2sim plant (flat armature, passive damping, 88 Nm
    # hip clamp). Swap to ISAAC_ARTICULATION for the IsaacLab flat-armature training plant.
    articulation = SONIC_SIM2SIM_ARTICULATION # SONIC_SIM2SIM_ARTICULATION and MJLAB_ARTICULATION both works! ISAAC_ARTICULATION doesn't lol.
    # Action scale must always match what the policy was TRAINED under (0.25 * effort / kp with
    # the TRAINING effort/stiffness), independent of the plant. SONIC keeps action-scale (hip
    # effort 139) and the torque clamp (hip effort 88) as separate arrays for exactly this
    # reason, so keep this pinned to the training articulation — never source it from `articulation`
    # above, or selecting SONIC_SIM2SIM_ARTICULATION (hip effort 88) would silently rescale actions.
    action_scale_articulation = SONIC_ARTICULATION
    robot_cfg.init_state = ISAAC_KEYFRAME
    robot_cfg.articulation = articulation
    cfg.scene.entities = {"robot": robot_cfg}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = _action_scale_from_articulation(action_scale_articulation)

    # Fill in robot-specific body name for body_orientation reward
    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg


# IsaacLab articulation DOF order (BFS/interleaved L-R), as printed by G1LowLevelEnv at startup.
# Only used by the _backup cfg below to run OLD checkpoints whose obs joint_pos/joint_vel are in
# this order. Current training emits obs in action (MJCF/block) order, so the non-backup cfg
# needs no reorder.
_ISAAC_DOF_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def unitree_g1_wbc_lowlevel_env_cfg_backup(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Pre-migration cfg: re-adds the IsaacLab DOF-order (BFS) obs reorder on joint_pos/joint_vel,
    so OLD checkpoints (trained with BFS-order obs) still run in sim2sim. New block-order policies
    use unitree_g1_wbc_lowlevel_env_cfg instead."""
    cfg = unitree_g1_wbc_lowlevel_env_cfg(play)
    isaac_dof_cfg = SceneEntityCfg("robot", joint_names=_ISAAC_DOF_ORDER, preserve_order=True)
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["joint_pos"].params = {"asset_cfg": isaac_dof_cfg}
        terms["joint_vel"].params = {"asset_cfg": isaac_dof_cfg}
    return cfg
