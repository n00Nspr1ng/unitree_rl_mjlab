"""Sim2sim evaluation script for TorchScript policies exported from IsaacSim.

Usage:
    python scripts/sim2sim.py Unitree-G1-WbcLowlevel \\
        --checkpoint /path/to/policy_v7.pt \\
        [--num-envs 1] [--device cuda:0] [--viewer native|viser]

The checkpoint must be a TorchScript module (torch.jit.load) that maps
obs (B, obs_dim) -> action (B, action_dim).
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import matrix_from_quat

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


# Scripted high-level command sequences, mirroring scripts/play_lowlevel_fake.py.
# Each row is 13-dim: [vx, vy, ang_vel_z, l_ee(x,y,z), r_ee(x,y,z), body_rpy(r,p,y), height].
# The first 3 drive the "twist" velocity command; the last 10 drive the wb_command obs term.
PRESET_SEQUENCES: dict[str, list[list[float]]] = {
    "neutral": [
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
    ],
    "forward": [
        [0.2, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.75],
        [0.5, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.65],
    ],
    "turn": [
        [0.0, 0.0, 0.6, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.0, 0.0, -0.6, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
    ],
    "sway": [
        [0.0, 0.2, 0.0, 0.22, 0.30, 0.15, 0.18, -0.20, 0.12, 0.0, 0.0, 0.0, 0.82],
        [0.0, 0.48, 0.0, 0.22, 0.30, 0.15, 0.18, -0.20, 0.12, 0.0, 0.0, 0.0, 0.82],
        [0.0, -0.2, 0.0, 0.18, 0.20, 0.12, 0.22, -0.30, 0.15, 0.0, 0.0, -0.0, 0.78],
        [0.0, -0.48, 0.0, 0.18, 0.20, 0.12, 0.22, -0.30, 0.15, 0.0, 0.0, -0.0, 0.78],
    ],
    "combined": [
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.0, 0.0, 0.0, 0.30, 0.15, 0.30, 0.30, -0.15, 0.30, 0.0, 0.0, 0.0, 0.65],
        [0.3, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.5, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.65],
        [0.0, 0.0, 1.2, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.0, 0.0, -1.2, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.0, 0.5, 0.0, 0.22, 0.30, 0.15, 0.18, -0.20, 0.12, 0.0, 0.0, 0.2, 0.82],
        [0.0, -0.5, 0.0, 0.18, 0.20, 0.12, 0.22, -0.30, 0.15, 0.0, 0.0, -0.2, 0.78],
    ],
    "tilt": [
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.80],
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, -0.45, 0.0, 0.0, 0.80],
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.45, 0.0, 0.0, 0.80],
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 1.50, 0.0, 0.80],
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 1.50, 0.0, 0.55],
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.55],
    ],
    "sim2sim_test": [
        [0.0, 0.0, 0.0, 0.20, 0.25, 0.10, 0.20, -0.25, 0.10, 0.0, 0.0, 0.0, 0.75],
    ],
}


@dataclass(frozen=True)
class Sim2SimConfig:
    checkpoint: Path
    """Path to TorchScript policy file (.pt exported via torch.jit.save)."""
    num_envs: int = 1
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    debug_steps: int = 0
    """If > 0, skip the viewer and run this many steps headless, printing
    per-step obs/action/state diagnostics instead."""
    sequence: str = ""
    """Name of a PRESET_SEQUENCES entry to drive scripted twist + wb_command over time.
    Empty uses the env's default fixed command."""
    segment_s: float = 4.0
    """Seconds to hold each command segment before advancing to the next."""


def _freeze_twist_resample(raw_env) -> None:
    """Stop the twist command from randomly resampling so scripted values persist.

    Overriding _resample_command to a no-op also prevents is_standing_env from ever being
    set, so _update_command won't zero the injected command (see UniformVelocityCommand)."""
    twist = raw_env.command_manager.get_term("twist")
    twist._resample_command = lambda env_ids: None
    twist.is_standing_env[:] = False
    if hasattr(twist, "is_heading_env"):
        twist.is_heading_env[:] = False


def _apply_command13(raw_env, cmd13) -> None:
    """Drive twist (vel) + wb_command from a 13-dim [vx,vy,wz, l_ee3, r_ee3, rpy3, height]."""
    twist = raw_env.command_manager.get_term("twist")
    twist.vel_command_b[:, 0] = cmd13[0]
    twist.vel_command_b[:, 1] = cmd13[1]
    twist.vel_command_b[:, 2] = cmd13[2]
    # wb_command obs term reads these params fresh on each observation_manager.compute().
    wb = raw_env.observation_manager.get_term_cfg("actor", "wb_command").params
    wb["left_ee_pos"] = (cmd13[3], cmd13[4], cmd13[5])
    wb["right_ee_pos"] = (cmd13[6], cmd13[7], cmd13[8])
    wb["body_rpy"] = (cmd13[9], cmd13[10], cmd13[11])
    wb["height"] = cmd13[12]


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """World-frame RPY (roll about X, pitch about Y, yaw about Z) -> 3x3 = Rz @ Ry @ Rx."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _install_command_visualizers(raw_env) -> None:
    """Draw target torso frame, target L/R eef frames, and target linear velocity.

    Replaces the default twist debug arrows (4 cluttered arrows) with a clean target set.
    Reads the live wb_command obs params + twist velocity, so it tracks scripted commands.
    """
    robot = raw_env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]

    # Disable the default velocity command arrows (target ang vel + actual vel + misplaced).
    try:
        raw_env.command_manager.get_term("twist")._debug_vis_enabled = False
    except Exception:
        pass

    def draw_targets(visualizer) -> None:
        wb = raw_env.observation_manager.get_term_cfg("actor", "wb_command").params
        l_ee = np.asarray(wb["left_ee_pos"], dtype=float)
        r_ee = np.asarray(wb["right_ee_pos"], dtype=float)
        # body_rpy is a WORLD-frame euler (the reward compares it to euler_xyz_from_quat of the
        # torso's world quat), so the target torso orientation is world-absolute.
        tgt_rot = _rpy_to_matrix(*wb["body_rpy"])
        height = float(wb["height"])

        # EEF commands are in the ACTUAL torso frame (see reward track_ee_pos_exp:
        # combine_frame_transforms(torso_pos_w, torso_quat_w, ee_pos_b)), NOT the target frame.
        torso_pos = robot.data.body_link_pos_w[:, torso_id].cpu().numpy()
        torso_mat = matrix_from_quat(robot.data.body_link_quat_w[:, torso_id]).cpu().numpy()
        base_pos = robot.data.root_link_pos_w.cpu().numpy()
        base_mat = matrix_from_quat(robot.data.root_link_quat_w).cpu().numpy()
        twist = raw_env.command_manager.get_command("twist").cpu().numpy()

        for b in visualizer.get_env_indices(raw_env.num_envs):
            tp = torso_pos[b]
            if np.linalg.norm(tp) < 1e-6:
                continue
            # Target torso frame: actual torso xy, target height, target world orientation.
            visualizer.add_frame(
                np.array([tp[0], tp[1], height]), tgt_rot, scale=0.35, label="torso_target"
            )
            # Target eef frames: ee_pos_b expressed in the actual torso frame.
            l_w = tp + torso_mat[b] @ l_ee
            r_w = tp + torso_mat[b] @ r_ee
            visualizer.add_frame(l_w, torso_mat[b], scale=0.12, label="l_ee_target")
            visualizer.add_frame(r_w, torso_mat[b], scale=0.12, label="r_ee_target")
            visualizer.add_sphere(l_w, radius=0.03, color=(0.9, 0.2, 0.2, 0.9))
            visualizer.add_sphere(r_w, radius=0.03, color=(0.2, 0.2, 0.9, 0.9))
            # Target linear velocity arrow (base frame -> world), yellow.
            start = base_pos[b] + np.array([0.0, 0.0, 0.05])
            direction = base_mat[b] @ np.array([twist[b][0], twist[b][1], 0.0])
            visualizer.add_arrow(
                start, start + direction, color=(0.95, 0.85, 0.1, 0.9), width=0.02,
                label="vel_target",
            )

    orig_update = raw_env.update_visualizers

    def wrapped(visualizer) -> None:
        orig_update(visualizer)
        draw_targets(visualizer)

    raw_env.update_visualizers = wrapped


class _ScriptedPolicy:
    """Wraps the policy and, once per step, advances a scripted command schedule.

    Called exactly once per env step by both the viewer and _run_debug. The command set
    here lands in the *next* step's observation (one-step lag, negligible per segment)."""

    def __init__(self, model, raw_env, sequence, steps_per_segment):
        self._model = model
        self._raw_env = raw_env
        self._sequence = sequence
        self._steps_per_segment = max(1, steps_per_segment)
        self._step = 0
        self._last_seg = -1

    def reset(self):
        self._step = 0
        self._last_seg = -1

    def __call__(self, obs):
        seg = (self._step // self._steps_per_segment) % len(self._sequence)
        if seg != self._last_seg:
            print(f"[sim2sim] segment {seg}: {self._sequence[seg]}")
            self._last_seg = seg
        _apply_command13(self._raw_env, self._sequence[seg])
        self._step += 1
        tensor = obs["actor"] if hasattr(obs, "__getitem__") else obs
        return self._model(tensor)


def _run_debug(env, policy, num_steps: int) -> None:
    """Headless step loop with per-step diagnostics (bypasses the viewer)."""
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    num_actions = raw_env.action_manager.total_action_dim

    # Actor obs layout (see wbc_lowlevel_env_cfg.py):
    # ang_vel(3) | proj_gravity(3) | vel_cmd(3) | wb_cmd(10) | q(n) | dq(n) | a(n)
    prefix = 3 + 3 + 3 + 10
    slices = {
        "ang_vel": slice(0, 3),
        "proj_grav": slice(3, 6),
        "vel_cmd": slice(6, 9),
        "wb_cmd": slice(9, prefix),
        "q": slice(prefix, prefix + num_actions),
        "dq": slice(prefix + num_actions, prefix + 2 * num_actions),
        "a": slice(prefix + 2 * num_actions, prefix + 3 * num_actions),
    }

    def full(t: torch.Tensor) -> str:
        return "[" + ", ".join(f"{v:+.3f}" for v in t.tolist()) + "]"

    obs, _ = env.reset()
    for step in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs_t = obs["actor"][0] if hasattr(obs, "__getitem__") else obs[0]
        print(f"\n--- step {step} ---")
        print(f"  action      : {full(actions[0])}")
        for name, sl in slices.items():
            print(f"  obs.{name:<9}: {full(obs_t[sl])}")

        height = robot.data.root_link_pos_w[0, 2].item()
        proj_grav_z = robot.data.projected_gravity_b[0, 2].item()
        print(f"  root height : {height:.3f}  proj_grav_z : {proj_grav_z:+.3f}")
        try:
            force = robot.data.actuator_force[0]
            print(f"  actuator_force: {full(force)}")
        except Exception as e:  # property may not exist on this mjlab version
            print(f"  actuator_force: <unavailable: {e}>")

        obs, rew, dones, extras = env.step(actions)
        print(f"  reward={rew[0].item():.4f} done={bool(dones[0])}")
        if bool(dones[0]):
            for term_name in raw_env.termination_manager.active_terms:
                val = raw_env.termination_manager.get_term(term_name)[0].item()
                if val:
                    print(f"  TERMINATED by: {term_name}")
            obs, _ = env.reset()


def main():
    import mjlab.tasks  # noqa: F401 — populate built-in task registry
    import src.tasks    # noqa: F401 — populate custom task registry

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg: Sim2SimConfig = tyro.cli(
        Sim2SimConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()
    device = cfg.device

    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    # Load TorchScript policy and wrap to unpack the TensorDict that
    # RslRlVecEnvWrapper returns (keyed by obs group name "actor").
    _model = torch.jit.load(str(cfg.checkpoint), map_location=device)
    _model.eval()
    print(f"[INFO] Loaded TorchScript policy from {cfg.checkpoint}")

    class _Policy:
        def __call__(self, obs):
            tensor = obs["actor"] if hasattr(obs, "__getitem__") else obs
            return _model(tensor)

    policy = _Policy()

    # Build env in play mode (no curriculum, infinite episode length, no noise)
    env_cfg = load_env_cfg(chosen_task, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled")

    agent_cfg = load_rl_cfg(chosen_task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Replace the default twist arrows with target torso/eef frames + target velocity.
    _install_command_visualizers(env.unwrapped)

    # Scripted command schedule (optional): drive twist + wb_command over time.
    if cfg.sequence:
        if cfg.sequence not in PRESET_SEQUENCES:
            raise ValueError(
                f"Unknown --sequence '{cfg.sequence}'. Choices: {sorted(PRESET_SEQUENCES)}"
            )
        raw_env = env.unwrapped
        sequence = PRESET_SEQUENCES[cfg.sequence]
        steps_per_segment = math.ceil(cfg.segment_s / raw_env.step_dt)
        _freeze_twist_resample(raw_env)
        _apply_command13(raw_env, sequence[0])  # avoid a one-step lag on the first obs
        policy = _ScriptedPolicy(_model, raw_env, sequence, steps_per_segment)
        print(
            f"[sim2sim] scripted sequence '{cfg.sequence}' "
            f"({len(sequence)} segments x {steps_per_segment} steps)"
        )

    if cfg.debug_steps > 0:
        _run_debug(env, policy, cfg.debug_steps)
        env.close()
        return

    # Viewer
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    import mjlab
    main()
