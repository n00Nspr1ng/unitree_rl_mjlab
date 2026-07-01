"""Sim2sim evaluation script for TorchScript policies exported from IsaacSim.

Usage:
    python scripts/sim2sim.py Unitree-G1-WbcLowlevel \\
        --checkpoint /path/to/policy_v7.pt \\
        [--num-envs 1] [--device cuda:0] [--viewer native|viser]

The checkpoint must be a TorchScript module (torch.jit.load) that maps
obs (B, obs_dim) -> action (B, action_dim).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


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
