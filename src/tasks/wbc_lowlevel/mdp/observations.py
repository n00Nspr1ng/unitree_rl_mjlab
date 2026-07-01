from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def fixed_wb_command(
    env: ManagerBasedRlEnv,
    left_ee_pos: tuple[float, float, float] = (0.20, 0.20, 0.20),
    right_ee_pos: tuple[float, float, float] = (0.20, -0.20, 0.20),
    body_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    height: float = 0.75,
) -> torch.Tensor:
    """Fixed whole-body command matching the IsaacSim policy's obs slot.

    Returns a constant 10-dim tensor per env:
      [l_ee_x, l_ee_y, l_ee_z, r_ee_x, r_ee_y, r_ee_z, roll, pitch, yaw, height]

    EE positions are in the torso frame (matches WBCommandRangesCfg convention).
    Tune params to match the desired eval scenario; defaults are a neutral
    upright standing pose with arms slightly forward.
    """
    cmd = torch.zeros(env.num_envs, 10, device=env.device)
    cmd[:, 0:3] = torch.tensor(left_ee_pos, device=env.device)
    cmd[:, 3:6] = torch.tensor(right_ee_pos, device=env.device)
    cmd[:, 6:9] = torch.tensor(body_rpy, device=env.device)
    cmd[:, 9] = height
    return cmd
