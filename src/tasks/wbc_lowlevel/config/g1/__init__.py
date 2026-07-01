from mjlab.tasks.registry import register_mjlab_task
from src.tasks.wbc_lowlevel.rl import WbcLowlevelOnPolicyRunner

from .env_cfgs import unitree_g1_wbc_lowlevel_env_cfg
from .rl_cfg import unitree_g1_wbc_lowlevel_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-WbcLowlevel",
    env_cfg=unitree_g1_wbc_lowlevel_env_cfg(),
    play_env_cfg=unitree_g1_wbc_lowlevel_env_cfg(play=True),
    rl_cfg=unitree_g1_wbc_lowlevel_ppo_runner_cfg(),
    runner_cls=WbcLowlevelOnPolicyRunner,
)
