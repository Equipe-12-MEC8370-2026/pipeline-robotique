"""Registers cta_learn gym environments. Import only after the simulator app is running."""

import gymnasium as gym

from cta_learn.sim.tasks.pick_place import TASK_ID as PICK_PLACE_ID

gym.register(
    id=PICK_PLACE_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "cta_learn.sim.tasks.pick_place:PickPlaceEnvCfg"},
    disable_env_checker=True,
)

from cta_learn.sim.tasks.profile_stack import TASK_ID as PROFILE_STACK_ID  # noqa: E402

gym.register(
    id=PROFILE_STACK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "cta_learn.sim.tasks.profile_stack:ProfileStackEnvCfg"},
    disable_env_checker=True,
)

from cta_learn.sim.tasks.profile_stack_mimic import TASK_ID as PROFILE_STACK_MIMIC_ID  # noqa: E402

gym.register(
    id=PROFILE_STACK_MIMIC_ID,
    entry_point="cta_learn.sim.tasks.profile_stack_mimic:ProfileStackMimicEnv",
    kwargs={"env_cfg_entry_point": "cta_learn.sim.tasks.profile_stack_mimic:ProfileStackMimicEnvCfg"},
    disable_env_checker=True,
)

from cta_learn.sim.tasks.taskspec_task import TASK_ID as TASKSPEC_ID  # noqa: E402

gym.register(
    id=TASKSPEC_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "cta_learn.sim.tasks.taskspec_task:TaskSpecEnvCfg"},
    disable_env_checker=True,
)

from cta_learn.sim.tasks.pick_place_real import TASK_ID as PICK_PLACE_REAL_ID  # noqa: E402

gym.register(
    id=PICK_PLACE_REAL_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "cta_learn.sim.tasks.pick_place_real:PickPlaceRealEnvCfg"},
    disable_env_checker=True,
)
