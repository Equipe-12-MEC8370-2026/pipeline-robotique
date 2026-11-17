"""CTA-SO101-ProfileStack-Mimic-v0: the P3 scene wired for Isaac Lab Mimic (MimicGen-style demo amplification).

D12 trial (2026-11-15): replace the hand-coded expert by a few SEED demos amplified automatically. Mimic works in
end-effector (TCP) poses; the SO-101 is commanded in joint positions, so ``target_eef_pose_to_action`` runs our
TCP IK (cta_learn.sim.ik.TcpIK, the one the expert uses). Differences with the P3 scene, for generation only:
  - the robot starts at the expert's READY pose (gripper open): the folded start -> ready move is joint-space
    and must not be re-targeted in Cartesian space (the IK falls into the elbow-up branch, P0.3); it is added
    back when the generated demos are rendered with cameras (scripts/sim/render_mimic_demos.py);
  - no cameras (generation is pure physics, ~3x faster).
Subtasks = the TaskSpec's steps split at the grasps, each tied to the object it is relative to:
  grasp the short part | place it in its zone (world frame) | grasp the long part | place it on the short part.
Generation uses one env (the IK is batched but integrates its command for every env at each call).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from cta_learn.datagen.pick_place_expert import READY_POSE_DEG
from cta_learn.datagen.stack_expert import TILT_WEIGHT_HIGH, TILT_WEIGHT_LOW
from cta_learn.sim.ik import TcpIK
from cta_learn.sim.tasks.pick_place import ARM_JOINTS
from cta_learn.sim.tasks.profile_stack import TCP_OFFSET, ProfileStackEnvCfg

TASK_ID = "CTA-SO101-ProfileStack-Mimic-v0"
EEF = "so101"
GRIPPER_OPEN_DEG = 35.0
VERTICAL_BELOW_Z = 0.07   # m: below this TCP height the IK keeps the gripper vertical (as the expert near the table)
SUBTASK_SIGNALS = ("grasp_short", "place_short", "grasp_long")


@configclass
class ProfileStackMimicEnvCfg(ProfileStackEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # robot starts at the expert's READY pose, gripper open
        ready = dict(zip(ARM_JOINTS, (math.radians(v) for v in READY_POSE_DEG)))
        ready["gripper"] = math.radians(GRIPPER_OPEN_DEG)
        self.scene.robot = self.scene.robot.replace(init_state=self.scene.robot.init_state.replace(joint_pos=ready))
        # no cameras during generation
        self.scene.front = None
        self.scene.wrist = None
        self.observations.policy.front = None
        self.observations.policy.wrist = None
        self.events.front_camera = None

        d = self.datagen_config
        d.name = "cta_profile_stack_seeds"
        d.generation_guarantee = True        # keep trying until generation_num_trials successes
        d.generation_keep_failed = False
        d.generation_num_trials = 10
        d.generation_select_src_per_subtask = True
        d.generation_transform_first_robot_pose = False
        d.generation_interpolate_from_last_target_pose = True
        d.max_num_failures = 50
        d.seed = 1

        def sub(obj, signal, offset=(5, 10)):
            return SubTaskConfig(object_ref=obj, subtask_term_signal=signal, subtask_term_offset_range=offset,
                                 selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3},
                                 action_noise=0.0, num_interpolation_steps=10, num_fixed_steps=0,
                                 apply_noise_during_interpolation=False)

        self.subtask_configs[EEF] = [
            sub("profile_short", "grasp_short"),
            sub(None, "place_short"),            # the short part's zone is fixed in the robot frame
            sub("profile_long", "grasp_long"),
            sub("profile_short", None, (0, 0)),  # last: stand the long part on the short part
        ]


class ProfileStackMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic API for the SO-101 (single end effector "so101", joint-position actions through the TCP IK)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ik = TcpIK(self.scene["robot"], ARM_JOINTS, TCP_OFFSET, dt=self.step_dt)

    # ------------------------------------------------------------------ reset: restart the IK's integrated command
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_ik"):
            robot = self.scene["robot"]
            self._ik.set_command(env_ids, robot.data.joint_pos.torch[env_ids][:, self._ik.joint_ids].clone())

    # ------------------------------------------------------------------ poses (env frame, as get_object_poses)
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        tcp, quat = self._ik.tcp_pose_w()
        pos = tcp[env_ids] - self.scene.env_origins[env_ids]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat[env_ids]))

    def closing_yaw(self, rot: torch.Tensor) -> torch.Tensor:
        """World yaw of the gripper closing axis for gripper-body rotation matrices (..., 3, 3)."""
        x = rot @ self._ik._R_local[:, 0]
        return torch.atan2(x[..., 1], x[..., 0])

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                                  action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor:
        (pose,) = target_eef_pose_dict.values()
        (grip,) = gripper_action_dict.values()
        pos, rot = PoseUtils.unmake_pose(pose)
        pos = pos.clone()
        if action_noise_dict is not None:
            noise = float(action_noise_dict.get(EEF, 0.0) or 0.0)
            if noise > 0:
                pos = pos + noise * torch.randn_like(pos)
        n = self.num_envs
        goal = self._ik.tcp_pose_w()[0].clone()          # other envs: hold (single-env generation)
        goal[env_id] = pos + self.scene.env_origins[env_id]
        yaw = torch.zeros(n, device=self.device)
        yaw[env_id] = self.closing_yaw(rot)
        tilt = torch.full((n,), TILT_WEIGHT_LOW, device=self.device)
        tilt[env_id] = TILT_WEIGHT_HIGH if float(pos[2]) < VERTICAL_BELOW_Z else TILT_WEIGHT_LOW
        arm_q = self._ik.compute(goal, yaw, tilt)[env_id]
        return torch.cat([arm_q, grip.reshape(1).to(arm_q.dtype)], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError("joint-position actions: seeds record their target TCP poses directly")

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        poses = super().get_object_poses(env_ids)
        return {k: v for k, v in poses.items() if k in ("profile_short", "profile_long")}
