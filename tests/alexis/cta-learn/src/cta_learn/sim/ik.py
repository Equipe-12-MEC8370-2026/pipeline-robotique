"""Damped-least-squares IK for the SO-101 TCP (simulator side, batched torch).

Task: put the TCP at a Cartesian goal with the gripper pointing straight down and its
closing axis (gripper local +X) at a given world yaw. The SO-101 arm has 5 joints, so
position (3) + orientation (3) is solved in the least-squares sense, position weighted
higher. Output = absolute joint targets (rad) for the 5 arm joints, rate-limited.
"""

from __future__ import annotations

import torch
from isaaclab.utils import math as mu


class TcpIK:
    def __init__(
        self,
        robot,
        arm_joint_names: list[str],
        tcp_offset: tuple[float, float, float],
        ee_body: str = "gripper",
        damping: float = 0.05,
        pos_gain: float = 1.0,
        rot_gain: float = 0.5,
        rot_weight: float = 0.3,
        max_joint_speed: float = 0.5,
        dt: float = 1 / 30,
    ):
        self.robot = robot
        self.device = robot.device
        self.joint_ids = [list(robot.data.joint_names).index(n) for n in arm_joint_names]
        body_idx = list(robot.data.body_names).index(ee_body)
        self.body_idx = body_idx
        self.jac_idx = body_idx - 1 if robot.is_fixed_base else body_idx
        self.tcp_off = torch.tensor(tcp_offset, device=self.device)
        self.u_local = self.tcp_off / torch.linalg.norm(self.tcp_off)  # approach axis in gripper frame
        self.damping, self.pos_gain, self.rot_gain = damping, pos_gain, rot_gain
        self.w = torch.tensor([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight], device=self.device)
        self.max_step = max_joint_speed * dt
        lim = robot.data.joint_pos_limits.torch[0, self.joint_ids]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self._R_local = self._local_basis()
        # Integrated joint command. Adding the IK step to the MEASURED angles does not work on the SO-101:
        # its soft position servos (stiffness 17.8) sag ~1 deg under gravity, which cancels each small
        # step and freezes the arm (found 2026-11-11). We integrate on the previous command instead and
        # keep it within max_lead of the measurement to avoid wind-up against contacts.
        self.q_cmd = robot.data.joint_pos.torch[:, self.joint_ids].clone()
        self.max_lead = 0.2  # rad

    def set_command(self, env_ids, q: torch.Tensor) -> None:
        """Set the integrated command (episode start, or when a joint-space phase drives the arm)."""
        self.q_cmd[env_ids] = q

    def _local_basis(self) -> torch.Tensor:
        """Columns: local closing axis x', approach axis u, and their cross product (right-handed)."""
        u = self.u_local
        x = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        x = x - torch.dot(x, u) * u
        x = x / torch.linalg.norm(x)
        y = torch.linalg.cross(u, x)
        return torch.stack([x, y, u], dim=1)

    def desired_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """World orientation of the gripper body for 'approach axis down, closing axis at yaw'."""
        n = yaw.shape[0]
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3)
        xw = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
        yw = torch.linalg.cross(down, xw, dim=1)
        W = torch.stack([xw, yw, down], dim=2)  # world images of the local basis columns
        R = W @ self._R_local.T.expand(n, 3, 3)
        return mu.quat_from_matrix(R)

    def tcp_pose_w(self):
        pos = self.robot.data.body_link_pos_w.torch[:, self.body_idx]
        quat = self.robot.data.body_link_quat_w.torch[:, self.body_idx]
        return pos + mu.quat_apply(quat, self.tcp_off.expand(pos.shape[0], 3)), quat

    def compute(self, tcp_goal_w: torch.Tensor, yaw: torch.Tensor, tilt_weight: torch.Tensor | None = None) -> torch.Tensor:
        """One DLS step. ``tilt_weight`` (per env) scales how strongly the approach axis must point down;
        the yaw about the vertical keeps weight ``rot_weight``. Position always has weight 1.

        A strictly vertical gripper is not reachable everywhere (the wrist hits its 95 deg limit when
        the TCP is high), so the expert asks for verticality only close to the table.
        """
        n = tcp_goal_w.shape[0]
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        tcp, quat = self.tcp_pose_w()
        J = self.robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, :][:, :, self.joint_ids].clone()
        r = mu.quat_apply(quat, self.tcp_off.expand(n, 3))  # gripper origin -> TCP, world
        J[:, 0:3, :] = J[:, 0:3, :] - torch.bmm(mu.skew_symmetric_matrix(r), J[:, 3:6, :])
        pos_err = tcp_goal_w - tcp

        R = mu.matrix_from_quat(quat)
        a = R @ self.u_local                       # approach axis, world
        xw = R @ self._R_local[:, 0]               # closing axis, world
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3)
        tilt_err = torch.linalg.cross(a, -z, dim=1)           # rotate approach axis toward straight down
        x_h = xw.clone()
        x_h[:, 2] = 0.0
        x_h = x_h / torch.linalg.norm(x_h, dim=1, keepdim=True).clamp_min(1e-6)
        x_d = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
        yaw_angle = torch.atan2(torch.linalg.cross(x_h, x_d, dim=1)[:, 2], (x_h * x_d).sum(1))
        rot_err = tilt_err + yaw_angle.unsqueeze(1) * z
        # diagnostics (recorded with the episodes)
        self.last_pos_err = torch.linalg.norm(pos_err, dim=1)
        self.last_yaw_err = yaw_angle
        self.last_tilt = torch.acos((-a[:, 2]).clamp(-1, 1))

        tw = self.w[3] if tilt_weight is None else tilt_weight.view(n, 1, 1)
        zz = z.unsqueeze(2) * z.unsqueeze(1)
        eye = torch.eye(3, device=self.device).expand(n, 3, 3)
        Wr = tw * (eye - zz) + self.w[3] * zz      # weight horizontal (tilt) vs vertical (yaw) rotation
        Jw = torch.cat([J[:, 0:3, :], Wr @ J[:, 3:6, :]], dim=1)
        err = torch.cat([self.pos_gain * pos_err, self.rot_gain * (Wr @ rot_err.unsqueeze(2)).squeeze(2)], dim=1)
        lam = (self.damping**2) * torch.eye(6, device=self.device)
        dq = Jw.transpose(1, 2) @ torch.linalg.solve(Jw @ Jw.transpose(1, 2) + lam, err.unsqueeze(-1))
        dq = dq.squeeze(-1)
        # Scale the whole step uniformly: clamping joints one by one would change its DIRECTION
        # (found 2026-11-11: per-joint clamping drove the TCP away from the goal).
        scale = (self.max_step / dq.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        dq = dq * scale
        q_new = torch.max(torch.min(self.q_cmd + dq, self.q_hi), self.q_lo)
        q_new = torch.max(torch.min(q_new, q + self.max_lead), q - self.max_lead)
        self.q_cmd = q_new
        return q_new.clone()
