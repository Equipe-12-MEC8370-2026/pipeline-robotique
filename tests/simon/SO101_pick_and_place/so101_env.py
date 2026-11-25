import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(
        self,
        xml_path="SO-ARM100/Simulation/SO101/pick_place_scene.xml",
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # 20 sous-pas physiques de 0.002s = 0.04s par action (25 Hz)
        self.n_substeps = 20
        self.max_steps = 200
        self.current_step = 0

        # Récupération automatique des limites des 6 servomoteurs
        self.n_actuators = self.model.nu
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Repères (sites et joints) dans la scène XML
        self.gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self.cube_qpos_addr = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_qvel_addr = self.model.jnt_dofadr[self.cube_joint_id]

        # Espace d'action : variations incrémentales normalisées [-1, 1] sur les 6 axes
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32
        )

        # Espace d'observation : 27 valeurs continues
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

    def _get_gripper_pos(self):
        if self.gripper_site_id != -1:
            return self.data.site_xpos[self.gripper_site_id].copy()
        # Repli sur le dernier segment du bras si le site porte un autre nom
        return self.data.xpos[self.model.nbody - 2].copy()

    def _get_cube_pos(self):
        return self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3].copy()

    def _get_target_pos(self):
        return self.model.site_pos[self.target_site_id].copy()

    def _get_obs(self):
        arm_qpos = self.data.qpos[: self.n_actuators].copy()
        arm_qvel = self.data.qvel[: self.n_actuators].copy()
        gripper_pos = self._get_gripper_pos()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()

        gripper_to_cube = cube_pos - gripper_pos
        cube_to_target = target_pos - cube_pos

        return np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                gripper_pos,
                cube_pos,
                target_pos,
                gripper_to_cube,
                cube_to_target,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        # Position aléatoire du cube dans la zone atteignable devant le bras
        cube_x = self.np_random.uniform(0.16, 0.25)
        cube_y = self.np_random.uniform(-0.08, 0.08)
        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = [
            cube_x,
            cube_y,
            0.015,
        ]
        self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qvel[self.cube_qvel_addr : self.cube_qvel_addr + 6] = 0.0

        # Position aléatoire de la cible verte
        target_x = self.np_random.uniform(0.16, 0.25)
        target_y = self.np_random.uniform(0.10, 0.18) * self.np_random.choice([-1, 1])
        self.model.site_pos[self.target_site_id] = [target_x, target_y, 0.015]

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # Contrôle fluide par petits incréments d'angles (évite les mouvements brusques)
        delta = np.clip(action, -1.0, 1.0) * 0.05
        self.data.ctrl[:] = np.clip(
            self.data.ctrl + delta, self.ctrl_low, self.ctrl_high
        )

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        gripper_pos = self._get_gripper_pos()
        cube_pos = self._get_cube_pos()
        target_pos = self._get_target_pos()

        dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
        dist_cube_target = np.linalg.norm(cube_pos - target_pos)

        # --- Fonction de récompense multi-étapes (Reward Shaping) ---
        # 1. Approche de la pince vers le cube
        reward_reach = -dist_gripper_cube + (1.0 - np.tanh(10.0 * dist_gripper_cube))

        # 2. Levage du cube au-dessus de la table (hauteur initiale = 0.015 m)
        lift_height = np.clip(cube_pos[2] - 0.015, 0.0, 0.15)
        reward_lift = 15.0 * lift_height if dist_gripper_cube < 0.05 else 0.0

        # 3. Transport du cube vers la cible verte
        reward_transport = (
            2.0 * (1.0 - np.tanh(5.0 * dist_cube_target))
            if cube_pos[2] > 0.022
            else 0.0
        )

        # 4. Bonus de maintien dans la zone cible (rayon de 3 cm)
        is_success = dist_cube_target < 0.03
        reward_success = 5.0 if is_success else 0.0

        # 5. Légère pénalité sur les commandes excessives
        reward_ctrl = -0.01 * np.sum(np.square(action))

        reward = float(
            reward_reach
            + reward_lift
            + reward_transport
            + reward_success
            + reward_ctrl
        )

        terminated = False
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"is_success": is_success}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.model.opt.timestep * self.n_substeps)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None