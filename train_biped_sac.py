import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import sys

XML_PATH = "xml_files/biped_3d_feet.xml"

VX_RANGE = (-0.4,  0.4)
VY_RANGE = (-0.2,  0.2)

MAX_STEPS     = 10000
MIN_HEIGHT    = 0.7
TARGET_HEIGHT = 1.35

CTRL_R_HIP_Y = 0
CTRL_R_HIP   = 1
CTRL_R_KNEE  = 2
CTRL_R_FOOT  = 3
CTRL_L_HIP_Y = 4
CTRL_L_HIP   = 5
CTRL_L_KNEE  = 6
CTRL_L_FOOT  = 7

JOINT_NAMES = [
    "left_hip_y_j", "left_hip", "left_knee",
    "right_hip_y_j", "right_hip", "right_knee",
]


class BipedFeetEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.lfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        self.rfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        self.ground_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        self.cmd_vx       = 0.0
        self.cmd_vy       = 0.0
        self.step_count   = 0
        self.prev_action  = np.zeros(6)
        self.prev_contact = (1.0, 1.0)
        self._fell        = False

    def _get_foot_contacts(self):
        left_contact  = False
        right_contact = False
        for i in range(self.data.ncon):
            c     = self.data.contact[i]
            geoms = {c.geom1, c.geom2}
            if self.ground_id in geoms:
                if self.lfoot_id in geoms:
                    left_contact = True
                if self.rfoot_id in geoms:
                    right_contact = True
        return float(left_contact), float(right_contact)

    def _get_joint_angles(self):
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])

    def _get_joint_vels(self):
        return np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]]
            for jid in self.joint_ids
        ])

    def _get_obs(self):
        q  = self._get_joint_angles()
        dq = self._get_joint_vels()
        torso_quat = self.data.xquat[self.torso_id].copy()
        v6d = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_id, v6d, 0)
        ang_vel    = v6d[:3]
        linear_vel = v6d[3:]
        lc, rc = self._get_foot_contacts()
        obs = np.concatenate([
            q, dq, torso_quat, ang_vel, linear_vel,
            [self.cmd_vx, self.cmd_vy],
            self.prev_action,
            [lc, rc],
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _get_reward(self, action):
        v6d = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_id, v6d, 0)
        vx = v6d[3]
        vy = v6d[4]
        torso_z = self.data.xpos[self.torso_id][2]
        lc, rc  = self._get_foot_contacts()

        # velocity tracking — dominant signal
        vx_err     = (vx - self.cmd_vx) ** 2
        vy_err     = (vy - self.cmd_vy) ** 2
        vel_reward = np.exp(-3.0 * (vx_err + vy_err))

        # penalise standing still when commanded to move
        cmd_speed    = np.sqrt(self.cmd_vx**2 + self.cmd_vy**2)
        actual_speed = np.sqrt(vx**2 + vy**2)
        stationary_penalty = -1.0 if (cmd_speed > 0.1 and actual_speed < 0.05) else 0.0

        # alive
        alive = 1.0 if torso_z > MIN_HEIGHT else 0.0

        # height reward
        height_reward = np.exp(-8.0 * (torso_z - TARGET_HEIGHT) ** 2)

        # upright reward
        q     = self.data.xquat[self.torso_id]
        roll  = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),
                           1 - 2*(q[1]**2 + q[2]**2))
        pitch = np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
        upright = np.exp(-3.0 * (roll**2 + pitch**2))

        # penalise wide hip abduction
        q_angles       = self._get_joint_angles()
        spread_penalty = -0.5 * (q_angles[0]**2 + q_angles[3]**2)

        # gait rewards
        both_in_air    = (lc == 0.0 and rc == 0.0)
        gait_penalty   = -1.0 if both_in_air else 0.0
        prev_lc, prev_rc = self.prev_contact
        alternating    = 0.2 if (lc != prev_lc or rc != prev_rc) else 0.0
        contact_reward = 0.3 if (lc + rc) >= 1.0 else 0.0

        # smoothness and energy
        smoothness = -0.02 * np.sum((action - self.prev_action) ** 2)
        energy     = -0.008 * np.sum(action ** 2)

        self._fell        = torso_z < MIN_HEIGHT
        self.prev_contact = (lc, rc)

        reward = (
            vel_reward      * 3.0
          + stationary_penalty
          + alive           * 1.0
          + height_reward   * 0.5
          + upright         * 0.5
          + spread_penalty
          + contact_reward
          + gait_penalty
          + alternating
          + smoothness
          + energy
        )
        return float(np.clip(reward, -20.0, 20.0))

    def _is_done(self):
        torso_z = self.data.xpos[self.torso_id][2]
        fell    = torso_z < MIN_HEIGHT
        timeout = self.step_count >= MAX_STEPS
        if timeout and not fell:
            self._fell = False
        return fell or timeout

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, self.model.nq - 7)
        self.data.qvel[:]  += np.random.uniform(-0.02, 0.02, self.model.nv)

        for name, deg in [("right_knee", -15), ("left_knee", -15)]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = np.deg2rad(deg)

        mujoco.mj_forward(self.model, self.data)

        self.cmd_vx       = float(np.random.uniform(*VX_RANGE))
        self.cmd_vy       = float(np.random.uniform(*VY_RANGE))
        self.step_count   = 0
        self.prev_action  = np.zeros(6)
        self.prev_contact = (1.0, 1.0)
        self._fell        = False

        return self._get_obs(), {}

    def step(self, action):
        scaled = action * 2.0

        self.data.ctrl[CTRL_R_HIP_Y] = scaled[3]
        self.data.ctrl[CTRL_R_HIP]   = scaled[4]
        self.data.ctrl[CTRL_R_KNEE]  = scaled[5]
        self.data.ctrl[CTRL_R_FOOT]  = 0.0
        self.data.ctrl[CTRL_L_HIP_Y] = scaled[0]
        self.data.ctrl[CTRL_L_HIP]   = scaled[1]
        self.data.ctrl[CTRL_L_KNEE]  = scaled[2]
        self.data.ctrl[CTRL_L_FOOT]  = 0.0

        # step physics with instability guard
        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            pass

        torso_pos = self.data.xpos[self.torso_id]
        if not np.all(np.isfinite(torso_pos)) or np.any(np.abs(torso_pos) > 100):
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self._fell = True

        reward           = self._get_reward(action)
        self.prev_action = action.copy()
        self.step_count += 1

        done = self._is_done()
        if done and self._fell:
            reward -= 10.0

        obs  = self._get_obs()
        info = {"cmd_vx": self.cmd_vx, "cmd_vy": self.cmd_vy}

        return obs, reward, done, False, info


class PrintCallback(BaseCallback):
    def __init__(self, print_every=10_000, verbose=0):
        super().__init__(verbose)
        self._print_every = print_every

    def _on_step(self):
        if self.num_timesteps % self._print_every == 0:
            print(f"  steps: {self.num_timesteps:,}")
        return True


if __name__ == "__main__":
    os.makedirs("checkpoints_sac", exist_ok=True)

    print("creating environment...")
    env = BipedFeetEnv()

    # find the best available saved model
    # priority: biped_sac_walker.zip > latest checkpoint > fresh start
    import glob

    prev_model  = "checkpoints_sac/run_0329_1835/biped_sac_1600000_steps.zip"
    checkpoints = sorted(glob.glob("checkpoints_sac/**/biped_sac_*_steps.zip", recursive=True))
    latest_ckpt = checkpoints[-1] if checkpoints else None

    if os.path.exists(prev_model):
        load_path   = prev_model
        fresh_start = False
        print(f"loading {prev_model}...")
    elif latest_ckpt:
        load_path   = latest_ckpt
        fresh_start = False
        print(f"loading latest checkpoint: {latest_ckpt}...")
    else:
        load_path   = None
        fresh_start = True
        print("no previous model found — starting fresh")

    if fresh_start:
        model = SAC(
            policy             = "MlpPolicy",
            env                = env,
            verbose            = 1,
            learning_rate      = 3e-4,
            buffer_size        = 1_000_000,
            learning_starts    = 10_000,
            batch_size         = 256,
            tau                = 0.005,
            gamma              = 0.99,
            train_freq         = 1,
            gradient_steps     = 1,
            ent_coef           = "auto",
            policy_kwargs      = dict(net_arch=[256, 256]),
        )
    else:
        model = SAC.load(load_path, env=env)
        # when loading a checkpoint the step counter resets to 0
        # set learning_starts to 0 so it does not repeat random exploration
        model.learning_starts = 0

    from datetime import datetime
    run_id = datetime.now().strftime("%m%d_%H%M")
    ckpt_dir = f"checkpoints_sac/run_{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"checkpoints saving to: {ckpt_dir}")

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = ckpt_dir,
        name_prefix = "biped_sac"
    )
    print_cb = PrintCallback(print_every=10_000)

    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 999_999_999

    print(f"training SAC — runs until Ctrl+C")
    print(f"first {10_000:,} steps are random exploration (normal)")
    print(f"checkpoints saved every 50k steps to checkpoints_sac/")
    print(f"Ctrl+C to stop — saves to biped_sac_walker.zip")

    try:
        model.learn(
            total_timesteps      = total_steps,
            callback             = [checkpoint_cb, print_cb],
            reset_num_timesteps  = fresh_start,
        )
    except KeyboardInterrupt:
        print("\ntraining stopped")

    model.save("biped_sac_walker")
    print("model saved to biped_sac_walker.zip")