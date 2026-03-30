import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import sys

XML_PATH = "xml_files/biped_3d_feet.xml"

VX_RANGE = (-0.4,  0.4)
VY_RANGE = (-0.2,  0.2)

MAX_STEPS     = 10000   # 10 seconds of sim time
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
        # clip to prevent NaN from extreme values destabilising training
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
        # if cmd speed > 0.1 m/s but actual speed is near zero, punish
        cmd_speed    = np.sqrt(self.cmd_vx**2 + self.cmd_vy**2)
        actual_speed = np.sqrt(vx**2 + vy**2)
        if cmd_speed > 0.1 and actual_speed < 0.05:
            stationary_penalty = -1.0
        else:
            stationary_penalty = 0.0

        # alive — reduced weight so it does not dominate
        alive = 1.0 if torso_z > MIN_HEIGHT else 0.0

        # height reward
        height_reward = np.exp(-8.0 * (torso_z - TARGET_HEIGHT) ** 2)

        # upright reward
        q     = self.data.xquat[self.torso_id]
        roll  = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),
                           1 - 2*(q[1]**2 + q[2]**2))
        pitch = np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
        upright = np.exp(-3.0 * (roll**2 + pitch**2))

        # penalise wide hip abduction — stops the legs-wide trick
        q_angles    = self._get_joint_angles()
        lhy = q_angles[0]   # left_hip_y_j
        rhy = q_angles[3]   # right_hip_y_j
        spread_penalty = -0.5 * (lhy**2 + rhy**2)

        # gait rewards
        both_in_air  = (lc == 0.0 and rc == 0.0)
        gait_penalty = -1.0 if both_in_air else 0.0

        prev_lc, prev_rc = self.prev_contact
        alternating = 0.2 if (lc != prev_lc or rc != prev_rc) else 0.0
        contact_reward = 0.3 if (lc + rc) >= 1.0 else 0.0

        # smoothness and energy
        smoothness = -0.02 * np.sum((action - self.prev_action) ** 2)
        energy     = -0.008 * np.sum(action ** 2)

        self._fell        = torso_z < MIN_HEIGHT
        self.prev_contact = (lc, rc)

        reward = (
            vel_reward      * 3.0    # dominant signal
          + stationary_penalty       # punish standing still
          + alive           * 1.0    # reduced from 2.0
          + height_reward   * 0.5
          + upright         * 0.5
          + spread_penalty           # anti wide-stance hack
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

        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05,
                                                 self.model.nq - 7)
        self.data.qvel[:]  += np.random.uniform(-0.02, 0.02,
                                                 self.model.nv)

        for name, deg in [("right_knee", -15), ("left_knee", -15)]:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
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
        scaled = action * 2.0   # reduced from 5.0 — prevents physics explosion

        self.data.ctrl[CTRL_R_HIP_Y] = scaled[3]
        self.data.ctrl[CTRL_R_HIP]   = scaled[4]
        self.data.ctrl[CTRL_R_KNEE]  = scaled[5]
        self.data.ctrl[CTRL_R_FOOT]  = 0.0
        self.data.ctrl[CTRL_L_HIP_Y] = scaled[0]
        self.data.ctrl[CTRL_L_HIP]   = scaled[1]
        self.data.ctrl[CTRL_L_KNEE]  = scaled[2]
        self.data.ctrl[CTRL_L_FOOT]  = 0.0

        try:
            mujoco.mj_step(self.model, self.data)
        except Exception:
            pass

        # detect physics instability — reset if torso position is NaN or huge
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
    def __init__(self, print_every=100_000, verbose=0):
        super().__init__(verbose)
        self._print_every = print_every

    def _on_step(self):
        if self.num_timesteps % self._print_every == 0:
            print(f"  steps: {self.num_timesteps:,}")
        return True


if __name__ == "__main__":
    os.makedirs("checkpoints_v3", exist_ok=True)

    # check if a previous model exists to continue from
    prev_model = "biped_feet_walker_v3.zip"
    fresh_start = not os.path.exists(prev_model)

    print("creating environment...")
    env = make_vec_env(BipedFeetEnv, n_envs=1)

    if fresh_start:
        print("no previous model found — starting fresh")
        model = PPO(
            policy        = "MlpPolicy",
            env           = env,
            verbose       = 1,
            n_steps       = 2048,
            batch_size    = 64,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            learning_rate = 3e-4,
            ent_coef      = 0.01,
            policy_kwargs = dict(net_arch=[256, 256]),
        )
        reset_steps = True
    else:
        print(f"loading {prev_model} and continuing...")
        model = PPO.load(prev_model, env=env)
        reset_steps = False

    checkpoint_cb = CheckpointCallback(
        save_freq   = 500_000,
        save_path   = "checkpoints_v3/",
        name_prefix = "biped_feet_v3"
    )
    print_cb = PrintCallback()

    # default 10M — pass a number as argument to override
    # e.g. python train_biped_feet_v3.py 5000000
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 999_999_999

    print(f"training for {total_steps:,} steps")
    print("checkpoints saved every 500k to checkpoints_v3/")
    print("Ctrl+C to stop — saves automatically")

    try:
        model.learn(
            total_timesteps     = total_steps,
            callback            = [checkpoint_cb, print_cb],
            reset_num_timesteps = reset_steps,
        )
    except KeyboardInterrupt:
        print("\ntraining stopped early")

    model.save("biped_feet_walker_v3")
    print("model saved to biped_feet_walker_v3.zip")