import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time
import threading

XML_PATH = "xml_files/biped_3d.xml"

# velocity commands the robot is asked to track
# [vx, vy] in m/s — sampled randomly each episode
VX_RANGE = (-0.5, 0.5)
VY_RANGE = (-0.3, 0.3)

# episode length in simulation steps
MAX_STEPS = 1000

# minimum torso height before episode ends (robot fell)
MIN_HEIGHT = 0.6

# target torso height reward band
TARGET_HEIGHT = 1.35


class BipedEnv(gym.Env):
    """
    Observation space (30 values):
      - 6 joint angles
      - 6 joint velocities
      - torso quaternion (4)
      - torso angular velocity (3)
      - torso linear velocity (3)
      - commanded vx, vy (2)
      - previous action (6)

    Action space (6 values):
      - velocity command for each joint, clipped to [-5, 5]

    Reward:
      - velocity tracking: how close actual velocity is to commanded
      - alive bonus: reward for staying upright
      - height penalty: penalise being too far from target height
      - action smoothness: penalise large jerky actions
      - torso upright: penalise tilt
    """

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (30,),
            dtype = np.float32
        )

        self.action_space = spaces.Box(
            low  = -1.0,
            high =  1.0,
            shape = (6,),
            dtype = np.float32
        )

        self.cmd_vx       = 0.0
        self.cmd_vy       = 0.0
        self.step_count   = 0
        self.prev_action  = np.zeros(6)

        # joint index mapping for the 3d feet model
        # order matches biped_3d_feet joint layout
        self.joint_names = [
            "left_hip_y_j", "left_hip", "left_knee",
            "right_hip_y_j", "right_hip", "right_knee"
        ]
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.joint_names
        ]
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    def _get_joint_angles(self):
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self.joint_ids
        ])

    def _get_joint_velocities(self):
        return np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]]
            for jid in self.joint_ids
        ])

    def _get_obs(self):
        q   = self._get_joint_angles()
        dq  = self._get_joint_velocities()

        # torso quaternion and velocities
        torso_quat  = self.data.xquat[self.torso_id].copy()
        v6d = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_id, v6d, 0)
        ang_vel    = v6d[:3]
        linear_vel = v6d[3:]

        obs = np.concatenate([
            q,                              # 6
            dq,                             # 6
            torso_quat,                     # 4
            ang_vel,                        # 3
            linear_vel,                     # 3
            np.array([self.cmd_vx, self.cmd_vy]),  # 2
            self.prev_action,               # 6
        ]).astype(np.float32)

        return obs

    def _get_reward(self, action):
        # torso linear velocity
        v6d = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_id, v6d, 0)
        vx = v6d[3]
        vy = v6d[4]

        # velocity tracking: gaussian reward, peaks when velocity matches command
        vx_error      = (vx - self.cmd_vx) ** 2
        vy_error      = (vy - self.cmd_vy) ** 2
        vel_reward    = np.exp(-2.0 * (vx_error + vy_error))

        # alive bonus for staying upright
        torso_z    = self.data.xpos[self.torso_id][2]
        alive      = 1.0 if torso_z > MIN_HEIGHT else 0.0

        # height reward: gaussian around target height
        height_reward = np.exp(-10.0 * (torso_z - TARGET_HEIGHT) ** 2)

        # torso upright: penalise roll and pitch
        # extract from quaternion: w,x,y,z
        q         = self.data.xquat[self.torso_id]
        roll      = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
        pitch     = np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
        upright   = np.exp(-2.0 * (roll**2 + pitch**2))

        # action smoothness: penalise large changes between steps
        smoothness = -0.01 * np.sum((action - self.prev_action) ** 2)

        # action magnitude: penalise large commands (energy efficiency)
        energy     = -0.005 * np.sum(action ** 2)

        reward = (
            vel_reward    * 2.0
          + alive         * 1.0
          + height_reward * 0.5
          + upright       * 0.5
          + smoothness
          + energy
        )

        return float(reward)

    def _is_done(self):
        torso_z = self.data.xpos[self.torso_id][2]
        return torso_z < MIN_HEIGHT or self.step_count >= MAX_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # slight random perturbation so policy learns to recover
        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, self.model.nq - 7)
        self.data.qvel[:]  += np.random.uniform(-0.02, 0.02, self.model.nv)

        # set initial knee bend so it doesn't start collapsed
        for name, angle in [("right_knee", -15), ("left_knee", -15),
                             ("right_hip_y_j", 0), ("left_hip_y_j", 0)]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = np.deg2rad(angle)

        mujoco.mj_forward(self.model, self.data)

        # sample a new random velocity command each episode
        self.cmd_vx = float(np.random.uniform(*VX_RANGE))
        self.cmd_vy = float(np.random.uniform(*VY_RANGE))

        self.step_count  = 0
        self.prev_action = np.zeros(6)

        return self._get_obs(), {}

    def step(self, action):
        # scale action from [-1,1] to joint velocity range [-5,5]
        scaled = action * 5.0

        # biped_3d.xml actuator order:
        # 0: right_hip_y, 1: right_hip, 2: right_knee
        # 3: left_hip_y,  4: left_hip,  5: left_knee
        self.data.ctrl[0] = scaled[3]   # right_hip_y
        self.data.ctrl[1] = scaled[4]   # right_hip
        self.data.ctrl[2] = scaled[5]   # right_knee
        self.data.ctrl[3] = scaled[0]   # left_hip_y
        self.data.ctrl[4] = scaled[1]   # left_hip
        self.data.ctrl[5] = scaled[2]   # left_knee

        mujoco.mj_step(self.model, self.data)

        reward = self._get_reward(action)

        self.prev_action  = action.copy()
        self.step_count  += 1

        done    = self._is_done()
        obs     = self._get_obs()
        info    = {"cmd_vx": self.cmd_vx, "cmd_vy": self.cmd_vy}

        return obs, reward, done, False, info


class ViewerCallback(BaseCallback):
    """
    live viewer using a shared numpy buffer + lock.
    training thread writes qpos/qvel into the buffer.
    viewer thread reads from the buffer and renders.
    they never touch the same mujoco data object simultaneously.
    """
    def __init__(self, render_every=16, verbose=0):
        super().__init__(verbose)
        self.render_every  = render_every
        self.viewer        = None
        self.render_model  = None
        self.render_data   = None
        self.thread        = None
        self.running       = False
        self._step_counter = 0
        self._lock         = threading.Lock()
        self._qpos_buf     = None
        self._qvel_buf     = None
        self._new_data     = False

    def _init_callback(self):
        self.render_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.render_data  = mujoco.MjData(self.render_model)
        nq = self.render_model.nq
        nv = self.render_model.nv
        self._qpos_buf = np.zeros(nq)
        self._qvel_buf = np.zeros(nv)
        self.running   = True
        self.thread    = threading.Thread(target=self._viewer_thread, daemon=True)
        self.thread.start()

    def _viewer_thread(self):
        with mujoco.viewer.launch_passive(self.render_model, self.render_data) as v:
            self.viewer = v
            while self.running and v.is_running():
                # read from shared buffer — safe because we only read numpy arrays
                with self._lock:
                    if self._new_data:
                        np.copyto(self.render_data.qpos, self._qpos_buf)
                        np.copyto(self.render_data.qvel, self._qvel_buf)
                        self._new_data = False
                mujoco.mj_forward(self.render_model, self.render_data)
                v.sync()
                time.sleep(1.0 / 30.0)

    def _on_step(self):
        self._step_counter += 1
        if self._step_counter % self.render_every != 0:
            return True
        if self._qpos_buf is None:
            return True

        # write current state into shared buffer — safe numpy copy
        try:
            train_env = self.training_env.envs[0].unwrapped
            with self._lock:
                np.copyto(self._qpos_buf, train_env.data.qpos)
                np.copyto(self._qvel_buf, train_env.data.qvel)
                self._new_data = True
        except Exception:
            pass

        return True

    def _on_training_end(self):
        self.running = False


if __name__ == "__main__":

    print("creating environment...")
    env = make_vec_env(BipedEnv, n_envs=1)

    print("setting up PPO...")
    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        verbose        = 1,
        n_steps        = 2048,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        learning_rate  = 3e-4,
        ent_coef       = 0.01,      # encourages exploration
        policy_kwargs  = dict(net_arch=[256, 256]),  # 2 hidden layers of 256
    )

    viewer_cb = ViewerCallback(render_every=4)

    print("starting training — viewer window will open shortly...")
    print("the robot will be terrible at first, that is normal.")
    print("Ctrl+C to stop at any time. model saves automatically.")

    try:
        model.learn(
            total_timesteps = 1_000_000,
            callback        = viewer_cb,
            progress_bar    = True,
        )
    except KeyboardInterrupt:
        print("\ntraining stopped early")

    model.save("biped_walker")
    print("model saved to biped_walker.zip")
    print("to load later:  model = PPO.load('biped_walker')")
