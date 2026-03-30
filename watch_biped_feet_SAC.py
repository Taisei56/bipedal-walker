import numpy as np
import mujoco
import mujoco.viewer
import time
import glob
import os
from stable_baselines3 import SAC

XML_PATH = "xml_files/biped_3d_feet.xml"

CTRL_R_HIP_Y = 0
CTRL_R_HIP   = 1
CTRL_R_KNEE  = 2
CTRL_R_FOOT  = 3
CTRL_L_HIP_Y = 4
CTRL_L_HIP   = 5
CTRL_L_KNEE  = 6
CTRL_L_FOOT  = 7

MIN_HEIGHT = 0.7

path = "checkpoints_sac/run_0329_1835/biped_sac_1600000_steps.zip"

print(f"loading: {path}")
model = SAC.load("checkpoints_sac/run_0329_1835/biped_sac_1600000_steps.zip")

# set up mujoco directly — no gym wrapper needed for watching
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data  = mujoco.MjData(mj_model)

torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")

joint_names = [
    "left_hip_y_j", "left_hip", "left_knee",
    "right_hip_y_j", "right_hip", "right_knee",
]
joint_ids = [
    mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
    for n in joint_names
]

lfoot_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
rfoot_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
ground_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

dt = mj_model.opt.timestep


def get_obs(cmd_vx, cmd_vy, prev_action):
    q  = np.array([mj_data.qpos[mj_model.jnt_qposadr[jid]] for jid in joint_ids])
    dq = np.array([mj_data.qvel[mj_model.jnt_dofadr[jid]]  for jid in joint_ids])

    torso_quat = mj_data.xquat[torso_id].copy()

    v6d = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data,
                             mujoco.mjtObj.mjOBJ_BODY, torso_id, v6d, 0)
    ang_vel    = v6d[:3]
    linear_vel = v6d[3:]

    lc = rc = 0.0
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        geoms = {c.geom1, c.geom2}
        if ground_id in geoms:
            if lfoot_id in geoms: lc = 1.0
            if rfoot_id in geoms: rc = 1.0

    obs = np.concatenate([
        q, dq, torso_quat, ang_vel, linear_vel,
        [cmd_vx, cmd_vy],
        prev_action,
        [lc, rc],
    ]).astype(np.float32)
    return np.clip(obs, -10.0, 10.0)


def reset():
    mujoco.mj_resetData(mj_model, mj_data)
    for name, deg in [("right_knee", -15), ("left_knee", -15)]:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_data.qpos[mj_model.jnt_qposadr[jid]] = np.deg2rad(deg)
    mujoco.mj_forward(mj_model, mj_data)
    cmd_vx = float(np.random.uniform(-0.4, 0.4))
    cmd_vy = float(np.random.uniform(-0.2, 0.2))
    return cmd_vx, cmd_vy


print("opening viewer...")
print("each episode has a random velocity command")
print("Ctrl+C to stop")

episode      = 0
step_count   = 0
total_reward = 0.0
prev_action  = np.zeros(6)
cmd_vx, cmd_vy = reset()

t_next = time.time()

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        obs    = get_obs(cmd_vx, cmd_vy, prev_action)
        action, _ = model.predict(obs, deterministic=True)

        scaled = action * 2.0
        mj_data.ctrl[CTRL_R_HIP_Y] = scaled[3]
        mj_data.ctrl[CTRL_R_HIP]   = scaled[4]
        mj_data.ctrl[CTRL_R_KNEE]  = scaled[5]
        mj_data.ctrl[CTRL_R_FOOT]  = 0.0
        mj_data.ctrl[CTRL_L_HIP_Y] = scaled[0]
        mj_data.ctrl[CTRL_L_HIP]   = scaled[1]
        mj_data.ctrl[CTRL_L_KNEE]  = scaled[2]
        mj_data.ctrl[CTRL_L_FOOT]  = 0.0

        mujoco.mj_step(mj_model, mj_data)

        prev_action  = action.copy()
        step_count  += 1

        torso_z = mj_data.xpos[torso_id][2]
        fell    = torso_z < MIN_HEIGHT
        timeout = step_count >= 10000

        if fell or timeout:
            episode     += 1
            reason       = "fell" if fell else "timeout"
            print(f"episode {episode} — cmd vx={cmd_vx:.2f} vy={cmd_vy:.2f} "
                  f"steps={step_count} ({reason})")
            step_count   = 0
            prev_action  = np.zeros(6)
            cmd_vx, cmd_vy = reset()

        viewer.sync()
        t_next += dt
        time.sleep(max(0.0, t_next - time.time()))