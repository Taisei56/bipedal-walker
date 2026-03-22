import numpy as np
import copy
import mujoco
import mujoco.viewer
import utils
import time
from get_jacobian_3d import get_pos_3d_jacobians

STEP_WIDTH   = 0.38
LIFT_HEIGHT  = 0.16
T_DES        = 0.9
WARMUP_SEC   = 2.0

K_pz = 8
K_pt = -12
K_pr = -12
K_px = -4    # forward/back position hold
K_sw = 20

Z_TARGET = 1.35
X_TARGET = 0.0

foot_bodies = {"Right": "right_shin", "Left": "left_shin"}
ground      = ["ground"]

ctrl_r_hip_y = 0
ctrl_r_hip   = 1
ctrl_r_knee  = 2
ctrl_l_hip_y = 3
ctrl_l_hip   = 4
ctrl_l_knee  = 5

model = mujoco.MjModel.from_xml_path("xml_files/biped_3d.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep


def get_q(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return data.qpos[model.jnt_qposadr[jid]]

def set_q(name, deg):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(deg)

def get_q_vec():
    return np.array([
        get_q("left_hip_y_j"), get_q("left_hip"),  get_q("left_knee"),
        get_q("right_hip_y_j"), get_q("right_hip"), get_q("right_knee"),
    ])


set_q("right_hip_y_j", -18)
set_q("left_hip_y_j",   18)
set_q("right_knee",    -15)
set_q("left_knee",     -15)
mujoco.mj_forward(model, data)


def rotation_error(R, R_des):
    skew = 0.5 * (R_des.T @ R - R.T @ R_des)
    return np.array([skew[2,1], skew[0,2], skew[1,0]])


def foot_height(h_0, t_start, t_now):
    tau = np.clip((t_now - t_start) / T_DES, 0.0, 1.0)
    h_m = max(1.1 * h_0, LIFT_HEIGHT)
    if tau < 0.5:
        return (h_m - h_0) * np.sin(np.pi * tau) + h_0
    else:
        return h_m * np.sin(np.pi * tau)


def stance_vel(p_com, R_com):
    R_des   = np.eye(3)
    vec_h   = -p_com / np.linalg.norm(p_com)
    vec_th  = np.array([-vec_h[2], 0, vec_h[0]])
    vec_phi = np.array([0, -1, 0])

    ez   = p_com[2] - Z_TARGET
    v_z  = K_pz * ez * vec_h

    er    = rotation_error(R_com, R_des)
    v_th  = K_pt * er[1] * vec_th
    v_phi = K_pr * er[0] * vec_phi

    # resist forward/back drift
    ex   = p_com[0] - X_TARGET
    v_px = K_px * ex * np.array([1, 0, 0])

    return -v_z + v_th + v_phi + v_px


def get_pose(stance, swing):
    p_st_w, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies[stance], torso_name="torso")
    p_sw_w, _ = utils.capsule_end_frame_world(
        model, data, foot_bodies[swing], torso_name="torso")
    p_sw = utils.world_p_to_frame(p_sw_w, p_st_w, R_st_w)
    ts   = utils.torso_state_in_stance_frame(
        model, data, p_c=p_st_w, R_c=R_st_w, torso_name="torso")
    return p_sw, ts


def apply_ctrl(dq):
    data.ctrl[ctrl_r_hip_y] = np.clip(dq["right_hip_y"], -5, 5)
    data.ctrl[ctrl_r_hip]   = np.clip(dq["right_hip"],   -5, 5)
    data.ctrl[ctrl_r_knee]  = np.clip(dq["right_knee"],  -5, 5)
    data.ctrl[ctrl_l_hip_y] = np.clip(dq["left_hip_y"],  -5, 5)
    data.ctrl[ctrl_l_hip]   = np.clip(dq["left_hip"],    -5, 5)
    data.ctrl[ctrl_l_knee]  = np.clip(dq["left_knee"],   -5, 5)


def warmup_ctrl():
    p_st_w, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies["Right"], torso_name="torso")
    ts = utils.torso_state_in_stance_frame(
        model, data, p_c=p_st_w, R_c=R_st_w, torso_name="torso")
    v  = stance_vel(ts["position"], ts["orientation"])
    R  = ts["orientation"]
    vv = R.T @ v.reshape(3,1)
    q  = get_q_vec()
    Jr, Jl = get_pos_3d_jacobians(q)
    dq_r = np.linalg.pinv(Jr) @ vv
    dq_l = np.linalg.pinv(Jl) @ vv
    apply_ctrl({
        "right_hip_y": dq_r[0,0], "right_hip": dq_r[1,0], "right_knee": dq_r[2,0],
        "left_hip_y":  dq_l[0,0], "left_hip":  dq_l[1,0], "left_knee":  dq_l[2,0],
    })


class Rocker:
    def __init__(self, t0):
        self.stance       = "Right"
        self.swing        = "Left"
        self.t_start      = t0
        self.pre_switch_t = 0.0
        self.min_dt       = 0.3
        p_sw, _ = get_pose(self.stance, self.swing)
        self.p_foot_0 = copy.deepcopy(p_sw)

    def _try_switch(self, contact_l, contact_r, t):
        c = contact_l if self.swing == "Left" else contact_r
        if c and t > self.pre_switch_t + self.min_dt:
            self.stance, self.swing = self.swing, self.stance
            self.pre_switch_t = t
            self.t_start      = t
            p_sw, _ = get_pose(self.stance, self.swing)
            self.p_foot_0 = copy.deepcopy(p_sw)

    def step(self, contact_l, contact_r, t):
        self._try_switch(contact_l, contact_r, t)
        p_sw, ts = get_pose(self.stance, self.swing)

        target_y = STEP_WIDTH if self.stance == "Right" else -STEP_WIDTH
        z_sw     = foot_height(self.p_foot_0[2], self.t_start, t)
        p_des    = np.array([0.0, target_y, z_sw])

        v_swing  = K_sw * (p_des - p_sw)
        v_stance = stance_vel(ts["position"], ts["orientation"])

        q    = get_q_vec()
        Jr, Jl = get_pos_3d_jacobians(q)
        J_st = Jr if self.stance == "Right" else Jl
        J_sw = Jl if self.stance == "Right" else Jr

        R     = ts["orientation"]
        dq_st = np.linalg.pinv(J_st) @ (R.T @ v_stance.reshape(3,1))
        dq_sw = np.linalg.pinv(J_sw) @ (R.T @ v_swing.reshape(3,1))

        if self.stance == "Right":
            return {
                "right_hip_y": dq_st[0,0], "right_hip": dq_st[1,0], "right_knee": dq_st[2,0],
                "left_hip_y":  dq_sw[0,0], "left_hip":  dq_sw[1,0], "left_knee":  dq_sw[2,0],
            }
        else:
            return {
                "left_hip_y":  dq_st[0,0], "left_hip":  dq_st[1,0], "left_knee":  dq_st[2,0],
                "right_hip_y": dq_sw[0,0], "right_hip": dq_sw[1,0], "right_knee": dq_sw[2,0],
            }


print("stabilizing...")
for _ in range(int(WARMUP_SEC / dt)):
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    warmup_ctrl()
print("opening viewer...")

t      = 0.0
t_next = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    rocker = Rocker(t0=t)
    while viewer.is_running():
        t += dt
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        viewer.user_scn.ngeom = 0

        contact = utils.geoms_contacting_geoms(
            model, data,
            ["left_shin_geom", "right_shin_geom"],
            ground
        )

        dq = rocker.step(
            contact_l=contact["left_shin_geom"],
            contact_r=contact["right_shin_geom"],
            t=t
        )

        apply_ctrl(dq)
        viewer.sync()
        t_next += dt
        time.sleep(max(0.0, t_next - time.time()))