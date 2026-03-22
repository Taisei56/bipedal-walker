import numpy as np
import copy
import mujoco
import mujoco.viewer
import utils
import time
from get_jacobian import get_pos_2d_jacobians

STEP_LENGTH  = 0.38   # foot spread front to back
LIFT_HEIGHT  = 0.16   # foot lift height
T_DES        = 0.9    # step duration
WARMUP_SEC   = 2.0    # stabilization time before viewer opens

# stance leg gains
K_pz = 8     # height
K_pt = -12   # pitch (tilt angle)
K_px = -4    # forward/back position hold — this is the new term

# swing foot gain
K_sw = 20

Z_TARGET = 1.35
X_TARGET = 0.0   # CoM should stay directly above stance foot in x

T_PX = -0.05

foot_bodies = {"Right": "right_shin", "Left": "left_shin"}
ground      = ["ground"]

ctrl_r_hip  = 0
ctrl_r_knee = 1
ctrl_l_hip  = 2
ctrl_l_knee = 3

model = mujoco.MjModel.from_xml_path("xml_files/biped.xml")
data  = mujoco.MjData(model)
dt    = model.opt.timestep


def get_q(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return data.qpos[model.jnt_qposadr[jid]]

def set_q(name, deg):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(deg)

def get_q_vec():
    return np.array([get_q("left_hip"), get_q("left_knee"),
                     get_q("right_hip"), get_q("right_knee")])


set_q("right_hip",   22)
set_q("left_hip",   -22)
set_q("right_knee", -12)
set_q("left_knee",  -12)
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
    """
    three corrections running simultaneously:
      v_z   — extend/retract leg to hold height
      v_th  — rotate leg to correct torso pitch (tilt angle)
      v_px  — push stance foot to correct horizontal drift
    """
    R_des  = np.eye(3)
    vec_h  = -p_com / np.linalg.norm(p_com)
    vec_th = np.array([-vec_h[2], 0, vec_h[0]])

    # height error
    ez  = p_com[2] - Z_TARGET
    v_z = K_pz * ez * vec_h

    # pitch error
    er   = rotation_error(R_com, R_des)
    v_th = K_pt * er[1] * vec_th

    # horizontal position error — CoM x relative to stance foot
    # positive x_com means CoM is in front of foot
    # correct by pushing foot forward (positive x velocity)
    ex   = p_com[0] - X_TARGET
    v_px = K_px * ex * np.array([1, 0, 0])

    return -v_z + v_th + v_px


def get_pose(stance, swing):
    p_st_w, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies[stance], torso_name="torso")
    p_sw_w, _ = utils.capsule_end_frame_world(
        model, data, foot_bodies[swing], torso_name="torso")
    p_sw = utils.world_p_to_frame(p_sw_w, p_st_w, R_st_w)
    ts   = utils.torso_state_in_stance_frame(
        model, data, p_c=p_st_w, R_c=R_st_w, torso_name="torso")
    ts["position"][0] += T_PX
    return p_sw, ts


def compute_dq(v_stance, v_swing, stance):
    q    = get_q_vec()
    Jr, Jl = get_pos_2d_jacobians(q)
    J_st = Jr if stance == "Right" else Jl
    J_sw = Jl if stance == "Right" else Jr

    _, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies[stance], torso_name="torso")
    ts = utils.torso_state_in_stance_frame(
        model, data, p_c=np.zeros(3), R_c=R_st_w, torso_name="torso")
    R = ts["orientation"]

    vs = R.T @ v_stance.reshape(3,1)
    vw = R.T @ v_swing.reshape(3,1)

    dq_st = np.linalg.pinv(J_st) @ np.array([[vs[0,0]], [vs[2,0]]])
    dq_sw = np.linalg.pinv(J_sw) @ np.array([[vw[0,0]], [vw[2,0]]])
    return dq_st, dq_sw


def apply_ctrl_both_stance(p_com, R_com):
    """warm-up: both legs act as stance"""
    v = stance_vel(p_com, R_com)
    q = get_q_vec()
    Jr, Jl = get_pos_2d_jacobians(q)

    p_st_w, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies["Right"], torso_name="torso")
    ts = utils.torso_state_in_stance_frame(
        model, data, p_c=p_st_w, R_c=R_st_w, torso_name="torso")
    R = ts["orientation"]
    vv = R.T @ v.reshape(3,1)
    vv2 = np.array([[vv[0,0]], [vv[2,0]]])

    dq_r = np.linalg.pinv(Jr) @ vv2
    dq_l = np.linalg.pinv(Jl) @ vv2
    data.ctrl[ctrl_r_hip]  = np.clip(dq_r[0,0], -5, 5)
    data.ctrl[ctrl_r_knee] = np.clip(dq_r[1,0], -5, 5)
    data.ctrl[ctrl_l_hip]  = np.clip(dq_l[0,0], -5, 5)
    data.ctrl[ctrl_l_knee] = np.clip(dq_l[1,0], -5, 5)


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

        target_x = -STEP_LENGTH if self.stance == "Right" else STEP_LENGTH
        z_sw     = foot_height(self.p_foot_0[2], self.t_start, t)
        p_des    = np.array([target_x, 0.0, z_sw])

        v_swing  = K_sw * (p_des - p_sw)
        v_stance = stance_vel(ts["position"], ts["orientation"])

        dq_st, dq_sw = compute_dq(v_stance, v_swing, self.stance)

        if self.stance == "Right":
            return {
                "right_hip": dq_st[0,0], "right_knee": dq_st[1,0],
                "left_hip":  dq_sw[0,0], "left_knee":  dq_sw[1,0],
            }
        else:
            return {
                "left_hip":  dq_st[0,0], "left_knee":  dq_st[1,0],
                "right_hip": dq_sw[0,0], "right_knee": dq_sw[1,0],
            }


# warm-up
print("stabilizing...")
for _ in range(int(WARMUP_SEC / dt)):
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    p_st_w, R_st_w = utils.capsule_end_frame_world(
        model, data, foot_bodies["Right"], torso_name="torso")
    ts = utils.torso_state_in_stance_frame(
        model, data, p_c=p_st_w, R_c=R_st_w, torso_name="torso")
    ts["position"][0] += T_PX
    apply_ctrl_both_stance(ts["position"], ts["orientation"])
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

        data.ctrl[ctrl_r_hip]  = np.clip(dq["right_hip"],  -5, 5)
        data.ctrl[ctrl_r_knee] = np.clip(dq["right_knee"], -5, 5)
        data.ctrl[ctrl_l_hip]  = np.clip(dq["left_hip"],   -5, 5)
        data.ctrl[ctrl_l_knee] = np.clip(dq["left_knee"],  -5, 5)

        viewer.sync()
        t_next += dt
        time.sleep(max(0.0, t_next - time.time()))

        