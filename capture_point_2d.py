import numpy as np
import copy
import mujoco
import mujoco.viewer
import utils
import time
from get_jacobian import get_pos_2d_jacobians

# These are not perfectly tuned
K_xi = -1 # tuning variable for capture point location
K_pz = 5 # gain for torso height
K_pt = -8 # gain for torso angle 
dx_des_ = 0.6 # desiserd velocity 
K_s = 20  # gain for swing foot tracking

T_px = - 0.05 # constant offset of COM from torso fame (the legs have a mass) 

g = 9.81 # gravity
T_des = 0.4 # how much time a step should take
foot_bodies = {"Right": "right_shin",
               "Left": "left_shin"}

# numbering of control joints in the XML
ctrl_r_hip = 0
ctrl_r_knee = 1
ctrl_l_hip = 2
ctrl_l_knee = 3

t = 0

ground = ["ground", "obstacle_box", "obstacle_box2" , "obstacle_box3", "obstacle_box4", "obstacle_box5", "obstacle_box6"]


model = mujoco.MjModel.from_xml_path("xml_files/biped.xml")
data = mujoco.MjData(model)

def get_p(x_com, dx_com, ddx_com, z_com, dx_des):
    """
    p is expressed in the x axis (forward) from the frame that exists at the stance foot location.
    x_com: is the position of the center of mass (com) in the x axis of the stance foot
    dx_com: is the velocity of the center of mass (com) in the x axis of the stance foot
    ddx_com: is the acceleration of the center of mass (com) in the x axis of the stance foot
    z_com: is the height of com above the stance foot. 
    dx_des: is the desiered forward velocity. 
    returns p, which is the continious target foot location
    """
    omega_n = np.sqrt(g/z_com)

    xi = x_com + dx_com/omega_n
    dxi = dx_com + ddx_com/omega_n
    xi_des = x_com + dx_des/omega_n

    p = xi - dxi/omega_n - K_xi*(xi - xi_des)

    return p


def get_target_0(x_com_0, dx_com_0, z_com_0, dx_des_0):
    # This is constant for each step 
    omega_0 = np.sqrt(g/z_com_0)
    xi_0 = x_com_0 + dx_com_0/omega_0
    xi_des_0 = x_com_0 + dx_des_0/omega_0
    p = (xi_des_0 - xi_0*np.exp(omega_0*T_des))/(1 - np.exp(omega_0*T_des))

    x_target = p + (xi_0 - p)*np.exp(omega_0*T_des)

    return x_target


def get_height(z_com_target, h_target, h_0, t_start, t_now):
    """
    returns z_1, z_com: where z_1 is the current target heright of the swing foot, 
        and z_com is the target height of the com from the stance frame. 
    z_com_target: is the target height of com
    h_target: is the foot lift height
    h_0: is the swing foot height at leg switch 
    t_start: is the time at leg switch
    t_now: is the current time
    """
    #time based
    t = t_now
    delta_t = t - t_start
    tau = delta_t/T_des

    def sin_adapt(tau, h=h_target, h0=h_0):
        tau = np.clip(tau, a_min=0.0, a_max=1.0)
        h_m = np.max([1.1*h0, h]) # this makes it so that the swing foot is lifted even for big a step down.
        s = tau 
        if tau < 0.5:
            h_t = (h_m-h0) * np.sin(np.pi * s) + h0
        else:            
            h_t = h_m * np.sin(np.pi * s)
        return h_t
    z_1 = sin_adapt(tau=tau, h=h_target, h0=h_0)

    z_offset = 0.0 # for when tau >1 
    if tau > 1:
        z_offset = -tau+1
    
    return z_1+z_offset, z_com_target+z_offset


def rotation_matrix_to_axis_error(R, R_d):
    # Skew-symmetric
    skew = 0.5 * (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) 
    # vee mapping (extract vector from skew-symmetric matrix)
    e_R = np.array([skew[2,1], skew[0,2], skew[1,0]])
    return e_R


def stance_foot_velocity(p_com, z_com, R_com, R_com_des):
    """
    This fuction genrates a velocity vector for the stance foot in the stance frame.
    The purpose of this function is to regulate orientaiton and height of the com. 
    p_com: is in foot frame
    z_com: target height from get height 
    R_com: is if foot frame
    R_com_des: is in foot frame

    """
    vec_h = -p_com/np.linalg.norm(p_com)
    vec_theta = np.array([-vec_h[2], 0, vec_h[0]])
    z = p_com[2]
    ez = z-z_com
    v_z = K_pz * ez * vec_h

    er = rotation_matrix_to_axis_error(R_com, R_com_des)
    e_theta = er[1]
    v_theta = K_pt * e_theta * vec_theta

    return -v_z + v_theta


def swing_leg_controller(p_des_swing, p_swing):
    e_x = p_des_swing[0] - p_swing[0]
    e_z = p_des_swing[2] - p_swing[2]

    return K_s*np.array([e_x, 0.0, e_z])
    

def get_pose(stance_side, swing_side):

    p_stance_w, R_stance_w = utils.capsule_end_frame_world(model, data, foot_bodies[stance_side], torso_name="torso")
    p_swing_w, _ = utils.capsule_end_frame_world(model, data, foot_bodies[swing_side], torso_name="torso")
    p_swing = utils.world_p_to_frame(p_world=p_swing_w, p_frame=p_stance_w, R_frame=R_stance_w)
    torso_state = utils.torso_state_in_stance_frame(model, data, p_c=p_stance_w, R_c=R_stance_w, torso_name="torso")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, torso_state["orientation_w"].reshape(9))    
    
    utils.draw_frame(viewer=viewer, pos=torso_state["position_w"], mat=torso_state["orientation_w"], size=0.5)
    utils.draw_frame(viewer=viewer, pos=np.array([0,0,0]), mat=np.eye(3), size=0.5)
    torso_state["position"][0] += T_px
    return p_swing, torso_state


class Walking_controller():
    def __init__(self, t0):

        # "stance": left or right,
        self.stance = {"stance": "Right"}
        # "swing": left or right, "p": position in stance frame
        self.swing = {"swing": "Left", "p": None}
        # "torso_id": body id of torso, "p": position in stance frame, "v": velocity, "a": acceleration, "R": Orientation in stance frame. 
        self.torso = {"p": None, "v": None, "a": None, "R": None}

        self.switch_properties = {"t0": None, "p_foot_0": None, "p_com_0": None, "x_target": None}

        self.R_target = np.eye(3)
        self.min_step_time = 0.1 # minimize contact detection chatter
        self.pre_step_time = 0.0
        self.contact_lifted = False

        # target height of torso com
        self.z_target = 1.45 
        # target height of step 
        self.h_target = 0.2

        self.initialize(t0=t0)

    def get_stance_name(self):
        return self.stance["stance"]
        
    def get_swing_name(self):
        return self.swing["swing"]    
    

    def switch_leg(self, contact_l, contact_r, t):
        min_time = False
        if self.swing["swing"] == "Left":
            contact = contact_l
            stance_c = contact_r
            stance = "Right"
            swing = "Left"
        else:
            contact = contact_r
            stance_c = contact_l
            stance = "Left"
            swing = "Right"

        if contact == False:
            self.contact_lifted = True

        if t > self.pre_step_time + self.min_step_time:
            min_time = True

        if contact == True and min_time:
            self.contact_lifted = False
            self.stance = {"stance": swing}
            self.swing = {"swing": stance}
            self.pre_step_time = t
            return True
        else: 
            return False
    
    def initialize(self, dx_des=0.0, t0=0.0):
        p_swing, torso_state = get_pose(self.stance["stance"], self.swing["swing"])
        self.swing["p"] = p_swing
        self.torso["p"] = torso_state["position"]
        self.torso["v"] = torso_state["velocity"]
        self.torso["a"] = torso_state["acceleration"]
        self.torso["R"] = torso_state["orientation"]
        self.switch_properties["x_target"] = get_target_0(x_com_0=self.torso["p"][0], 
                                                              dx_com_0=self.torso["v"][0], 
                                                              z_com_0=self.torso["p"][2], 
                                                              dx_des_0=dx_des)
        self.switch_properties["p_foot_0"] = copy.deepcopy(self.swing["p"])
        self.switch_properties["p_com_0"] = copy.deepcopy(self.torso["p"])
        self.switch_properties["t0"] = t0

    def step(self, contact_l, contact_r, q, dx_des, t):
        """
        p_swing: position of swing foot, in stance frame
        p_torso: position of torso, in stance frame
        v_torso: velocity of torso, in stance frame
        a_torso: acceleration of torso, in stance frame
        contact_l: False or True if in contact
        contact_r: False or True if in contact
        q: joint position
        """
        # change stance leg

        switch = self.switch_leg(contact_l, contact_r, t)
        # update_pose() # update p swing and p,v,a, R torso
        p_swing, torso_state = get_pose(self.stance["stance"], self.swing["swing"])
        
        self.swing["p"] = p_swing
        self.torso["p"] = torso_state["position"]
        self.torso["v"] = torso_state["velocity"]
        self.torso["a"] = torso_state["acceleration"]
        self.torso["R"] = torso_state["orientation"]
        
        # get_p()
        p_x = get_p(x_com=self.torso["p"][0], 
                    dx_com=self.torso["v"][0], 
                    ddx_com=self.torso["a"][0], 
                    z_com=self.torso["p"][2], 
                    dx_des=dx_des)
        
        # get_height()
        if switch:
            self.switch_properties["x_target"] = get_target_0(x_com_0=self.torso["p"][0], 
                                                              dx_com_0=self.torso["v"][0], 
                                                              z_com_0=self.torso["p"][2], 
                                                              dx_des_0=dx_des)
            self.switch_properties["p_foot_0"] = copy.deepcopy(self.swing["p"])
            self.switch_properties["p_com_0"] = copy.deepcopy(self.torso["p"])
            self.switch_properties["t0"] = t

        z_swing, z_com = get_height(z_com_target=self.z_target, 
                                    h_target=self.h_target, 
                                    h_0=self.switch_properties["p_foot_0"][2], 
                                    t_start=self.switch_properties["t0"],
                                    t_now=t)
        
        # stance_foot_velocity()
        # velocity vecto to torso frame

        vel_vector = stance_foot_velocity(p_com=self.torso["p"], 
                                          z_com=z_com,
                                          R_com=self.torso["R"],
                                          R_com_des=self.R_target)
        # generate joint velocity 
        Jr, Jl = get_pos_2d_jacobians(q)
        if self.stance["stance"] == "Right":
            stance_jacobian = Jr
            swing_jacobian = Jl
        else:
            stance_jacobian = Jl
            swing_jacobian = Jr

        v_t = self.torso["R"].T @ np.array([[vel_vector[0]], [vel_vector[1]], [vel_vector[2]]])
        vel_vector_ = np.array([[v_t[0,0]], [v_t[2,0]]])

        dq_stance = np.matmul(np.linalg.pinv(stance_jacobian),vel_vector_)

        vel_swing = swing_leg_controller(np.array([p_x, 0.0, z_swing]), self.swing["p"])
    
        v_s = self.torso["R"].T @ np.array([[vel_swing[0]], [vel_swing[1]], [vel_swing[2]]])
        vel_swing_ = np.array([[v_s[0,0]], [v_s[2,0]]])


        dq_swing =  np.matmul(np.linalg.pinv(swing_jacobian), vel_swing_)
        # return joint velovity commands

        if self.stance["stance"] == "Right":
            return {"right_hip": dq_stance[0,0], "right_knee": dq_stance[1,0], "left_hip": dq_swing[0,0], "left_knee": dq_swing[1,0]}
        else:
            return {"left_hip": dq_stance[0,0], "left_knee": dq_stance[1,0], "right_hip": dq_swing[0,0], "right_knee": dq_swing[1,0]}




jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-5)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-5)

mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)
t_next = time.time()
t_ = 0
dt = model.opt.timestep

def get_joint_angle(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[jid]
    return data.qpos[qpos_addr]

def get_joint_velocity(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qvel_addr = model.jnt_dofadr[jid]
    return data.qvel[qvel_addr]


with mujoco.viewer.launch_passive(model, data) as viewer:
    walking_controller = Walking_controller(t0=t)

    while viewer.is_running():
        t += dt
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        # 2. Reset custom geoms for this frame
        viewer.user_scn.ngeom = 0
        # foot contact 
        contact = utils.geoms_contacting_geoms(model, data, ["left_shin_geom", "right_shin_geom"], ground)
       
        right_hip  = get_joint_angle(model, data, "right_hip")
        right_knee = get_joint_angle(model, data, "right_knee")
        left_hip   = get_joint_angle(model, data, "left_hip")
        left_knee  = get_joint_angle(model, data, "left_knee")

        q = np.array([left_hip, left_knee, right_hip, right_knee])

        dq_c = walking_controller.step(contact_l=contact["left_shin_geom"],
                                       contact_r=contact["right_shin_geom"],
                                       q= q,
                                       dx_des=-dx_des_,
                                       t=t)

        # Velocity to effort controller
        data.ctrl[ctrl_r_hip] = np.clip(dq_c["right_hip"], -5, 5)
        data.ctrl[ctrl_r_knee] = np.clip(dq_c["right_knee"], -5, 5)
        data.ctrl[ctrl_l_hip] = np.clip(dq_c["left_hip"], -5, 5)
        data.ctrl[ctrl_l_knee] = np.clip(dq_c["left_knee"], -5, 5)

        viewer.sync()
        t_next += dt
        t_ += dt
        time.sleep(max(0.0, t_next - time.time()))


    
    
















