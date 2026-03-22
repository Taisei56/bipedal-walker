import numpy as np
import copy
import mujoco
import mujoco.viewer
import utils
import time
from get_jacobian_robot import get_pos_3d_jacobians

K_xi = -1.5# tuning variable
K_pz = 5
K_pt = -20
dx_des_ = 0.0
dy_des_ = 0.0
dz_omega_ = 0.0
dy_rocking = -0.0
 
g = 9.81 # gravity
min_step = 0.10
T_des = 0.2
foot_bodies = {"Right": "right_foot",
               "Left": "left_foot"}
ctrl_r_hip_z = 0
ctrl_r_hip_y = 1
ctrl_r_hip = 2
ctrl_r_knee = 3
ctrl_r_foot = 4

ctrl_l_hip_z = 5
ctrl_l_hip_y = 6
ctrl_l_hip = 7
ctrl_l_knee = 8
ctrl_l_foot = 9


t = 0

ground = ["ground", "obstacle_box", "obstacle_box_2"]


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
    z_com = np.clip(z_com, 0.1, 1000)
    omega_n = np.sqrt(g/z_com)
    xi = x_com + dx_com/omega_n
    dxi = dx_com + ddx_com/omega_n
    xi_des = x_com + dx_des/omega_n

    p = xi - dxi/omega_n - K_xi*(xi - xi_des)

    return p


def get_p_y(y_com, dy_com, ddy_com, z_com, dy_des):
    """
    p is expressed in the x axis (forward) from the frame that exists at the stance foot location.
    y_com: is the position of the center of mass (com) in the y axis of the stance foot
    dy_com: is the velocity of the center of mass (com) in the y axis of the stance foot
    ddy_com: is the acceleration of the center of mass (com) in the y axis of the stance foot
    z_com: is the height of com above the stance foot. 
    dy_des: is the desiered forward velocity. 
    returns p, which is the continious target foot location
    """
    z_com = np.clip(z_com, 0.1, 1000)

    omega_n = np.sqrt(g/z_com)

    xi = y_com + dy_com/omega_n
    dxi = dy_com + ddy_com/omega_n
    xi_des = y_com + dy_des/omega_n

    p = xi - dxi/omega_n - K_xi*(xi - xi_des)

    return p


def get_height(z_com_target, h_target, h_0, t_start, t_now, e_step_down):
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
    tau = delta_t/T_des + e_step_down

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
    if ez < 0:
        v_z = K_pz * ez * vec_h
    else:
        v_z = 0.5*K_pz * ez * vec_h

    # sideways direction for roll correction
    vec_phi = np.array([0, -1, 0])

    er = rotation_matrix_to_axis_error(R_com, R_com_des)
    e_theta = er[1]
    v_theta = K_pt * e_theta * vec_theta

     # roll (x)
    e_phi = er[0]
    v_phi = K_pt * e_phi * vec_phi

    return -v_z + v_theta + v_phi


def swing_leg_controller(p_des_swing, p_swing):
    e_x = p_des_swing[0] - p_swing[0]
    e_y = p_des_swing[1] - p_swing[1]
    e_z = p_des_swing[2] - p_swing[2]

    return np.clip(10*np.array([e_x, 0.8*e_y, e_z]), -10, 10)


def turn_controller(q_l, q_r, dz_omega, t, stance_leg):
    """
    q_l: joint position for left leg hip z
    q_r: joint position for right leg hip z
    dz_omega: turn rate, radians
    t: time since stance switch 
    stance_leg: Left or Right
    """
    delta_omega = dz_omega*T_des
    t = np.clip(t, 0, T_des)
    if stance_leg == "Right":
        if delta_omega > 0:
            offset = np.min([delta_omega/2, q_r])
        else:
            offset = np.max([delta_omega/2, q_r])
        q_r_des = offset - delta_omega*(t/T_des)
        q_l_des = -delta_omega/2 + delta_omega*(t/T_des)
        dq_r = q_r_des - q_r
        dq_l = q_l_des - q_l
        return dq_r, dq_l
    else:
        if delta_omega > 0:
            offset = np.min([delta_omega/2, q_l])
        else:
            offset = np.max([delta_omega/2, q_l])
        q_l_des = offset - delta_omega*(t/T_des)
        q_r_des = - delta_omega/2 + delta_omega*(t/T_des)
        dq_r = q_r_des - q_r
        dq_l = q_l_des - q_l
        return dq_l, dq_r


def foot_pitch_error_stance(model, data, foot_body_name, R_stance_w):
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_body_name)

    R_foot_w = data.xmat[foot_id].reshape(3,3)

    # express foot in stance frame (removes yaw!)
    R = R_stance_w.T @ R_foot_w

    # pitch error (about y-axis)
    err = np.arctan2(-R[2,0], R[2,2])

    return err
    

def get_pose(stance_side, swing_side):

    p_stance_w, R_stance_w = utils.foot_end_frame_world(model, data, foot_bodies[stance_side], torso_name="torso")
    #utils.draw_frame(viewer=viewer, pos=p_stance_w, mat=R_stance_w, size=0.2)
    p_swing_w, _ = utils.foot_end_frame_world(model, data, foot_bodies[swing_side], torso_name="torso")
    p_swing = utils.world_p_to_frame(p_world=p_swing_w, p_frame=p_stance_w, R_frame=R_stance_w)
    torso_state = utils.torso_state_in_stance_frame(model, data, p_c=p_stance_w, R_c=R_stance_w, torso_name="torso")
    #utils.draw_frame(viewer=viewer, pos=torso_state["position_w"], mat=torso_state["orientation_w"], size=0.5)
    #utils.draw_frame(viewer=viewer, pos=np.array([0,0,0]), mat=np.eye(3), size=0.5)
    torso_state["position"][0] -= 0.08 
    torso_state["position"][2] -= 0.05
    err = foot_pitch_error_stance(model, data, foot_bodies[swing_side], R_stance_w)
    return p_swing, torso_state, err


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
        self.min_step_time = 0.1
        self.pre_step_time = 0.0
        self.contact_lifted = False
        # target height of torso com
        self.z_target = 0.39
        # target height of step 
        self.h_target = 0.1

        self.initialize(t0=t0)

    def get_stance_name(self):
        return self.stance["stance"]
        
    def get_swing_name(self):
        return self.swing["swing"]    
    

    def switch_leg(self, contact_l, contact_r, t):
        min_time = False
        if self.swing["swing"] == "Left":
            contact = contact_l
            stance = "Right"
            swing = "Left"
        else:
            contact = contact_r
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
        p_swing, torso_state, _ = get_pose(self.stance["stance"], self.swing["swing"])
        self.swing["p"] = p_swing
        self.torso["p"] = torso_state["position"]
        self.torso["v"] = torso_state["velocity"]
        self.torso["a"] = torso_state["acceleration"]
        self.torso["R"] = torso_state["orientation"]
        self.switch_properties["p_foot_0"] = copy.deepcopy(self.swing["p"])
        self.switch_properties["p_com_0"] = copy.deepcopy(self.torso["p"])
        self.switch_properties["t0"] = t0

    def step(self, contact_l, contact_r, q, dx_des, dy_des, dz_omega, t):
        """
        contact_l: False or True if in contact
        contact_r: False or True if in contact
        q: joint position
        """
        # change stance leg

        switch = self.switch_leg(contact_l, contact_r, t)
        # update_pose() # update p swing and p,v,a, R torso
        p_swing, torso_state, err = get_pose(self.stance["stance"], self.swing["swing"])
        
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
        
        dy_rock = 0
        if self.stance["stance"] == "Right":
            dy_rock = -dy_rocking
        else:
            dy_rock = dy_rocking

        p_y = get_p_y(y_com=self.torso["p"][1], 
                      dy_com=self.torso["v"][1], 
                      ddy_com=self.torso["a"][1], 
                      z_com=self.torso["p"][2], 
                      dy_des=dy_des+dy_rock)
        e_step_down = 0
        if self.stance["stance"] == "Right":
            if p_y < 0.0:
                e_step_down += -2*p_y
            elif p_y > 0.5:
                e_step_down += 2*(p_y-0.5)
            
            p_y = np.clip(p_y, a_min=0.05, a_max=0.6)
        else:
            if p_y > 0.0:
                e_step_down += 2*p_y
            elif p_y < -0.5:
                e_step_down += -2*(p_y+0.5)
            p_y = np.clip(p_y, a_min=-0.6, a_max=-0.05)
        if abs(p_x) > 0.5:
            e_step_down += 2*(abs(p_x)-0.5)
        p_x = np.clip(p_x, a_min=-0.6, a_max=0.6)
        # get_height()
        if switch:
            self.switch_properties["p_foot_0"] = copy.deepcopy(self.swing["p"])
            self.switch_properties["p_com_0"] = copy.deepcopy(self.torso["p"])
            self.switch_properties["t0"] = t

        z_swing, z_com = get_height(z_com_target=self.z_target, 
                                    h_target=self.h_target, 
                                    h_0=self.switch_properties["p_foot_0"][2], 
                                    t_start=self.switch_properties["t0"],
                                    t_now=t,
                                    e_step_down=e_step_down)
        # stance_foot_velocity()
        # velocity vecto to torso frame

        vel_vector = stance_foot_velocity(p_com=self.torso["p"], 
                                          z_com=z_com,
                                          R_com=self.torso["R"],
                                          R_com_des=self.R_target)
        # generate joint velocity 
        Jr, Jl = get_pos_3d_jacobians(q)
        if self.stance["stance"] == "Right":
            stance_jacobian = Jr
            swing_jacobian = Jl
        else:
            stance_jacobian = Jl
            swing_jacobian = Jr

        dqz_stance, dqz_swing = turn_controller(q[0], q[5], dz_omega, t-self.switch_properties["t0"], self.stance["stance"])
        v_t = self.torso["R"].T @ np.array([[vel_vector[0]], [vel_vector[1]], [vel_vector[2]]])
        vel_vector_ = np.array([[v_t[0,0]], [v_t[1,0]], [v_t[2,0]], [5*dqz_stance]])
        dq_stance = np.matmul(np.linalg.pinv(stance_jacobian), vel_vector_)
        
        vel_swing = swing_leg_controller(np.array([p_x, p_y, z_swing]), self.swing["p"])
        v_s = self.torso["R"].T @ np.array([[vel_swing[0]], [vel_swing[1]], [vel_swing[2]]])
        vel_swing_ = np.array([[v_s[0,0]], [v_s[1,0]], [v_s[2,0]], [5*dqz_swing]])


        dq_swing =  np.matmul(np.linalg.pinv(swing_jacobian), vel_swing_)


        
        K_ankle = 100
        v_swing_ankle = -K_ankle * err

        # return joint velovity commands

        if self.stance["stance"] == "Right":
            return {"right_hip_z": dq_stance[0,0], "right_hip_y": dq_stance[1,0], "right_hip": dq_stance[2,0], "right_knee": dq_stance[3,0], "right_ankle": 0.0, "left_hip_z": dq_swing[0,0], "left_hip_y": dq_swing[1,0], "left_hip": dq_swing[2,0], "left_knee": dq_swing[3,0], "left_ankle": v_swing_ankle}
        else:
            return {"left_hip_z": dq_stance[0,0], "left_hip_y": dq_stance[1,0], "left_hip": dq_stance[2,0], "left_knee": dq_stance[3,0], "left_ankle": 0.0, "right_hip_z": dq_swing[0,0], "right_hip_y": dq_swing[1,0], "right_hip": dq_swing[2,0], "right_knee": dq_swing[3,0], "right_ankle": v_swing_ankle}



model = mujoco.MjModel.from_xml_path("xml_files/biped_robot.xml")
data = mujoco.MjData(model)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_y_j")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(10)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(25)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-45)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(25)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-45)

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


# Initialize the controller once
controller = utils.KeyboardController(v_step=0.45, v_side=0.15, yaw_step=1.0)
#controller = utils.XboxController(v_step=0.45, v_side=0.15, yaw_step=2.0)
recording = False

from mujoco.usd import exporter

# 1. Initialize the exporter
# 'scene.xml' should be the same model you are using
usd_exporter = exporter.USDExporter(model)
fps = 30
last_render_time = 0

itt = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    walking_controller = Walking_controller(t0=t)

    while viewer.is_running():
        t += dt
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        # 2. Reset custom geoms for this frame
        
        # foot contact 
        contact = utils.geoms_contacting_geoms(model, data, ["left_shin_geom", "right_shin_geom", "right_foot_geom", "left_foot_geom"], ground)
       
        #print(contact, torso_state)
        right_hip_z_j  = get_joint_angle(model, data, "right_hip_z_j")
        right_hip_y_j  = get_joint_angle(model, data, "right_hip_y_j")
        right_hip  = get_joint_angle(model, data, "right_hip")
        right_knee = get_joint_angle(model, data, "right_knee")
        right_foot = get_joint_angle(model, data, "right_foot_j")
        
        left_hip_z_j   = get_joint_angle(model, data, "left_hip_z_j")
        left_hip_y_j   = get_joint_angle(model, data, "left_hip_y_j")
        left_hip   = get_joint_angle(model, data, "left_hip")
        left_knee  = get_joint_angle(model, data, "left_knee")
        left_foot = get_joint_angle(model, data, "left_foot_j")

        q = np.array([left_hip_z_j, left_hip_y_j, left_hip, left_knee, left_foot, 
                      right_hip_z_j, right_hip_y_j, right_hip, right_knee, right_foot])
        #dq = np.array([d_left_hip, d_left_knee, d_right_hip, d_right_knee])
        if itt % 30 == 0:
            # Just call this one function!
            dx_des_, dy_des_, dz_omega_ = controller.get_cmd()
            viewer.user_scn.ngeom = 0
            dq_c = walking_controller.step(contact_l=contact["left_foot_geom"],
                                        contact_r=contact["right_foot_geom"],
                                        q= q,
                                        dx_des=dx_des_,
                                        dy_des=dy_des_,
                                        dz_omega=dz_omega_,
                                        t=t)
            
        if recording:
            if data.time - last_render_time >= (1.0 / fps):
                usd_exporter.update_scene(data) 
                last_render_time = data.time

        if itt % 100 == 0:
            
            rec_cmd = controller.record_toggle()
            if rec_cmd == "start" and not recording:
                recording = True
                print(" Recording to USD buffer...")
            
            if rec_cmd == "stop" and recording:
                # 4. Write the buffer to the actual file
                usd_exporter.save_scene()
                recording = False
                print(" USD file saved as walk.usdc")
            viewer.sync()


        
        # Velocity to effort controller
        data.ctrl[ctrl_r_hip_z] = np.clip(dq_c["right_hip_z"], -10, 10)
        data.ctrl[ctrl_r_hip_y] = np.clip(dq_c["right_hip_y"], -10, 10)
        data.ctrl[ctrl_r_hip] = np.clip(dq_c["right_hip"], -10, 10)
        data.ctrl[ctrl_r_knee] = np.clip(dq_c["right_knee"], -10, 10)
        data.ctrl[ctrl_r_foot] = dq_c["right_ankle"]

        data.ctrl[ctrl_l_hip_z] = np.clip(dq_c["left_hip_z"], -10, 10)
        data.ctrl[ctrl_l_hip_y] = np.clip(dq_c["left_hip_y"], -10, 10)
        data.ctrl[ctrl_l_hip] = np.clip(dq_c["left_hip"], -10, 10)
        data.ctrl[ctrl_l_knee] = np.clip(dq_c["left_knee"], -10, 10)
        data.ctrl[ctrl_l_foot] = dq_c["left_ankle"]

        #print("Sim time:", data.time)
        
        t_next += dt
        t_ += dt
        itt +=1
        time.sleep(max(0.00, t_next - time.time()))


    
    
















