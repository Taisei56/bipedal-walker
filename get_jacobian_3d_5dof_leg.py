import pinocchio as pin
import numpy as np

# -----------------------------
# Load model (relative model)
# -----------------------------
model = pin.buildModelFromUrdf("xml_files/biped_3d_5dof_leg.urdf")

# -----------------------------
# Add frames BEFORE createData
# -----------------------------
# addFrame returns an integer frame id

left_foot_fid  = model.getFrameId("left_foot")
right_foot_fid = model.getFrameId("right_foot")


# NOW create data
data = model.createData()

# -----------------------------
# helper: get velocity indices for a joint (handles multi-dof)
# -----------------------------
def joint_velocity_indices(model, joint_name):
    jid = model.getJointId(joint_name)
    start = model.idx_vs[jid]                 # starting index in v
    nv = model.joints[jid].nv                 # number of velocity dofs for this joint
    return list(range(start, start + nv))

# -----------------------------
# Joint state (example)
# -----------------------------
q = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 
              0.0, 0.0, 0.5, 0.0, 0.0])   # use your desired q

# -----------------------------
# Correct Jacobian function
# -----------------------------
def get_pos_3d_jacobians(q):
    """
    q = [left_hip, left_knee, right_hip, right_knee]
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    Jw_right = pin.computeFrameJacobian(
        model, data, q, right_foot_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    Jw_left = pin.computeFrameJacobian(
        model, data, q, left_foot_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

 
    Jr_lin = Jw_right[[0,1,2,5], :]
    Jl_lin = Jw_left[[0,1,2,5], :]

    right_cols = joint_velocity_indices(model, "right_hip_yaw") + joint_velocity_indices(model, "right_hip_roll") + joint_velocity_indices(model, "right_hip_pitch") + joint_velocity_indices(model, "right_knee") + joint_velocity_indices(model, "right_ankle")
    left_cols  = joint_velocity_indices(model, "left_hip_yaw") + joint_velocity_indices(model, "left_hip_roll")  + joint_velocity_indices(model, "left_hip_pitch")  + joint_velocity_indices(model, "left_knee")  + joint_velocity_indices(model, "left_ankle")
    J_r = Jr_lin[:, right_cols]
    J_l = Jl_lin[:, left_cols]
    return J_r, J_l

print(get_pos_3d_jacobians(q))
