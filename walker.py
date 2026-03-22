"""
walker.py

Simulation entry point for the straight-legged bipedal walker.

The model uses lateral body sway over the stance leg to generate
ground clearance for the swing leg — no knee joint, no leg shortening.
Two pendulum motions (frontal-plane rock + sagittal-plane swing) are
coupled through the hip, and their interaction is what produces walking.

Usage:
    python sim/walker.py --mode passive --duration 10
    python sim/walker.py --mode actuated --duration 10 --render

Tested with MuJoCo 3.x and Python 3.10+.
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer


# ── constants ────────────────────────────────────────────────────────────────

MODEL_PATH = "model/bipedal.xml"

LEG_LENGTH  = 1.0   # metres
HIP_MASS    = 10.0  # kg
G           = 9.81  # m/s^2

# natural frequency of the pendulum (same omega as in the LIPM derivation)
OMEGA = np.sqrt(G / LEG_LENGTH)

# minimum lateral sway angle required for clearance at a given stride half-angle
# derived from: L(1 - cos(theta_f)) >= L(1 - cos(alpha))
# => theta_f >= alpha
def min_sway_angle(stride_half_angle):
    return stride_half_angle


# ── initial state ────────────────────────────────────────────────────────────

def set_initial_state(model, data, theta_f=0.08, theta_s=0.15):
    """
    Place the robot in a configuration that approximates mid-stride.

    theta_f: initial lateral tilt (radians) — should satisfy theta_f >= theta_s
    theta_s: initial sagittal swing angle (radians)
    """
    # check clearance condition before starting
    if theta_f < min_sway_angle(theta_s):
        print(f"Warning: lateral sway {theta_f:.3f} rad may be insufficient "
              f"for stride angle {theta_s:.3f} rad. Expect foot scuffing.")

    # joint indices — order matches actuator/joint definition in xml
    # right leg
    data.qpos[7]  = theta_f   # right frontal
    data.qpos[8]  = -theta_s  # right sagittal (stance, trailing)
    # left leg
    data.qpos[9]  = -theta_f  # left frontal
    data.qpos[10] = theta_s   # left sagittal (swing, leading)

    # small initial lateral velocity to get the rock going
    data.qvel[6] = 0.3   # right frontal rate
    data.qvel[9] = -0.3  # left frontal rate

    mujoco.mj_forward(model, data)


# ── controllers ──────────────────────────────────────────────────────────────

def passive_controller(model, data):
    """
    Zero torque. Walking driven entirely by gravity (slope required).
    Tests whether passive dynamics alone can sustain the gait.
    """
    data.ctrl[:] = 0.0


def energy_injection_controller(model, data):
    """
    Minimal actuation to sustain walking on flat ground.

    Strategy: apply a small lateral hip torque timed to amplify the
    natural rocking motion, similar to ankle push-off in human walking.
    This is the simplest possible active controller — one parameter (gain).

    TODO: tune gain, explore timing relative to foot-strike event,
    consider switching to a proper Poincare-based controller later.
    """
    gain = 1.2

    # read current joint angles and rates
    right_frontal_angle = data.qpos[7]
    left_frontal_angle  = data.qpos[9]
    right_frontal_rate  = data.qvel[6]
    left_frontal_rate   = data.qvel[9]

    # apply torque in the direction of current lateral motion
    # (positive feedback timed to amplify the rock, not damp it)
    data.ctrl[0] = gain * right_frontal_rate   # right frontal
    data.ctrl[2] = gain * left_frontal_rate    # left frontal

    # sagittal torques: zero for now, let gravity handle the swing
    data.ctrl[1] = 0.0
    data.ctrl[3] = 0.0


# ── foot contact detection ───────────────────────────────────────────────────

def get_foot_contacts(model, data):
    """
    Returns which feet are currently in contact with the ground.
    Used to detect foot-strike events and switch stance/swing roles.
    """
    contacts = {"right": False, "left": False}
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        for name in [geom1, geom2]:
            if name and "right_foot" in name:
                contacts["right"] = True
            if name and "left_foot" in name:
                contacts["left"] = True
    return contacts


# ── logging ──────────────────────────────────────────────────────────────────

class GaitLogger:
    """
    Records hip position, foot contacts, and joint angles over time.
    Used to compute stride length, cadence, and cost of transport post-hoc.
    """
    def __init__(self):
        self.time       = []
        self.hip_pos    = []
        self.contacts   = []
        self.joint_angles = []

    def record(self, data, contacts):
        self.time.append(data.time)
        self.hip_pos.append(data.sensor("hip_pos").data.copy())
        self.contacts.append(contacts)
        self.joint_angles.append(data.qpos[7:11].copy())

    def summary(self):
        if len(self.hip_pos) < 2:
            return
        pos = np.array(self.hip_pos)
        forward_displacement = pos[-1, 1] - pos[0, 1]
        total_time = self.time[-1] - self.time[0]
        print(f"\n── gait summary ──────────────────────")
        print(f"  duration          : {total_time:.2f} s")
        print(f"  forward travel    : {forward_displacement:.3f} m")
        print(f"  average speed     : {forward_displacement/total_time:.3f} m/s")
        print(f"──────────────────────────────────────\n")


# ── main simulation loop ─────────────────────────────────────────────────────

def run(mode="passive", duration=10.0, render=False):

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    logger = GaitLogger()

    set_initial_state(model, data)

    controller = passive_controller if mode == "passive" else energy_injection_controller

    print(f"Running {mode} simulation for {duration}s ...")
    print(f"  leg length : {LEG_LENGTH} m")
    print(f"  hip mass   : {HIP_MASS} kg")
    print(f"  omega      : {OMEGA:.4f} rad/s")

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while data.time < duration and viewer.is_running():
                controller(model, data)
                mujoco.mj_step(model, data)
                contacts = get_foot_contacts(model, data)
                logger.record(data, contacts)
                viewer.sync()
    else:
        while data.time < duration:
            controller(model, data)
            mujoco.mj_step(model, data)
            contacts = get_foot_contacts(model, data)
            logger.record(data, contacts)

    logger.summary()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Straight-legged bipedal walker simulation")
    parser.add_argument("--mode",     type=str,   default="passive",
                        choices=["passive", "actuated"],
                        help="passive: gravity-driven on slope | actuated: flat ground")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="simulation duration in seconds")
    parser.add_argument("--render",   action="store_true",
                        help="open MuJoCo viewer")
    args = parser.parse_args()

    run(mode=args.mode, duration=args.duration, render=args.render)
