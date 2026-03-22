import mujoco
import numpy as np
import time

def bodies_contacting_objects(model, data, bodies, targets):
    body_ids = set(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies)
    target_ids = set(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t) for t in targets)

    contact = {b: False for b in bodies}

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]

        if b1 in body_ids and b2 in target_ids:
            contact[model.body(b1).name] = True
        elif b2 in body_ids and b1 in target_ids:
            contact[model.body(b2).name] = True

    return contact

def geoms_contacting_geoms(model, data, source_geoms, target_geoms):
    source_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in source_geoms
    )
    target_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in target_geoms
    )

    contact = {g: False for g in source_geoms}

    for i in range(data.ncon):
        c = data.contact[i]

        if c.geom1 in source_ids and c.geom2 in target_ids:
            contact[model.geom(c.geom1).name] = True
        elif c.geom2 in source_ids and c.geom1 in target_ids:
            contact[model.geom(c.geom2).name] = True

    return contact


def capsule_end_frame_world(model, data, body_name, torso_name="torso"):
    """
    Returns:
        p (3,)   position of capsule end in world frame
        R (3,3)  orientation frame:
                 z = world up
                 x = torso x projected onto ground
    """

    # --- IDs ---
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)

    # --- capsule body pose ---
    p_body = data.xpos[body_id]
    R_body = data.xmat[body_id].reshape(3, 3)

    # --- Find capsule length from geoms ---
    body_geoms = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id
    ]

    # assume capsule 
    g = body_geoms[0]
    half_length = model.geom_size[g][1]

    # local direction: negative z in body frame
    body_axis_world = R_body[:, 2]      # local +z
    p_end = p_body - body_axis_world * (2 * half_length)

    # --- Orientation construction ---
    # torso x-axis in world
    R_torso = data.xmat[torso_id].reshape(3, 3)

    x_torso = R_torso[:, 0]

    # world up
    z = np.array([0.0, 0.0, 1.0])

    # project torso x onto ground plane
    x = x_torso - np.dot(x_torso, z) * z
    x /= np.linalg.norm(x)

    # complete right-handed frame
    y = np.cross(z, x)

    R = np.column_stack((x, y, z))


    return p_end, R




def foot_end_frame_world(model, data, foot_body_name, torso_name="torso"):
    """
    Returns:
        p (3,)   position of body end in world frame
        R (3,3)  orientation frame:
                 z = world up
                 x = torso x projected onto ground
    """

    # --- IDs ---
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_body_name)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)

    # --- body body pose ---
    p_body = data.xpos[body_id]
    R_body = data.xmat[body_id].reshape(3, 3)


    # local direction: negative z in body frame
    body_axis_world = R_body[:, 2]      # local +z
    p_end = p_body - body_axis_world * 0.07 # offset (I got lazy with the implementation)

    # --- Orientation construction ---
    # torso x-axis in world
    R_torso = data.xmat[torso_id].reshape(3, 3)

    x_torso = R_torso[:, 0]

    # world up
    z = np.array([0.0, 0.0, 1.0])

    # project torso x onto ground plane
    x = x_torso - np.dot(x_torso, z) * z
    x /= np.linalg.norm(x)

    # complete right-handed frame
    y = np.cross(z, x)

    R = np.column_stack((x, y, z))


    return p_end, R

def world_p_to_frame(p_world, p_frame, R_frame):
    """
    Express world point in a local frame

    Args:
        p_world: (3,) point in world
        p_frame: (3,) frame origin in world
        R_frame: (3,3) frame rotation (columns = axes in world)

    Returns:
        p_local: (3,) point expressed in frame coordinates
    """
    return R_frame.T @ (p_world - p_frame)


def torso_state_in_stance_frame(model, data, p_c, R_c, torso_name="torso"):
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)

    # --- world quantities ---
    p_t = data.xpos[torso_id]
    R_t = data.xmat[torso_id].reshape(3, 3)

    # Get Velocity: [angular, linear]
    v_6d = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, v_6d, 0)
    w_t = v_6d[:3]
    v_t = v_6d[3:]

    # Get Acceleration: [angular, linear]
    a_6d = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, a_6d, 0)
    alpha_t = a_6d[:3]
    a_t = a_6d[3:]

    # --- transform to stance frame ---
    p = R_c.T @ (p_t - p_c)
    v = R_c.T @ v_t
    a = R_c.T @ a_t
    R = R_c.T @ R_t
    w = R_c.T @ w_t
    alpha = R_c.T @ alpha_t

    return {
        "position": p,
        "position_w": p_t,
        "velocity": v,
        "acceleration": a,
        "orientation": R,
        "orientation_w": R_t,
        "angular_velocity": w,
        "angular_acceleration": alpha
    }


class Joint_vel_PID_controller():
    def __init__(self, Kp, Ki, Kd, dt):
        self.kp = Kp
        self.ki = Ki
        self.kd = Kd
        self.dt = dt
        self.pre_omega = 0
        self.I = 0
        self.I_min = -20
        self.I_max = 20

    def update(self, omega, omega_des):
        e = omega_des - omega
        self.I += e * self.dt
        self.I = np.clip(self.I, self.I_min, self.I_max)
        delta = (omega - self.pre_omega)/self.dt
        self.pre_omega = omega
        torque = self.kp * e + self.ki*self.I - self.kd * delta
        return torque


def draw_frame(viewer, pos, mat, size=0.1, rgba_alpha=1.0):
    mat = np.asarray(mat).reshape(3, 3)
    colors = [
        [1, 0, 0, rgba_alpha],
        [0, 1, 0, rgba_alpha],
        [0, 0, 1, rgba_alpha]
    ]
    
    for i in range(3):
        axis_dir = mat[:, i]
        start = pos
        end = pos + axis_dir * size
        
        # Changed from mjv_makeConnector to mjv_connector
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            width=size * 0.05,
            from_=start, # Note: some versions use 'from_' and 'to' as arrays
            to=end
        )
        
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = colors[i]
        viewer.user_scn.ngeom += 1


def draw_point(viewer, p, radius=0.01, rgba=(1,0,0,1)):
    g = mujoco.mjvGeom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = [radius, 0, 0]
    g.pos[:] = p
    g.rgba[:] = rgba
    viewer.add_geom(g)

from pynput import keyboard
class KeyboardController:
    def __init__(self, v_step=0.5, v_side=0.1, yaw_step=0.8, alpha=0.01):
        self.v_step = v_step
        self.v_side = v_side
        self.yaw_step = yaw_step
        
        # Filtering parameter (0 < alpha <= 1)
        # Lower is smoother/slower, Higher is more responsive
        self.alpha = alpha
        
        # Target velocities (what the keys want)
        self.target_dx = 0.0
        self.target_dy = 0.0
        self.target_yaw = 0.0
        
        # Current velocities (the smoothed output)
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_yaw = 0.0
        
        self.pressed_keys = set()
        
        self.key_map = {
            keyboard.Key.up: 'up',
            keyboard.Key.down: 'down',
            keyboard.Key.left: 'left',
            keyboard.Key.right: 'right',
        }
        
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        if key in self.key_map:
            self.pressed_keys.add(self.key_map[key])
        try:
            if hasattr(key, 'char') and key.char is not None:
                self.pressed_keys.add(key.char.lower())
        except AttributeError: pass

    def _on_release(self, key):
        if key in self.key_map:
            self.pressed_keys.discard(self.key_map[key])
        try:
            if hasattr(key, 'char') and key.char is not None:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError: pass
    
    def record_toggle(self):
        return None

    def get_cmd(self):
        """Returns (dx, dy, yaw) with smoothed ramp-up and ramp-down."""
        # 1. Determine targets based on keys
        target_dx = 0.0
        target_dy = 0.0
        target_yaw = 0.0
        
        if 'up' in self.pressed_keys:    target_dx += self.v_step
        if 'down' in self.pressed_keys:  target_dx -= self.v_step
        if 'left' in self.pressed_keys:  target_dy += self.v_side
        if 'right' in self.pressed_keys: target_dy -= self.v_side
        if 'q' in self.pressed_keys:     target_yaw += self.yaw_step
        if 'e' in self.pressed_keys:     target_yaw -= self.yaw_step
        
        # 2. Smoothly update current velocities toward targets
        # Formula: current = current + alpha * (target - current)
        self.current_dx  += self.alpha * (target_dx - self.current_dx)
        self.current_dy  += self.alpha * (target_dy - self.current_dy)
        self.current_yaw += self.alpha * (target_yaw - self.current_yaw)
        
        # Optional: Snap to zero if values are very small to avoid "drifting"
        if abs(self.current_dx) < 0.001: self.current_dx = 0.0
        if abs(self.current_dy) < 0.001: self.current_dy = 0.0
        if abs(self.current_yaw) < 0.001: self.current_yaw = 0.0
        
        return self.current_dx, self.current_dy, self.current_yaw
    
""" # uncomment for xbox controller 
import pygame

class XboxController:
    def __init__(self, v_step=0.5, v_side=0.3, yaw_step=1.0, alpha=0.01, deadzone=0.1, debounce=0.3):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No controller detected")

        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        self.v_step = v_step
        self.v_side = v_side
        self.yaw_step = yaw_step
        self.alpha = alpha
        self.deadzone = deadzone

        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_yaw = 0.0

        # recording toggle state
        self._last_press = 0.0
        self._debounce = debounce
        self._recording = False

    def _filter(self, val):
        if abs(val) < self.deadzone:
            return 0.0
        return val

    def get_cmd(self):
        pygame.event.pump()

        # Left stick: forward/side
        lx = -self._filter(self.joy.get_axis(0))   # left/right
        ly = -self._filter(self.joy.get_axis(1))  # up/down (inverted)

        # Right stick: yaw
        rx = -self._filter(self.joy.get_axis(3))

        target_dx = ly * self.v_step
        target_dy = lx * self.v_side
        target_yaw = rx * self.yaw_step

        # smoothing
        self.current_dx  += self.alpha * (target_dx - self.current_dx)
        self.current_dy  += self.alpha * (target_dy - self.current_dy)
        self.current_yaw += self.alpha * (target_yaw - self.current_yaw)

        return self.current_dx, self.current_dy, self.current_yaw

    def record_toggle(self):
        pygame.event.pump()
        now = time.time()

        A = self.joy.get_button(0)  # A button
        B = self.joy.get_button(1)  # B button

        if now - self._last_press < self._debounce:
            return None

        if A and not self._recording:
            self._recording = True
            self._last_press = now
            return "start"

        if B and self._recording:
            self._recording = False
            self._last_press = now
            return "stop"

        return None
"""