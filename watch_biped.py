import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from train_biped import BipedEnv

print("loading trained model...")
model = PPO.load("biped_walker")

env = BipedEnv()
obs, _ = env.reset()

print("opening viewer — press Ctrl+C to stop")
print("each episode gets a random velocity command")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        viewer.sync()
        time.sleep(env.model.opt.timestep)

        if done:
            vx = info["cmd_vx"]
            vy = info["cmd_vy"]
            print(f"episode done — commanded vx={vx:.2f}  vy={vy:.2f}")
            obs, _ = env.reset()
