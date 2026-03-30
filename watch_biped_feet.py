import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from train_biped_feet import BipedFeetEnv
model = PPO.load("checkpoints_v3/biped_feet_v3_12000000_steps.zip")


env = BipedFeetEnv()
obs, _ = env.reset()

print("opening viewer...")
print("each episode gets a random velocity command")
print("Ctrl+C to stop")

episode = 0
total_reward = 0.0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        viewer.sync()
        time.sleep(env.model.opt.timestep)

        if done:
            episode += 1
            print(f"episode {episode} — cmd vx={info['cmd_vx']:.2f}  vy={info['cmd_vy']:.2f}  "
                  f"total reward={total_reward:.1f}")
            total_reward = 0.0
            obs, _ = env.reset()
