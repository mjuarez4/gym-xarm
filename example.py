import gymnasium as gym
import gym_xarm
import os
import imageio
from gym_xarm.tasks.lift import Lift
from gym_xarm.tasks.controllers import LiftController
import numpy as np


os.environ["MUJOCO_GL"] = "glfw"
env = gym.make("gym_xarm/XarmLift-v0", render_mode="rgb_array")
lift_env = env.unwrapped  
controller = LiftController(lift_env)
observation, info = env.reset()


frames = []
for _ in range(1000):
    #action = lift_env.position_above_block()  

    #observation, reward, terminated, truncated, info = env.step(action)
    #obs, reward = 
    
    #lift_env.execute()
    controller.position_above_block()
    controller.grab_block()
    #
    #print("completed")
    #lift_env.lift_block()
    print(lift_env.eef)
    image = env.render()
    frames.append(image)

    #if terminated or truncated:
    #    observation, info = env.reset()


print("gripper angle BEFORE:", lift_env.gripper_angle)
for _ in range(10):
    controller.close_gripper()
    #print("gripper angle:", lift_env.gripper_angle)

    image = env.render()
    frames.append(image)

for _ in range(1000):
    controller.maintain_grip_and_lift()
    #lift_env.lift_block()
    #lift_env.close_gripper()
    #lift_env.lift_eef()
    image = env.render()
    frames.append(image)

for _ in range(1000):
    controller.move_to_bowl()
    #lift_env.lift_block()
    #lift_env.close_gripper()
    #lift_env.lift_eef()
    image = env.render()
    frames.append(image)

for _ in range(1000):
    #print(f"Gripper angle: {controller.gripper_angle}")
    #print(f"Block Z: {controller.obj[2]}")

    controller.drop_in_bowl()
    #lift_env.lift_block()
    #lift_env.close_gripper()
    #lift_env.lift_eef()
    image = env.render()
    frames.append(image)
env.reset()

for _ in range(10):
    #controller.position_above_block()
    controller.grab_block()
    #xss
    #print("completed")
    #lift_env.lift_block()
    print(lift_env.block)
    image = env.render()
    frames.append(image)

    
env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
