import json_numpy as jp
jp.patch()
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Optional

import jax
# import keyboard
import numpy as np
#import pynput
# spacemouse imports
#import pyspacemouse
#import requests
#from pynput.keyboard import Key
#from pyspacemouse.pyspacemouse import SpaceNavigator


@dataclass
class SpaceMouseConfig:

    # sensitivity
    scale = np.array(
        [
            5.0,
            5.0,
            5.0,
            0.02,
            0.02,
            0.02,
            100,
        ]
    )
    flip = [1, 1, 1, -1, -1, -1, 1]
    order = [0, 1, 2, 3, 5, 4, 6]


class VelocitySMC(SpaceMouseConfig):

    # xyz, rpy, gripper
    # yxz, rpy, gripper
    # rpy is rotation along x, y, z
    scale = np.array(
        [
            125.0,
            125.0,
            125.0,
            0.5,
            0.5,
            0.5,
            200,
        ]
    )

    order = [1, 0, 2, 5, 3, 4, 6]
    # applied after order
    flip = [-1, 1, 1, -1, -1, -1, 1]
    sensitivity = 1.5  # from 1 -> inf


def ema_2d(data, window):
    """Calculate EMA for 2D array"""

    alpha = 2 / (window + 1)  # Smoothing factor
    ema = np.empty_like(data)
    ema[0, :] = data[0, :]  # Start with first row

    for i in range(1, data.shape[0]):
        ema[i, :] = alpha * data[i, :] + (1 - alpha) * ema[i - 1, :]

    return ema


class SpaceMouseController:  # from 1 -> inf

    def __init__(self, cfg: SpaceMouseConfig = VelocitySMC()):

        self.cfg = cfg
        self.state: Optional[SpaceNavigator] = SpaceNavigator(
            t=0.0,
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            buttons=(0, 0),
        )

        self._running = False
        self.thread = None

        self.freq = 300  # Frequency Hz
        self.dt = 1.0 / self.freq

        self.hist = np.zeros((self.freq, 7))

        self.success = pyspacemouse.open(
            dof_callback=self.set_state,
            button_callback=None,
            button_callback_arr=[],
        )

        if self.success:
            print("SpaceMouse connected.")
        else:
            print("Failed to connect to SpaceMouse.")

        self.start()

    def start(self):
        """Start reading the SpaceMouse in the background."""
        if not self._running and self.success:
            self._running = True
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        time.sleep(0.1)

    def _read(self):
        """Read input in a loop and update the last state."""
        while self._running and self.success:
            pyspacemouse.read()
            time.sleep(self.dt)

    def stop(self):
        """Stop reading and close the SpaceMouse."""
        if self._running:
            self._running = False
            if self.thread is not None:
                self.thread.join()  # Wait for the thread to finish
                del self.thread
            pyspacemouse.close()  # Close the SpaceMouse device when stopping
            print("SpaceMouse disconnected.")

    def set_state(self, state):
        self.state = state

    def read(self, as_dict=False):

        if as_dict:
            print(type(self.state))
            return self.state._asdict()

        else:  # x,y,z,roll,pitch,yaw,gripper
            gripper = self.state.buttons
            gripper = -1 * gripper[0] + gripper[1]
            out = np.array(
                [
                    self.state.x,
                    self.state.y,
                    self.state.z,
                    self.state.pitch,
                    self.state.yaw,
                    self.state.roll,
                    gripper,
                ]
            )
            out = out[self.cfg.order]
            # out = out ** self.cfg.sensitivity * np.sign(out) # make it less sensitive
            out = out * self.cfg.scale * self.cfg.flip

            self.hist = np.roll(self.hist, -1, axis=0)  # smooth
            self.hist[-1] = out
            # out = ema_2d(self.hist, 10)[-1]

            # out = np.where(out > self.hist[-2], out, 0)
            return out


class Controller(ABC):

    def __init__(self):
        pass


class KeyboardController:
    def __init__(self):

        self.vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.running = True
        self.keys_pressed = set()
        self.listener = pynput.keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )

        self.size = 10.0
        self.rsize = 0.05
        self.gsize = 50.0

        self.funcs = {}

        self.run()  # Begin the run method at the end of initialization

    def on_key_press(self, key):

        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char

        if key in self.funcs:
            self.funcs[key]()
        self.keys_pressed.add(key)
        self.update_vector()

    def on_key_release(self, key):

        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char
        self.keys_pressed.discard(key)
        self.update_vector()

    def update_vector(self):
        self.vec = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # Reset vec to 0 at the beginning

        shift_pressed = (
            Key.shift in self.keys_pressed or Key.shift_r in self.keys_pressed
        )

        if Key.up in self.keys_pressed:
            if shift_pressed:
                self.vec[4] += self.rsize  # Increase ry value
            else:
                self.vec[1] += self.size  # Increase y value
        if Key.down in self.keys_pressed:
            if shift_pressed:
                self.vec[4] -= self.rsize  # Decrease ry value
            else:
                self.vec[1] -= self.size  # Decrease y value
        if Key.left in self.keys_pressed:
            if shift_pressed:
                self.vec[3] -= self.rsize  # Decrease rx value
            else:
                self.vec[0] -= self.size  # Decrease x value
        if Key.right in self.keys_pressed:
            if shift_pressed:
                self.vec[3] += self.rsize  # Increase rx value
            else:
                self.vec[0] += self.size  # Increase x value

        if any(hasattr(k, "char") and k.char == "w" for k in self.keys_pressed):
            self.vec[2] += self.size  # Increase z value
        if any(hasattr(k, "char") and k.char == "s" for k in self.keys_pressed):
            self.vec[2] -= self.size  # Decrease z value
        if any(hasattr(k, "char") and k.char == "e" for k in self.keys_pressed):
            self.vec[5] += self.rsize  # Increase rz value
        if any(hasattr(k, "char") and k.char == "d" for k in self.keys_pressed):
            self.vec[5] -= self.rsize  # Decrease rz value

        if any(hasattr(k, "char") and k.char == "m" for k in self.keys_pressed):
            self.vec[6] += self.gsize  # Increase gripper value (open)
        if any(hasattr(k, "char") and k.char == "n" for k in self.keys_pressed):
            self.vec[6] -= self.gsize  # Decrease gripper value (close)

        self.vec = np.array(self.vec)

    def run(self):
        print("running")
        thread = threading.Thread(target=self.listener.start, daemon=True)
        thread.start()
        """
        try:
            while True:
                # print(self.vec)  # Continuously print the vector to see the changes
                time.sleep(0.1)  # Print every 0.1 second
        except KeyboardInterrupt:
            self.running = False
            self.listener.stop()
        """

    def close(self):
        self.running = False
        self.listener.stop()

    def __call__(self, *args, **kwargs):
        return self.vec

    def register(self, key, func):
        self.funcs[key] = func


# kb = KeyboardController()
# while True:
# time.sleep(0.1)


class ScriptedController(Controller):
    def __init__(self):
        self.action = None

    def __call__(self, obs):
        return self.action

    def update(self, obs, reward, truncated, terminated, info):
        self.action = 1

class LiftController():
    def __init__(self, lift_env):
        self.env = lift_env
        
    def is_above_block(self):
        return self.env.block[2] <= self.env.eef[2]

    def is_block_grabbed(self):
        return self.env.block[2] <= self.env.eef[2]

    def is_gripper_closed(self):
        return self.env.gripper_angle < 0.1  # adjust based on your robot's gripper range

    def has_lifted(self):
        return self.env.obj[2] >= self.env.z_target

    
    def position_above_block(self):
        rel_pos = self.env.block - self.env.eef
        rel_pos[2] += 0.10  # 10cm above the block
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 0  # keep gripper open
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward

        

    def grab_block(self, max_steps=500):
        obs, reward = None, 0.0
        steps = 0
        grip = 0 # keep gripper open

        while steps < max_steps:
            z_dist = self.env.eef[2] - self.env.block[2]

            # Stop just slightly above the block
            if z_dist <= 0.01:
                break

            # Keep x and y fixed on top of the block, only move z toward the block
            rel_pos = self.env.block - self.env.eef
            rel_pos[0] = 0.0  # no x motion
            rel_pos[1] = 0.0  # no y motion
            rel_pos[2] -= 0.05  # small downward step toward the block
            rel_pos = np.clip(rel_pos, -0.05, 0.05)  # safe step size

            action = np.concatenate([rel_pos, [grip]]).astype(np.float32)
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            obs, reward, done, truncated, info = self.env.step(action)
            steps += 1

        # Close the gripper once close enough
        #grip = -1.0
        #action = np.concatenate([np.zeros(3), [grip]]).astype(np.float32)
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        #obs, reward, done, truncated, info = self.step(action)
        action = np.concatenate([np.zeros(3), [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.env.step(action)

        return obs, reward


    def lift_eef(self):
        """Lifts the EEF straight up by 10 cm without changing x, y, or gripper."""
        dz = 0.10  # 10 cm lift
        grip = self.env._action[-1]  # maintain current gripper state
        action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward


    def close_gripper(self):
        grip = 1.0  # adjust if your robot uses a different value for 'closed'
        action = np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward

    def lift_block(self):
        rel_pos = self.env.block - self.env.eef
        rel_pos[2] += 0.10  # 10cm above the block
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 1.0# keep gripper closed
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward

    def gentle_lift(self):
        """Lifts the EEF slowly by 5 cm total in small increments to stabilize the grasp."""
        dz = 0.005  # small upward step (5mm)
        grip = 1.0  # keep gripper closed

        for _ in range(10):  # 10 steps * 5mm = 5cm total lift
            action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
            obs, reward, done, truncated, info = self.env.step(action)
        
        return obs, reward

    def maintain_grip_and_lift(self):
        dz = 0.005  # small upward motion
        grip = 0.550  # hold current gripper position

        for _ in range(10):
            action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
            obs, reward, done, truncated, info = self.env.step(action)
        
        return obs, reward

    def move_to_bowl(self):
        rel_pos = self.env.obj - self.env.eef
        rel_pos[2] += 0.10  # 10cm above the bowl
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 0.55  # keep gripper open
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward

    def drop_in_bowl(self):
        grip = 0  #release gripper
        action = np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward




#def main():
    #sm = SpaceMouseController()
    #while True:
    #    print(sm.read().round(4))
    #    time.sleep(0.1)


#if __name__ == "__main__":
    #main()
