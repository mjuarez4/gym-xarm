import numpy as np
from gym_xarm.tasks import Base


class Lift(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 50,
        "description": "Lift a cube above a height threshold",
    }

    def __init__(self, **kwargs):
        self._z_threshold = 0.15
        super().__init__("lift", **kwargs)

    @property
    def z_target(self):
        return self._init_z + self._z_threshold

    def is_success(self):
        return self.obj[2] >= self.z_target
    """
    def is_above_block(self):
        return self.block[2] <= self.eef[2]

    def is_block_grabbed(self):
        return self.block[2] <= self.eef[2]

    def is_gripper_closed(self):
        return self.gripper_angle < 0.1  # adjust based on your robot's gripper range

    def has_lifted(self):
        return self.obj[2] >= self.z_target

    
    def position_above_block(self):
        rel_pos = self.block - self.eef
        rel_pos[2] += 0.10  # 10cm above the block
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 0  # keep gripper open
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward

        

    def grab_block(self, max_steps=500):
        obs, reward = None, 0.0
        steps = 0
        grip = 0 # keep gripper open

        while steps < max_steps:
            z_dist = self.eef[2] - self.block[2]

            # Stop just slightly above the block
            if z_dist <= 0.01:
                break

            # Keep x and y fixed on top of the block, only move z toward the block
            rel_pos = self.block - self.eef
            rel_pos[0] = 0.0  # no x motion
            rel_pos[1] = 0.0  # no y motion
            rel_pos[2] -= 0.05  # small downward step toward the block
            rel_pos = np.clip(rel_pos, -0.05, 0.05)  # safe step size

            action = np.concatenate([rel_pos, [grip]]).astype(np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            obs, reward, done, truncated, info = self.step(action)
            steps += 1

        # Close the gripper once close enough
        #grip = -1.0
        #action = np.concatenate([np.zeros(3), [grip]]).astype(np.float32)
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        #obs, reward, done, truncated, info = self.step(action)
        action = np.concatenate([np.zeros(3), [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.step(action)

        return obs, reward


    def lift_eef(self):
        
        dz = 0.10  # 10 cm lift
        grip = self._action[-1]  # maintain current gripper state
        action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward


    def close_gripper(self):
        grip = 1.0  # adjust if your robot uses a different value for 'closed'
        action = np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward

    def lift_block(self):
        rel_pos = self.block - self.eef
        rel_pos[2] += 0.10  # 10cm above the block
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 1.0# keep gripper closed
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward

    def gentle_lift(self):
        #Lifts the EEF slowly by 5 cm total in small increments to stabilize the grasp.
        dz = 0.005  # small upward step (5mm)
        grip = 1.0  # keep gripper closed

        for _ in range(10):  # 10 steps * 5mm = 5cm total lift
            action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
            obs, reward, done, truncated, info = self.step(action)
        
        return obs, reward

    def maintain_grip_and_lift(self):
        dz = 0.005  # small upward motion
        grip = 0.550  # hold current gripper position

        for _ in range(10):
            action = np.array([0.0, 0.0, dz, grip], dtype=np.float32)
            obs, reward, done, truncated, info = self.step(action)
        
        return obs, reward

    def move_to_bowl(self):
        rel_pos = self.obj - self.eef
        rel_pos[2] += 0.10  # 10cm above the bowl
        action_xyz = np.clip(rel_pos, -0.2, 0.2)  # tighter clip to improve precision
        grip = 0.55  # keep gripper open
        action = np.concatenate([action_xyz, [grip]]).astype(np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward

    def drop_in_bowl(self):
        grip = 0  #release gripper
        action = np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)
        obs, reward, done, truncated, info = self.step(action)
        return obs, reward

    def execute(self):
        self.position_above_block()
        self.grab_block()
        #self.position_above_block()

"""





    def get_reward(self):
        reach_dist = np.linalg.norm(self.obj - self.eef)
        reach_dist_xy = np.linalg.norm(self.obj[:-1] - self.eef[:-1])
        pick_completed = self.obj[2] >= (self.z_target - 0.01)
        obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.02)

        # Reach
        if reach_dist < 0.05:
            reach_reward = -reach_dist + max(self._action[-1], 0) / 50
        elif reach_dist_xy < 0.05:
            reach_reward = -reach_dist
        else:
            z_bonus = np.linalg.norm(np.linalg.norm(self.obj[-1] - self.eef[-1]))
            reach_reward = -reach_dist - 2 * z_bonus

        # Pick
        if pick_completed and not obj_dropped:
            pick_reward = self.z_target
        elif (reach_dist < 0.1) and (self.obj[2] > (self._init_z + 0.005)):
            pick_reward = min(self.z_target, self.obj[2])
        else:
            pick_reward = 0

        return reach_reward / 100 + pick_reward

    def _get_obs(self):
        return np.concatenate(
            [
                self.eef,
                self.eef_velp,
                self.obj,
                self.obj_rot,
                self.obj_velp,
                self.obj_velr,
                self.eef - self.obj,
                np.array(
                    [
                        np.linalg.norm(self.eef - self.obj),
                        np.linalg.norm(self.eef[:-1] - self.obj[:-1]),
                        self.z_target,
                        self.z_target - self.obj[-1],
                        self.z_target - self.eef[-1],
                    ]
                ),
                self.gripper_angle,
            ],
            axis=0,
        )

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = self.center_of_table - np.array([0.15, 0.10, 0.07])
        object_pos[0] += self.np_random.uniform(-0.05, 0.05)
        object_pos[1] += self.np_random.uniform(-0.05, 0.05)
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)
        self._init_z = object_pos[2]

        # Goal
        return object_pos + np.array([0, 0, self._z_threshold])

    def reset(
        self,
        seed=None,
        options: dict | None = None,
    ):
        self._action = np.zeros(4)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._action = action.copy()
        #print(f"[DEBUG] Action sent: {action}, Gripper angle: {self.gripper_angle}")
        return super().step(action)
