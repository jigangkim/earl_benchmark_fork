""""Kitchen environment."""

# Add Kitchen assets adept_envs/ folder to the python path.
import sys
import os
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(parent_dir, "kitchen_assets/adept_envs"))

import time
import numpy as np
import copy
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

ELEMENT_INDICES_LL = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Arm
    [9, 10],  # Burners
    [11, 12],  # Burners
    [13, 14],  # Burners
    [15, 16],  # Burners
    [17, 18],  # Lightswitch
    [19],  # Slide
    [20, 21],  # Hinge
    [22] # Microwave
]

# [23, 24, 25, 26, 27, 28, 29]  # Kettle
initial_states = np.array([[
    -0.56617326,
    -1.6541005,
    1.4447045,
    -2.4378936,
    0.71086496,
    1.3657048,
    0.80830157,
    0.019943988,
    0.019964991,
    2.456005e-05,
    2.9547007e-07,
    2.4559975e-05,
    2.954692e-07,
    2.4559975e-05,
    2.954692e-07,
    2.4559975e-05,
    2.954692e-07,
    2.161876e-05,
    5.0806757e-06,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.269,
    0.35,
    1.6192839,
    1.0,
    -8.145112e-19,
    -1.1252103e-05,
    -2.8055027e-19,
]])

supported_tasks = ['open_microwave', 'bottom_burner', 'top_burner', 'light_switch', 'slide_cabinet', 'hinge_cabinet']

goal_list = {}
goal_states = {}
goal_indices = {}

def set_goal_list(task_name, initial_states=initial_states, goal_states=goal_states):
    goal_list[task_name] = initial_states[0].copy()
    start_idx = goal_indices[task_name][0]
    end_idx = goal_indices[task_name][1]
    goal_list[task_name][start_idx:end_idx] = goal_states[task_name]

goal_indices['open_microwave'] = [22, 23]
goal_states['open_microwave'] = [-0.7]

# Goal states from https://github.com/rail-berkeley/d4rl/blob/master/d4rl/kitchen/kitchen_envs.py
goal_indices['bottom_burner'] = [11, 13]
goal_states['bottom_burner'] = [-0.88, -0.01]

goal_indices['top_burner'] = [15, 17]
goal_states['top_burner'] = [-0.92, -0.01]

goal_indices['light_switch'] = [17, 19]
goal_states['light_switch'] = [-0.69, -0.05]

goal_indices['slide_cabinet'] = [19, 20]
goal_states['slide_cabinet'] = [0.37]

goal_indices['hinge_cabinet'] = [20, 22]
goal_states['hinge_cabinet'] = [0., 1.45]

set_goal_list("open_microwave")
set_goal_list("bottom_burner")
set_goal_list("top_burner")
set_goal_list("light_switch")
set_goal_list("slide_cabinet")
set_goal_list("hinge_cabinet")

class Kitchen(KitchenTaskRelaxV1):

  def __init__(self, task="open_microwave", reverse=False, reward_type="dense"):
    self._initial_states = copy.deepcopy(initial_states)
    self._goal_states = copy.deepcopy(goal_states)
    self.total_time = 0
    self.num_steps = 0
    if reward_type != 'dense':
        raise ValueError("Kitchen environment only supports dense rewards.")
    if not task in supported_tasks:
        raise ValueError("Error: Kitchen environment does not support the given task.")
    if reverse:
        goal_start_idx = goal_indices[task][0]
        goal_end_idx = goal_indices[task][1]
        # Swap initial and goal states
        new_goal = copy.deepcopy(self._initial_states[0][goal_start_idx:goal_end_idx])
        self._initial_states[0][goal_start_idx:goal_end_idx] = copy.deepcopy(self._goal_states[task])
        self._goal_states[task] = new_goal
        set_goal_list(task, self._initial_states, self._goal_states)

    self._reward_type = reward_type
    self._task = task
    super().__init__()

  def get_task(self):
    return self._task

  def _get_obs(self):
    ob = super()._get_obs()
    return ob[:23]

  def get_init_states(self):
    return self._initial_states

  def get_next_goal(self):
    return goal_list[self._task]

  def _task_reward(self, task_name, joint1, joint2=None):
    reward_dict  = {}
    reward_dict['true_reward'] = -np.linalg.norm(self.sim.named.data.qpos[joint1][0] - self._goal_states[task_name][0])
    if joint2 != None:
        reward_dict['true_reward'] += -np.linalg.norm(self.sim.named.data.qpos[joint2][0] - self._goal_states[task_name][1])
    reward_dict['r_total'] = reward_dict['true_reward']
    return reward_dict
        
  def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        if self._task == 'open_microwave':
        # Write rewards for each task
            microwave_door_handle_goal = -0.7
            reward_dict['true_reward'] = -np.linalg.norm(self.sim.named.data.qpos['microjoint'][0] - microwave_door_handle_goal)
            reward_dict['bonus'] = -np.linalg.norm(self.sim.data.mocap_pos[0]-self.sim.named.data.site_xpos['microhandle_site'])
            reward_dict['r_total'] = 10 * reward_dict['true_reward'] + reward_dict['bonus']
        elif self._task == 'bottom_burner':
            reward_dict = self._task_reward("bottom_burner", "knob_Joint_2", "burner_Joint_2")
        elif self._task == 'top_burner':
            reward_dict = self._task_reward("top_burner", "knob_Joint_4", "burner_Joint_4")
        elif self._task == 'slide_cabinet':
            reward_dict = self._task_reward("slide_cabinet", "slidedoor_joint")
        elif self._task == 'hinge_cabinet':
            reward_dict = self._task_reward("hinge_cabinet", "leftdoorhinge", "rightdoorhinge")
        elif self._task == 'light_switch':
            reward_dict = self._task_reward("light_switch", "lightswitch_joint", "light_joint")
        else:
            raise Exception("Error: Task not implemented.")
        score = 0.
        return reward_dict, score

  def compute_reward(self, obs):
    return self._get_reward_n_score(obs)['r_total']

  def is_successful(self, obs=None):
    return False

  def step(self, a, b=None):
    start = time.time()
    obs, reward, done, info = super().step(a, b)
    end = time.time()
    self.total_time += start - end
    self.num_steps += 1
    # if self.num_steps % 1000 == 0:
    #    print("Average step time: ", self.total_time/self.num_steps)
    return obs, reward, done, info

  def _reset(self):
    self.set_goal(goal=self.get_next_goal())
    return super()._reset()

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()
    self.set_goal(goal)
