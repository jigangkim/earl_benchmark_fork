import os

import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from scipy.spatial.transform import Rotation

import mujoco_py

initial_states = np.array([[0.00615235, 0.6001898, 0.19430117]])
goal_states = np.random.uniform(low=(-0.1, 0.8, 0.1), high=(0.1, 0.9, 0.35), size=(100,3)) # 100 goal states

class SawyerReachV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    max_path_length = int(1e8)
    TARGET_RADIUS = 0.05

    def __init__(self, reward_type='dense', reset_at_goal=False):
        goal_low = (-0.1, 0.8, 0.1)
        goal_high = (0.1, 0.9, 0.35)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0., 0.6, 0.02]),
            'hand_init_pos': np.array([0., 0.6, 0.2]),
        }

        self.initial_states = initial_states
        self.goal_states = goal_states

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._partially_observable = False
        self._set_task_called = True
        self._freeze_rand_vec = False

        self._reset_at_goal = reset_at_goal
        self._reward_type = reward_type

        self._random_reset_space = Box(
            np.hstack((goal_low)),
            np.hstack((goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    @property
    def model_name(self):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "metaworld_assets/sawyer_xyz", 'sawyer_reach_v2.xml')

    @property
    def observation_space(self):
        goal_low = self.goal_space.low
        goal_high = self.goal_space.high
        return Box(
            np.hstack((self._HAND_SPACE.low, goal_low)),
            np.hstack((self._HAND_SPACE.high, goal_high))
        )

    def _get_obs(self):
        obs = super()._get_obs()
        # xyz for end effector
        endeff_config = obs[:3]
        obs = np.concatenate([endeff_config, self.goal,])
        return obs

    def get_next_goal(self):
        num_goals = self.goal_states.shape[0]
        goal_idx = np.random.randint(0, num_goals)
        return self.goal_states[goal_idx]

    def reset_goal(self, goal=None):
        if goal is None:
            goal = self.get_next_goal()
        
        self.goal = goal
        self._target_pos = goal.copy()

        self.sim.model.site_pos[self.model.site_name2id('goal')] = goal

    @_assert_task_is_set
    def evaluate_state(self, obs, action):

        reward, reach_dist, in_place = self.compute_reward(obs, action)
        success = float(reach_dist <= self.TARGET_RADIUS)

        info = {
            'success': success,
            'near_object': reach_dist,
            'grasp_success': 1.,
            'grasp_reward': reach_dist,
            'in_place_reward': in_place,
            'obj_to_target': reach_dist,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def reset_model(self):
        self.reset_goal()
        if self._reset_at_goal:
            self.hand_init_pos = self._target_pos.copy() + np.random.uniform(-self.TARGET_RADIUS/5, self.TARGET_RADIUS/5, size=3)
            self._reset_hand()
            while not self.is_successful():
                self.hand_init_pos = self._target_pos.copy() + np.random.uniform(-self.TARGET_RADIUS/5, self.TARGET_RADIUS/5, size=3)
                self._reset_hand()
        else:
            self._reset_hand()

        return self._get_obs()

    def compute_reward(self, obs, actions=None, vectorized=True):
        obs = np.atleast_2d(obs)

        tcp = obs[:,:3]
        target = obs[:,3:]

        tcp_to_target = np.linalg.norm(tcp - target, axis=-1)

        in_place_margin = (np.linalg.norm(self.hand_init_pos - target, axis=-1))
        in_place = reward_utils.tolerance(tcp_to_target,
                                    bounds=(0, self.TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        if self._reward_type == 'dense':
            reward = 10 * in_place
        elif self._reward_type == 'sparse':
            reward = np.array(self.is_successful(obs=obs), dtype=np.float32)

        return [np.squeeze(reward), np.squeeze(tcp_to_target), np.squeeze(in_place)]

    def is_successful(self, obs=None, vectorized=True):
        return self.dist_to_goal(obs=obs, vectorized=vectorized) <= self.TARGET_RADIUS

    def dist_to_goal(self, obs=None, vectorized=True):
        if obs is None:
            obs = self._get_obs()

        return np.linalg.norm(obs[:,:3] - obs[:,3:], axis=-1) if obs.ndim == 2 else np.linalg.norm(obs[:3] - obs[3:])

    def viewer_setup(self):
        self.viewer.cam.distance = 1.6
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = -45

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            if 'rgb_array' in mode:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self._viewers[mode] = self.viewer

        self.viewer_setup()
        return self.viewer
  
    def render(self, mode='human', height=480, width=640):
        if mode == 'human':
            self._get_viewer(mode).render()
        elif mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")

    def close(self):
        if self.viewer is not None:
            if isinstance(self.viewer, mujoco_py.MjViewer):
                glfw.destroy_window(self.viewer.window)
            self.viewer = None