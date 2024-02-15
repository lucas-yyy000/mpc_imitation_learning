# import gymnasium as gym
import gym
import numpy as np
from scipy.stats import bernoulli
import math
from gym import spaces, logger
import sys
import pwlf
import matplotlib.pyplot as plt

sys.path.append('../radar_avoidance')
import planning_on_voronoi

class RadarMap_DoubleIntegrator(gym.Env):
    def __init__(self, map_size, goal_location, radar_detection_range, grid_size, dist_between_radars, num_radars, time_step=0.5, v=20.0, u_lim=2.0):
        super().__init__()
        self.time_step = time_step
        self.map_size = map_size
        self.goal = goal_location


        self.radar_detection_range = radar_detection_range
        self.grid_size = grid_size

        # Example for using image as input (channel-first; channel-last also works):
        self.img_size = [1, 2*int(radar_detection_range/grid_size), 2*int(radar_detection_range/grid_size)]

        self.x_threshold = 1.2*map_size
        self.y_threshold = 1.2*map_size
        self.xdot_threshold = v
        self.ydot_threshold = v
        self.u_lim = u_lim

        '''
        Hardcoded for bc.
        '''
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        

        radar_locs, voronoi_diagram, path_idx = planning_on_voronoi.get_baseline_path(size=map_size, dist_between_radars=dist_between_radars, num_radar=num_radars)
        self.radar_locs = radar_locs



        self.dist_between_radars = dist_between_radars
        self.num_radars = num_radars


        '''
        EXTRACT optimal waypoint sequence.
        '''
        min_dist=30.0
        path_vertices = voronoi_diagram.vertices[path_idx]
        path_ref = []
        path_ref.append(path_vertices[0])
        path_ref_index_tmp = 0

        '''
        Remove clustered vertices.
        '''
        for i in range(len(path_vertices)):
            if np.linalg.norm(path_ref[path_ref_index_tmp] - path_vertices[i]) < min_dist:
                continue
            path_ref.append(path_vertices[i])
            path_ref_index_tmp += 1
        path_ref.append(self.goal)
        path_ref = np.array(path_ref)
        self.reference_path = path_ref
        self.reference_path_pwlf = pwlf.PiecewiseLinFit(path_ref[:, 0], path_ref[:, 1])
        self.reference_path_pwlf.fit_with_breaks(path_ref[:, 0])


        high = np.array(
            [
                self.x_threshold,
                self.xdot_threshold,
                self.y_threshold,
                self.ydot_threshold
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            spaces={
                "state": spaces.Box(-high, high, dtype=np.float64),
                "img": spaces.Box(0, 255, self.img_size, dtype=np.float32),
            }
        )

        self.steps_beyond_terminated = 0

        # self.reset()

    def get_radar_heat_map(self, state):
        '''
        Given the current state of the agent, generate a heat map indicating proximity of nearby radars.
        '''
        radar_grid_size = self.img_size[1]
        radars_encoding = np.zeros((radar_grid_size, radar_grid_size))
        theta = np.arctan2(state[3], state[1])
        loc_to_glob = np.array([[np.cos(theta), -np.sin(theta), state[0]],
                                [np.sin(theta), np.cos(theta), state[2]],
                                [0., 0., 1.]])
        glob_to_loc = np.linalg.inv(loc_to_glob)

        for radar_loc in self.radar_locs:
            if abs(state[0] - radar_loc[0]) < self.radar_detection_range or abs(state[2] - radar_loc[1]) < self.radar_detection_range:
                # print("Radar global: ", radar_loc)
                glob_loc_hom = np.array([radar_loc[0], radar_loc[1], 1])
                local_loc_hom = np.dot(glob_to_loc, glob_loc_hom)
                radars_loc_coord = local_loc_hom[:2]

                # print("Radars local coord: ", radars_loc_coord)
                y_grid = np.rint((radars_loc_coord[1]) / self.grid_size) 
                x_grid = np.rint((radars_loc_coord[0]) / self.grid_size) 

                for i in range(-int(radar_grid_size/2), int(radar_grid_size/2)):
                    for j in range(-int(radar_grid_size/2), int(radar_grid_size/2)):
                        radars_encoding[int(i + radar_grid_size/2), int(j + radar_grid_size/2)] += np.exp((-(x_grid - i)**2 - (y_grid - j)**2))*1e3

        radars_encoding = radars_encoding.T
        if np.max(radars_encoding) > 0:
            formatted = (radars_encoding * 255.0 / np.max(radars_encoding)).astype('float32')
        else:
            formatted = radars_encoding.astype('float32')

        formatted = formatted[np.newaxis, :, :]
        return formatted
    
    def step(self, action):
        x = self.state['state'][0]
        xdot = self.state['state'][1]
        y = self.state['state'][2]
        ydot = self.state['state'][3]

        x = x + self.time_step*xdot
        y = y + self.time_step*ydot
        xdot = xdot + self.u_lim*self.time_step*action[0]
        ydot = ydot + self.u_lim*self.time_step*action[1]

        
        # Saturation
        xdot = np.clip(xdot, a_min = -self.xdot_threshold, a_max=self.xdot_threshold)
        ydot = np.clip(ydot, a_min = -self.ydot_threshold, a_max=self.ydot_threshold)
        state_cur = [x, xdot, y, ydot]
        state_cur_normalized = [state_cur[0]/self.x_threshold, state_cur[1]/self.xdot_threshold, state_cur[2]/self.y_threshold, state_cur[3]/self.ydot_threshold]

        heat_map = self.get_radar_heat_map(state_cur)

        self.state = {
                "state": np.array(state_cur),
                "img": heat_map
            }

        state_normalized = {
                "state": np.array(state_cur_normalized),
                "img": heat_map
            }
        
        terminated, reward = self._update_state(state_cur)
        # print(self.state)

        return state_normalized, reward, terminated, False, {}

    def _update_state(self, state):
        '''
        Update state of the env.
        Return terminated, reward.
        '''
        x = state[0]
        xdot = state[1]
        y = state[2]
        ydot = state[3]


        dist_to_goal_squared = (x - self.goal[0])**2 + (y - self.goal[1])**2

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.y_threshold
            or y > self.y_threshold
            or dist_to_goal_squared < 50.0**2
         )
        
        if not terminated:
            dist_to_ref_path = np.abs(self.reference_path_pwlf.predict(x) - y)
            return False, -dist_to_ref_path - np.sqrt(dist_to_goal_squared)

        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            out_of_bound = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or y < -self.y_threshold
                or y > self.y_threshold
            )
            if out_of_bound:
                return True, -1e6
            else:
                return True, 1e8
    
    def _calculate_instantaneous_probability_of_detection(self, state, c_1=1.0, c_2=1.0):
        instant_prob = np.zeros(self.num_radars)
        for i in range(self.num_radars):
            R = ((state[0] - self.radar_locs[i][0])**2 + (state[2] - self.radar_locs[i][1])**2)
            sigma = 1.0
            instant_prob[i] = 1.0/(1.0 + (c_2*(R**4)/sigma)**(c_1))
        return instant_prob

    def seed(self, seed):
        self.reset(seed)

    def reset(self, seed=None, options=None, noise_level=0.0):
        super().reset(seed=seed)

        x = np.random.uniform(-0.1*self.map_size, 0.1*self.map_size)
        y = np.random.uniform(-0.1*self.map_size, 0.1*self.map_size)
        xdot = 0.
        ydot = 0.
        state = [x, xdot, y, ydot]
        state_normalized = [x/self.x_threshold, xdot/self.xdot_threshold, y/self.y_threshold, ydot/self.ydot_threshold]
        
        self.steps_beyond_terminated = 0

        formatted_heat_map = self.get_radar_heat_map(state)
        
        self.state = {
                "state": np.array(state),
                "img": formatted_heat_map
            }

        normalized_state = {
                "state": np.array(state_normalized),
                "img": formatted_heat_map
            }
        
        return normalized_state, {}

    def render(self):
        return

    def close(self):
        return 