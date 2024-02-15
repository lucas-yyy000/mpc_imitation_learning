import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import simplified_model, simplified_model_mpc, simplified_model_simulator 

import sys
sys.path.append('../imitation_learning')
import planning_on_voronoi

import pickle
def update_path_index(path_vertices, cur_path_index, cur_loc):
    if (cur_loc[0] - path_vertices[cur_path_index][0])**2 + (cur_loc[2] - path_vertices[cur_path_index][1])**2 < 5**2:
        return cur_path_index + 1
    return cur_path_index

def generate_reference_path_from_radar_config(radar_config, size_of_map, x_f, add_noise=False, noise_level=30.0, min_dist=10.0, visualize=False):
    '''
    Prevent the original radar_config from being modified.
    '''
    radar_config_noised = np.zeros(radar_config.shape)
    if add_noise:
        for i in range(len(radar_config_noised)):
            radar_config_noised[i, :] = radar_config[i, :] + noise_level*np.random.multivariate_normal([0.0, 0.0], np.eye(2))

    radar_locs, voronoi_diagram, path = planning_on_voronoi.get_baseline_path_with_vertices(radar_config_noised, size_of_map)
    radar_locs = radar_locs[:-2]

    path_vertices = voronoi_diagram.vertices[path]

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
    
    if visualize:
        '''
        Plot the Voronoi diagram with the shortest path.
        '''
        fig = voronoi_plot_2d(voronoi_diagram)
        plt.plot(voronoi_diagram.vertices[path][:, 0], voronoi_diagram.vertices[path][:, 1], 'ro')
        for i in range(len(np.array(path))-1):
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'ro-')
        plt.show()

    path_ref.append(x_f)

    return path_ref

def generate_reference_path_and_random_radar_config(size_of_map, x_f, dist_between_radars, num_radar, min_dist=30.0, visualize=True):
    radar_locs, voronoi_diagram, path = planning_on_voronoi.get_baseline_path(size_of_map, dist_between_radars, num_radar)
    radar_locs = radar_locs[:-2]

    path_vertices = voronoi_diagram.vertices[path]

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
    
    if visualize:
        '''
        Plot the Voronoi diagram with the shortest path.
        '''
        fig = voronoi_plot_2d(voronoi_diagram)
        plt.plot(voronoi_diagram.vertices[path][:, 0], voronoi_diagram.vertices[path][:, 1], 'ro')
        for i in range(len(np.array(path))-1):
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'ro-')
        plt.show()

    path_ref.append(x_f)

    return radar_locs, path_ref, voronoi_diagram, path

def run_sim(iter):
    '''
    Parameters
    '''
    v = 20.0           # Speed of the agent. 
    z = 100.0              # Z coordinate of the agent.

    T_horizon = 20.0        # Planning horizon (s).
    N_horizon = 40          # Planning time steps.
    t_step = T_horizon/N_horizon    # Duration of each time step.
    u_lim = 2.0                     # Input limit.

    num_modes = 3 # Number of modes.
    noise_level = 30.0

    size_of_map = 1000          
    x_f = np.array([size_of_map, size_of_map])
    x_0 = np.array([100.0*np.random.rand(1)[0], 0, 100.0*np.random.rand(1)[0], 0])

    '''
    Get radar locations.
    '''
    # radar_config = np.load("radar_config.npy")

    '''
    Generate reference path.
    '''
    # path_ref, voronoi_diagram = generate_reference_path_from_radar_config(radar_config, size_of_map, x_f, visualize=True)
    radar_locs, path_ref, voronoi_diagram, path_indices = generate_reference_path_and_random_radar_config(size_of_map, x_f, dist_between_radars=size_of_map/5.0, num_radar=10, visualize=False)


    '''
    Data to be collected.
    '''
    trajectory = []
    target_traj = []
    control_traj = []
    trajectory_multimodal = []
    optimal_path_multimodal = []
    path_multimodal = []

    '''
    Keep track of next waypoint.
    '''
    cur_path_index = 0

    '''
    Loop index.
    '''
    loop_index = 0

    # time_begin = time.process_time()
    while np.linalg.norm(np.array([x_0[0], x_0[2]]) - x_f) > 50:
        loop_index += 1
        if loop_index > 500:
            break
        
        prev_path_index = cur_path_index
        cur_path_index = update_path_index(path_vertices=path_ref, cur_path_index=cur_path_index, cur_loc=x_0)
        if prev_path_index != cur_path_index:
            if cur_path_index >= len(path_ref):
                break


        x_next_waypoint = path_ref[cur_path_index]
        dist_to_next_waypoint = np.linalg.norm(np.array([x_0[0], x_0[2]])- x_next_waypoint)
        
        if dist_to_next_waypoint < T_horizon*v:
            # print("NEXT TARGET WITHIN REACH")
            horizon_tmp = np.ceil(dist_to_next_waypoint/v)
            if horizon_tmp <= 1:
                cur_path_index += 1
                if cur_path_index >= len(path_ref):
                    break
                continue
            # print("N_horizon: ", np.ceil((dist_to_next_waypoint/v)/t_step))
            model = simplified_model.model(x_next_waypoint)
            mpc = simplified_model_mpc.mpc(model, u_lim, int(np.ceil((dist_to_next_waypoint/v)/t_step)), t_step)
            simulator = simplified_model_simulator.simulator(model, t_step)
        else:
            # print("NEXT TARGET OUT OF REACH")
            model = simplified_model.model(x_next_waypoint)
            mpc = simplified_model_mpc.mpc(model, u_lim, N_horizon, t_step)
            simulator = simplified_model_simulator.simulator(model, t_step)

        '''
        Generate control input.
        '''
        simulator.x0['x'] = x_0
        mpc.x0 = x_0

        u_init_baseline = np.full((2, 1), 0.0)
        mpc.u0 = u_init_baseline
        simulator.u0 = u_init_baseline
        mpc.set_initial_guess()

        u_next = mpc.make_step(x_0)

        '''
        Add noise to the control input to force the agent to explore more states.
        '''
        x_next = simulator.make_step(u_next+(u_lim/10.0)*np.random.normal(size=(2, 1)))

        '''
        Perturb radar locations to generate multi-modal behaviors.
        '''
        u_multi_mode = [u_next[0][0], u_next[1][0]]
        x_multi_mode = [np.array([x_next[0][0], x_next[1][0], x_next[2][0], x_next[3][0]])]
        path_ref_multi_mode = [path_ref[cur_path_index:]]
        for _ in range(num_modes-1):
            path_mm = generate_reference_path_from_radar_config(radar_locs, size_of_map, x_f, add_noise=True)
            index_of_next_waypoint = -1
            for i in range(len(path_mm)):
                if np.linalg.norm(np.array([x_0[0], x_0[2]]) - x_f) > np.linalg.norm(path_mm[i] - x_f):
                    index_of_next_waypoint = i
                    break
            if index_of_next_waypoint < 0:
                raise Exception("Invalid configuration.")
            
            model_mm = simplified_model.model(path_mm[i])
            mpc_mm = simplified_model_mpc.mpc(model_mm, u_lim, N_horizon, t_step)

            mpc.x0 = x_0
            u_init = np.full((2, 1), 0.0)
            mpc_mm.u0 = u_init
            mpc_mm.set_initial_guess()
            u_mm = mpc_mm.make_step(x_0)

            simulator.x0['x'] = x_0
            x_mm_next = simulator.make_step(u_mm+(u_lim/10.0)*np.random.normal(size=(2, 1)))

            u_multi_mode.extend([u_mm[0][0], u_mm[1][0]])
            x_multi_mode.append(np.array([x_mm_next[0][0], x_mm_next[1][0], x_mm_next[2][0], x_mm_next[3][0]]))
            path_ref_multi_mode.append(path_mm[index_of_next_waypoint:])
            # print(path_ref_multi_mode)

        trajectory.append(x_0)
        trajectory_multimodal.extend(x_multi_mode)
        control_traj.append(np.array(u_multi_mode))
        path_multimodal.extend(path_ref_multi_mode)
        # print(path_multimodal)
        x_0 = np.array([x_next[0][0], x_next[1][0], x_next[2][0], x_next[3][0]])

    # time_end = time.process_time()
    np.save(f'double_integrator/data_multimodal/state_traj_{iter}', np.array(trajectory))
    np.save(f'double_integrator/data_multimodal/state_multimodal_traj_{iter}', np.array(trajectory_multimodal))
    np.save(f'double_integrator/data_multimodal/control_traj_{iter}', np.array(control_traj))
    np.save(f'double_integrator/data_multimodal/radar_config_{iter}', radar_locs)
    np.save(f"double_integrator/data_multimodal/nominal_path_{iter}", np.array(path_ref))
    with open(f"double_integrator/data_multimodal/nominal_path_multimodal_{iter}.pkl", "wb") as f:
        pickle.dump(path_multimodal, f)

    # print("Reference Path: ", path_ref)
    # visualization.visualiza_traj(np.array(trajectory), radar_locs, voronoi_diagram, path_indices)



if __name__ == '__main__':
    data_num = 1
    process_time = []
    for i in range(data_num):
        run_sim(i)
        # process_time.append(ti)
    # np.save(f"expert_process_time", np.array(process_time))