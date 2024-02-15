import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Voronoi, voronoi_plot_2d


def func(num, dataSet, dotsTraj, headingTraj):
    dotsTraj.set_data([dataSet[num][0], dataSet[num][2]])
    return dotsTraj


def visualiza_traj(traj, radar_config, voronoi_diagram, path, gif_name='bc.gif', save=False):
    fig = voronoi_plot_2d(voronoi_diagram)
    for i in range(len(np.array(path))-1):
        if i == 0:
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'bo-', label="Shortest Path")
        else:
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'bo-')

    numDataPoints = len(traj)
    # GET SOME MATPLOTLIB OBJECTS
    dotsTraj = plt.plot(traj[0][0], traj[0][2], 'go', label="Agent")[0] # For scatter plot
    x=np.linspace(0, 1000, numDataPoints)
    radar = plt.plot(radar_config[:, 0], radar_config[:, 1], 'ro', label="Radar")[0]

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(traj, dotsTraj, None), interval=10)
    plt.legend()
    if save:
        line_ani.save(gif_name)
    # plt.legend()
    plt.show()
