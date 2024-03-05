# Imitation Learning for compressing MPC.
This repo contains an implementation of using imitation learning to copy the behavior of an mpc. \
To make the implementation easier (bypass version conflicts and etc), we borrowed several utility functions from:\
[Stable Baseline3](https://github.com/DLR-RM/stable-baselines3)
and [Tao Chen's repo](https://github.com/taochenshh/easyrl) 

## Installation ##
After creating a python virtual environment with python 3.8, run `python setup.py install` inside the cloned folder [Tao Chen's repo](https://github.com/taochenshh/easyrl) Note that it might generate errors `ERROR: Failed building wheel for gputil`, that's okay for us.
Then you can work through the example in [bc_trainer](https://github.com/lucas-yyy000/mpc_imitation_learning/blob/main/bc_trainer.ipynb) to get an idea of the workflow.


The file [double_integrator](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/double_integrator) includes the expert mpc, and some sample data were collected and stored in [data_multimodal](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/double_integrator/data_multimodal).

To start, the main file used for behavior cloning is the [bc_trainer](https://github.com/lucas-yyy000/mpc_imitation_learning/blob/main/bc_trainer.ipynb) file, which parses the collected data and trains a student policy based on the processed data. After installing the requirements listed at the begging of the notebook, users should be able to run the code in the notebook to generate a student policy as well as to test it.

The [radar_maps](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/radar_maps) folder contains a gym environment that simulates the tracking problem.



