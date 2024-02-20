# Imitation Learning for compressing MPC.
This repo contains an implementation of using imitation learning to copy the behavior of an mpc. \
To make the implementation easier (bypass version conflicts and etc), we borrowed several utility functions from:\
[Stable Baseline3](https://github.com/DLR-RM/stable-baselines3)
and [Tao Chen's repo](https://github.com/taochenshh/easyrl) 

To start, [double_integrator](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/double_integrator) includes the expert mpc, and some sample data were collected and stored in [data_multimodal](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/double_integrator/data_multimodal).
The main file used for behavior cloning is the [bc_trainer](https://github.com/lucas-yyy000/mpc_imitation_learning/blob/main/bc_trainer.ipynb) file, which parses the collected data and trains a student policy based on the processed data.

The [radar_maps](https://github.com/lucas-yyy000/mpc_imitation_learning/tree/main/radar_maps) folder contains a gym environment that simulates the tracking problem.



