import numpy as np
from casadi import *
import do_mpc

def simulator(model, t_step):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = t_step)
    simulator.setup()

    return simulator