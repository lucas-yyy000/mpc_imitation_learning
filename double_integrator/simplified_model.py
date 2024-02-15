import numpy as np
from casadi import *
import do_mpc


def model(x_f):
    model_type = "continuous"
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(4, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))

    model.set_rhs('x', vertcat(x[1], u[0], x[3], u[1]))

    model.set_expression('running_cost',  (x[1]**2 + x[3]**2))
    model.set_expression('terminal_cost', ((x[0] - x_f[0])**2 + (x[2] - x_f[1])**2))

    model.setup()

    return model
