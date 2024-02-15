import numpy as np
from casadi import *
import do_mpc

def mpc(model, u_lim, N_horizon, t_step):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,
        'n_robust': 0,
        'open_loop': 0,
        't_step': t_step,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': False,
        'nlpsol_opts': {"ipopt":{"max_iter":1000, "print_level": 0}}
    }

    mpc.set_param(**setup_mpc)

    lterm = 2.0*model.aux['running_cost']
    mterm = model.aux['terminal_cost']

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=np.array([10.0]))

    mpc.bounds['lower', '_u', 'u'] = -np.array([[u_lim], [u_lim]])
    mpc.bounds['upper', '_u', 'u'] = np.array([[u_lim], [u_lim]])

    mpc.bounds['lower', '_x', 'x'] = np.array([[-1200], [-100.0], [-1200], [-100.0]])
    mpc.bounds['upper', '_x', 'x'] = np.array([[1200], [100.0], [1200], [100.0]])


    mpc.setup()
    return mpc