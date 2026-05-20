
import numpy as np

def modelEOM(state, t, mu):
    '''
    @brief Defines Equations of Motion for CR3BP and in odeint to integrate orrbit trajectories
    @param state: the current (soon to be previous) state list
    @param t: The time (TU) thtat corresponds to state
    @param mu: The mass ratio between the primary and secondary body
    @return state_dot: the derivative of the state list

    Note: odeint takes in a derivative state  list, integrates, and outputs the state trajectory
    Note: t is not used because the EOM are time-independent. It is used to satisfy odeint which looks for the time input in the function call.
    '''
    x, y, z, x_dot, y_dot, z_dot = state 

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_ddot = 2 * y_dot + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_ddot = -2 * x_dot + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_ddot = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    state_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

    return state_dot