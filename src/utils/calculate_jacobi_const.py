'''
File: calculate_jacobi_const.py
Description: 
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 20 May 2026
'''

import numpy as np

def calculate_jacobi_const(all_solutions):
    # calculates the jacobi constant of the orbit for points along the orbit
    # jacobi constant is a conserved quantity in CR3BP, so we should see no change in the value for each orbit
    num_cols = np.size(all_solutions, axis=1)
    jacobian_const = np.zeros(num_cols)
    
    for sol, jac_const, mu, t_eval in all_solutions:
        x = sol[:, 0]
        y = sol[:, 1]
        z = sol[:, 2]

        x_dot = sol[:, 3]
        y_dot = sol[:, 4]
        z_dot = sol[:, 5]

        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

        u_star = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2
        v_2 = x_dot**2 + y_dot**2 + z_dot**2
        temp_jacobian_const = 2 * u_star - v_2
        jacobian_const = np.vstack((jacobian_const, temp_jacobian_const)) # adds the next value of calculated jacobians underneath for no rewriting

    return jacobian_const