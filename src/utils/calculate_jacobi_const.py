'''
File: main.py
Description: The main function for the project: "CR3BP_Trajectory Propogation".
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 20 May 2026
'''

import numpy as np

def calculate_jacobi_const(all_solutions):

    for sol, jac_const, mu, t_eval in all_solutions:
        x = sol[:, 0]
        y = sol[:, 1]
        z = sol[:, 2]

        x_dot = sol[:, 3]
        y_dot = sol[:, 4]
        z_dot = sol[:, 5]

        if min(z) < 1e-10 and min(z_dot) < 1e-10:
            z = 0
            z_dot = 0

        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

        u_star = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2
        v_2 = x_dot**2 + y_dot**2 + z_dot**2
        jacobian_const = 2 * u_star - v_2

    return jacobian_const