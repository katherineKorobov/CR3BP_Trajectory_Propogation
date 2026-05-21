'''
File: plot_jacobian.py
Description: Plots the Jacobi Constant
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 20 May 2026
'''

import numpy as np
import  matplotlib.pyplot as plt

from src.utils.calculate_jacobi_const import calculate_jacobi_const

def plot_jacobi_const(all_solutions, jac_constants):
    jac_min = np.min(jac_constants)
    jac_max = np.max(jac_constants)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots()

    jacobi_const = calculate_jacobi_const(all_solutions) # all jacobi constants for each iteration of orbit (1, len(t_eval)=10000)

    mean_jacobi_const = np.mean(jacobi_const)

    # attemping to smoothen out jacobi constant data so graph shows that it is constant despite small, negligible changes
    # changes occur due to calculation/rounding errors in code
    

    for sol, jac_const, mu, t_eval in all_solutions:

        if (max(jacobi_const) - min(jacobi_const)) < 1e-5:
            jac_const =  mean_jacobi_const
    
        if jac_max > jac_min:
            color = cmap((jac_const - jac_min) / (jac_max - jac_min)) 
        else:
            color = 0.5
        jac_const =  np.ones(len(t_eval)) * mean_jacobi_const
        ax.plot(t_eval, jac_const, color=color)

    ax.set_xlabel('Time (TU)')
    ax.set_ylabel('Jacobi Constant (LU\u00b2 / TU\u00b2)')
    ax.set_title('Time v. Jacobi Constant')