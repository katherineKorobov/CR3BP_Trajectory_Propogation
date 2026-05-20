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

    jacobian_const = calculate_jacobi_const(all_solutions)

    for sol, jac_const, mu, t_eval in all_solutions:
        color = cmap((jac_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5)
        ax.plot(t_eval, jacobian_const, color=color)

    ax.set_xlabel('Time (TU)')
    ax.set_ylabel('Jacobi Constant (LU\u00b2 / TU\u00b2)')
    ax.set_title('Time v. Jacobi Constant')