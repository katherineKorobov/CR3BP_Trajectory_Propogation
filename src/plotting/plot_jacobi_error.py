'''
File: plot_jacobian.py
Description: Plots the Jacobi Constant
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 20 May 2026
'''

import matplotlib.pyplot as plt
import numpy as np

from src.utils.calculate_jacobi_error import calculate_jacobi_error

def plot_jacobi_error(all_states, all_solutions, real_jacobi_consts):

    # first calculate the jacobi error
    jacobi_error = calculate_jacobi_error(all_states, all_solutions, real_jacobi_consts)

    fig, ax = plt.subplots()

    for t_eval in all_solutions:
        ax.plot(t_eval, jacobi_error)