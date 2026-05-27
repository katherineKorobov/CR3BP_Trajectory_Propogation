'''
File: calculate_jacobi_error.py
Description: Calculates the difference (the error) between the given Jacobi Constant and the calculated constant.
Author: Katherine Korobov
Created: 21 May 2026
Last Modified: 21 May 2026
'''

import numpy as np

from src.utils.calculate_jacobi_const import calculate_jacobi_const

def calculate_jacobi_error(all_states, all_solutions, real_jacobi_consts):
    # first step is to retrive the jacobi constant from the data
    
    calculated_jacobi_const = calculate_jacobi_const(all_solutions) 

    for i in real_jacobi_consts:
        real_jacobi_consts = np.ones(len(calculated_jacobi_const)) * i
        jacobi_difference = abs(real_jacobi_consts - calculated_jacobi_const)
    return jacobi_difference
    






    