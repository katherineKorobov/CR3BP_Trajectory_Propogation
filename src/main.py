'''
File: main.py
Description: The main function for the project: "CR3BP_Trajectory Propogation".
Author: Katherine Korobov
Created: 20 May 2026
Last Modified: 26 May 2026
'''

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as spi 
import sys

sys.path.append(r'E:\projects\CR3BP_Trajectory_Propogation') # create path to the parent directory

from src.utils.gen_util import readCSV
from src.utils.gen_util import toFloat
from src.plotting.plot_orbits import plotOrbits, plotSpherical
from src.dynamics.cartesianCR3BP import modelEOM
from src.plotting.plot_jacobi_const import plot_jacobi_const
from src.plotting.plot_jacobi_error import plot_jacobi_error
from src.objects.Measurement import Measurement
from src.objects.State import State

def buildInitialConditions(filename):
    '''
    @brief Collects Orbit information, creates a state vector, and builds a vector containing the state, Jacobi constant, orbit period, and mass ratio 
    @param File to read data from (calls readCSV(filename))
    @return List of all orbit state information
    '''
    data = readCSV(filename)
    
    #Checks if data is scalar and ensures it is iterable
    if data.shape == ():
        data = [data]   


    states = [] #Empty list for all orbital state information
    for row in data:
        x_0 = toFloat(row['x0_LU_'])
        y_0 = toFloat(row['y0_LU_'])
        z_0 = toFloat(row['z0_LU_'])
        x_prime0 = toFloat(row['vx0_LUTU_'])
        y_prime0 = toFloat(row['vy0_LUTU_'])
        z_prime0 = toFloat(row['vz0_LUTU_'])
        jac_const = toFloat(row['Jacobi_constant_LU2TU2_'])
        period = toFloat(row['Period_TU_'])
        mu = toFloat(row['Mass_ratio'])

        initial_state = [x_0, y_0, z_0, x_prime0, y_prime0, z_prime0] #Initial state vector
        state = State(initial_state, 0, jac_const, period, mu)
        states.append(state)

    return states

def main():
    if len(sys.argv) < 2:
        print("Format: python main.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]
    all_states = buildInitialConditions(filename) # returns array of State classes for each orbit data provided

    #Define 
    all_solutions = []
    jac_constants = []
    steps = 10000

    for state in all_states:
        t_eval = np.linspace(state.init_time, state.period, steps)

        sol = spi.odeint(modelEOM, state.init_state_vector, t_eval, args=(state.mu,)) # propogates

        state.state_vector.vstack(sol)
        state.time = t_eval
        
        #all_solutions.append((sol, state.jacobi_constant, state.mu, t_eval))
        jac_constants.append(state.jacobi_constant)

    jac_constants = np.array(jac_constants) # Convert to numpy array for ease

    plotOrbits([(sol, jac_const) for sol, jac_const, mu, t_eval in all_solutions], jac_constants)
    
    #plot_jacobi_const(all_solutions, jac_constants)

    #plotSpherical(all_solutions, np.array(jac_constants))

    #plot_jacobi_error(all_states, all_solutions, jac_constants)
    
    

    plt.show()

if __name__ == "__main__":
    main()