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


    orbital_data = [] #Empty list for all orbital state information
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

        initial_state = State(x_0, y_0, z_0, x_prime0, y_prime0, z_prime0) #Initial state vector
        measurement = Measurement(initial_state, 0, jac_const, period, mu)
        orbital_data.append(measurement)

    return orbital_data

def main():
    if len(sys.argv) < 2:
        print("Format: python main.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]
    all_data = buildInitialConditions(filename) # returns array of State classes for each orbit data provided

    #Define 
    all_solutions = []
    jac_constants = [] # holds all jacobi constants to plot
    steps = 10000 # number of steps propagator should do 

    for measurement in all_data:
        t_eval = np.linspace(measurement.init_time, measurement.period, steps)
        measurement.time = t_eval

        #TODO the odeint has specific input and outputs but I want classes so I can either create my own propagator or decompose the classes into the correct inputs
        sol = spi.odeint(modelEOM, measurement.init_state_vector.full_state, t_eval, args=(measurement.mu,)) # propogates

        measurement.state_vector.append(sol)
        
        #all_solutions.append((sol, state.jacobi_constant, state.mu, t_eval))
        jac_constants.append(measurement.jacobi_constant)

    jac_constants = np.array(jac_constants) # Convert to numpy array for ease

    plotOrbits(all_data, jac_constants)
    #plotOrbits([(sol, jac_const) for sol, jac_const, mu, t_eval in all_solutions], jac_constants)
    
    

    plt.show()

if __name__ == "__main__":
    main()