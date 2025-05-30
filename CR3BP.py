"""
Katherine Korobov
Cislunar Initial Orbit Determination Using Adbissible Regions Theory CU SPUR Project
Mentors: Queenique Dinh, Marcus Holzinger
Date: May 28, 2024
This code is a Python script that simulates the motion of a spacecraft in the 
Circular Restricted Three-Body Problem (CR3BP) using the - method for numerical integration.

orbits:
L2 Souther Halo
L1 Norhern Halo
"""

# Import useful libraries
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as spi 
import sys

def readCSV(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1) # Read CSV file and skip header, generates new row for each line
    return data

def initialConditions(filename):
    data = readCSV(filename)

    orbit_id = data[0, 0]  # ID of Orbit
    x = data[0, 1]  # Initial x position
    y = data[0, 2]  # Initial y position
    z = data[0, 3]  # Initial z position
    x_prime = data[0, 4]  # Initial x velocity
    y_prime = data[0, 5]  # Initial y velocity
    z_prime = data[0, 6]  # Initial z velocity
    mu = data[0, 12] #Mass ratio of the two primary bodies
    
    return [x, y, z, x_prime, y_prime, z_prime] , orbit_id, mu 

def EOM(state, t, mu):
    x, y, z, x_prime, y_prime, z_prime = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_2prime = 2 * y_prime + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_2prime = -2 * x_prime + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_2prime = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    return [x_prime, y_prime, z_prime, x_2prime, y_2prime, z_2prime]

def main():
    if len(sys.argv) != 3:
        print("Usage: python CR3BP.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]  # Get the filename from command line arguments
    state, orbit_id, mu = initialConditions(filename)

    print(f"Orbit ID: {orbit_id}")
    print(f"Initial State: {state}")
    print(f"Mass Ratio (mu): {mu}")

"""""
    t_0 = 0  # Initial time
    t_f = 1000  # Final time
    t_eval = np.linspace(t_0, t_f, 1000)  # Time points at which to store the solution

    sol = spi.odeint(EOM, state, t_eval)
"""
    #plotting the results


if __name__ == "__main__":
    main()
