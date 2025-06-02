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
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding='utf-8') # Read CSV file and skip header, generates new row for each line
    return data

def initialConditions(filename):
    data = readCSV(filename)

    row = data[0] if data.shape else data  # Get the first row of data

    def toFloat(val):
            return float(str(val).strip().replace('"', ''))

    orbit_id = toFloat(row['Id_'])  # ID of Orbit
    x = toFloat(row['x0_LU_'])
    y = toFloat(row['y0_LU_'])
    z = toFloat(row['z0_LU_'])
    x_prime = toFloat(row['vx0_LUTU_'])
    y_prime = toFloat(row['vy0_LUTU_'])
    z_prime = toFloat(row['vz0_LUTU_'])
    mu = toFloat(row['Mass_ratio'])
    
    return [x, y, z, x_prime, y_prime, z_prime], orbit_id, mu 

def EOM(state, t, mu):
    x, y, z, x_prime, y_prime, z_prime = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_2prime = 2 * y_prime + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_2prime = -2 * x_prime + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_2prime = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    return [x_prime, y_prime, z_prime, x_2prime, y_2prime, z_2prime]

def main():
    if len(sys.argv) < 2:
        print("python CR3BP.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]  # Get the filename from command line arguments
    state, orbit_id, mu = initialConditions(filename)

    t_0 = 0  # Initial time
    t_f = 1000  # Final time
    t_eval = np.linspace(t_0, t_f, 1000)  # Time points at which to store the solution

    sol = spi.odeint(EOM, state, t_eval, args=(mu,)) 

    #plotting the results


if __name__ == "__main__":
    main()
