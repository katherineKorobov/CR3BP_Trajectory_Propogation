"""
Katherine Korobov
Cislunar Initial Orbit Determination Using Adbissible Regions Theory CU SPUR Project
Mentors: Queenique Dinh, Marcus Holzinger
Date: May 28, 2024
This code is a Python script that simulates the motion of a spacecraft in the 
Circular Restricted Three-Body Problem (CR3BP) using odeint, a 4th Order Runge-Kutta method for numerical integration.

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
    x_0 = toFloat(row['x0_LU_'])
    y_0 = toFloat(row['y0_LU_'])
    z_0 = toFloat(row['z0_LU_'])
    x_prime0 = toFloat(row['vx0_LUTU_'])
    y_prime0 = toFloat(row['vy0_LUTU_'])
    z_prime0 = toFloat(row['vz0_LUTU_'])
    mu = toFloat(row['Mass_ratio'])

    state_0 = [x_0, y_0, z_0, x_prime0, y_prime0, z_prime0]  # Initial state vector
    
    return state_0, orbit_id, mu 

def modelEOM(state, t, mu):
    x, y, z, x_dot, y_dot, z_dot = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_ddot = 2 * y_dot + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_ddot = -2 * x_dot + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_ddot = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    state_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

    return state_dot

def main():
    if len(sys.argv) < 2:
        print("python CR3BP.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]  # Get the filename from command line arguments
    state_0, orbit_id, mu = initialConditions(filename)

    t_0 = 0  # Initial time
    t_f = 15  # Final time
    t_eval = np.linspace(t_0, t_f, 1000)  # Time points at which to store the solution

    sol = spi.odeint(modelEOM, state_0, t_eval, args=(mu,))

    #Solutions in the rotational frame
    x_rot = sol[:, 0]  # x position
    y_rot = sol[:, 1]  # y position
    z_rot = sol[:, 2]  # z position

    #Solutions in the inertial frame
    x_inert = sol[:, 0] * np.cos(t_eval) - sol[:, 1] * np.sin(t_eval)
    y_inert = sol[:, 0] * np.sin(t_eval) + sol[:, 1] * np.cos(t_eval)
    z_inert = sol[:, 2]  # z position remains the same in both frames

    # Plotting the results
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(x_rot, y_rot, z_rot, label='Rotational Frame', color='blue')

    fig_inert = plt.figure()
    ax_inert = plt.axes(projection='3d')
    ax_inert.plot3D(x_inert, y_inert, z_inert, label='Inertial Frame', color='red')


    plt.show()


if __name__ == "__main__":
    main()