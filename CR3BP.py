"""
Katherine Korobov
Cislunar Initial Orbit Determination Using Adbissible Regions Theory CU SPUR Project
Mentors: Queenique Dinh, Marcus Holzinger
Date: May 28, 2024
This code is a Python script that simulates the motion of a spacecraft in the 
Circular Restricted Three-Body Problem (CR3BP) using odeint, a 4th Order Runge-Kutta method for numerical integration.
"""
# Import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as spi 
import sys

def readCSV(filename):
    data = np.genfromtxt(filename, delimiter = ',', names = True, dtype = None, encoding = 'utf-8') # Read CSV file
    return data

def initialConditions(filename):
    data = readCSV(filename)
    
    if data.shape == ():
        data = [data]   

    def toFloat(val):
        return float(str(val).strip().replace('"', '')) # Get rid of quotes and convert to float

    states = []
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

        state_0 = [x_0, y_0, z_0, x_prime0, y_prime0, z_prime0]  # Initial state vector

        states.append((state_0, jac_const, period, mu))
    
    return states

def modelEOM(state, t, mu):
    x, y, z, x_dot, y_dot, z_dot = state 

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_ddot = 2 * y_dot + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_ddot = -2 * x_dot + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_ddot = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    state_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

    return state_dot

def plotResults(sol, color, ax):
   #Solutions in the rotational frame
    x_rot = sol[:, 0]  # x position
    y_rot = sol[:, 1]  # y position
    z_rot = sol[:, 2]  # z position

    # Plotting the results
    ax.plot3D(x_rot, y_rot, z_rot, color=color)


def main():
    if len(sys.argv) < 2:
        print("Format: python CR3BP.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]  # Get the filename from command line arguments
    all_states = initialConditions(filename)

    # Get all Jacobi constants for color mapping
    jac_constants = np.array([jac_const for _, jac_const, _, _ in all_states])
    jac_min, jac_max = np.min(jac_constants), np.max(jac_constants)
    cmap = plt.cm.viridis

    fig_rot = plt.figure()
    ax = plt.axes(projection='3d')  # Create a 3D axis for plotting


    for state_0, jac_const, period, mu in all_states:
        t_0 = 0  # Initial time
        t_f = period
        t_eval = np.linspace(t_0, t_f, 1000)
        sol = spi.odeint(modelEOM, state_0, t_eval, args=(mu,))
        color = cmap((jac_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5)
        plotResults(sol, color, ax)

    ax.set_xlabel('X Position (LU)')
    ax.set_ylabel('Y Position (LU)')
    ax.set_zlabel('Z Position (LU)')
    ax.set_title('Visualization of CR3BP Halo Orbits') 

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(jac_min, jac_max)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Jacobi Constant')


    plt.show()  # Show the plot

if __name__ == "__main__":
    main()