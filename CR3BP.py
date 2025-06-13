import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as spi 
import sys

def readCSV(filename):
    '''
    @brief Reads medium-sized CSV file
    @param File to read data from
    @return Nunmpy array labeled data

    Notes: names = True -> first row of file is column names, dtype = None -> infers data types automatically
    '''
    data = np.genfromtxt(filename, delimiter = ',', names = True, dtype = None) 
    return data

def toFloat(val):
        '''
        @brief Converts a quoted string (or any value) to a float
        @args val: A value expected to represent a float, wrapped in quotes
        @return float: The floating-pointing number after removing quotes and converting

        Note: Could error if the cleaned value cannot be converted to a flaot
        '''
        return float(str(val).strip().replace('"', '')) # Get rid of quotes and convert to float

def initialConditions(filename):
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

        state_0 = [x_0, y_0, z_0, x_prime0, y_prime0, z_prime0] #Initial state vector

        states.append((state_0, jac_const, period, mu)) #Alters list by adding orbit
    
    return states

def modelEOM(state, t, mu):
    '''
    @brief Defines Equations of Motion for CR3BP and in odeint to integrate orrbit trajectories
    @param state: the current (soon to be previous) state list
    @param t: The time (TU) thtat corresponds to state
    @param mu: The mass ratio between the primary and secondary body
    @return state_dot: the derivative of the state list

    Note: odeint takes in a derivative state  list, integrates, and outputs the state trajectory
    Note: t is not used because the EOM are time-independent. However, it is used to satisfy odeint which looks for the time input in the function call
    '''
    x, y, z, x_dot, y_dot, z_dot = state 

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to primary body
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to secondary body

    x_ddot = 2 * y_dot + x - ((1 - mu) * (x + mu)) / r1**3 - (mu * (x - 1 + mu)) / r2**3
    y_ddot = -2 * x_dot + y - ((1 - mu) * y) / r1**3 - (mu * y) / r2**3
    z_ddot = -((1 - mu) * z) / r1**3 - (mu * z) / r2**3

    state_dot = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

    return state_dot

def plotOrbits(all_solutions, jac_constants):
    jac_min = np.min(jac_constants)
    jac_max = np.max(jac_constants)
    cmap = plt.cm.viridis

    fig_rot = plt.figure()
    ax = plt.axes(projection='3d')

    for sol, jac_const in all_solutions:
        color = cmap((jac_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5)
        x_rot = sol[:, 0]
        y_rot = sol[:, 1]
        z_rot = sol[:, 2]
        ax.plot3D(x_rot, y_rot, z_rot, color=color)

    ax.set_xlabel('X Position (LU)')
    ax.set_ylabel('Y Position (LU)')
    ax.set_zlabel('Z Position (LU)')
    ax.set_title('Visualization of CR3BP Halo Orbits')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(jac_min, jac_max)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Jacobi Constant (LU\u00b2/TU\u00b2)')
    plt.tight_layout()

def plotJacobi(all_solutions, jac_constants):
    jac_min = np.min(jac_constants)
    jac_max = np.max(jac_constants)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots()

    for sol, jac_const, mu, t_eval in all_solutions:
        color = cmap((jac_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5)
        x = sol[:, 0]
        y = sol[:, 1]
        z = sol[:, 2]

        x_dot = sol[:, 3]
        y_dot = sol[:, 4]
        z_dot = sol[:, 5]

        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

        u_star = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2
        v_2 = x_dot**2 + y_dot**2 + z_dot**2
        jacobian_const = 2 * u_star - v_2

        ax.plot(t_eval, jacobian_const, color=color)
    ax.set_xlabel('Time (TU)')
    ax.set_ylabel('Jacobi Constant (LU\u00b2 / TU\u00b2)')
    ax.set_title('Time v. Jacobi Constant')

def plotSpherical(all_solutions, jac_constants):
    jac_min = np.min(jac_constants)
    jac_max = np.max(jac_constants)
    cmap = plt.cm.viridis

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    ax_angular, ax_dRange, ax_dTheta, ax_dPhi = axs.flatten()

    for sol, jac_const, mu, t_eval in all_solutions:
        #normalize Jacobi Const for colormap
        color = cmap((jac_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5) 
        x = sol[:, 0]
        y = sol[:, 1]
        z = sol[:, 2]

        x_dot = sol[:, 3]
        y_dot = sol[:, 4]
        z_dot = sol[:, 5]

        # Spherical coordinates and their derivatives
        rho = np.sqrt(x**2 + y**2 + z**2)
        rho_dot = (x * x_dot + y * y_dot + z * z_dot) / rho

        theta = np.arctan2(y, x)
        theta_dot = (x * y_dot - y * x_dot) / (x**2 + y**2)

        phi_num = np.sqrt(x**2 + y**2)
        phi = np.arctan2(phi_num, z)
        # Avoid division by zero in denominator
        dphi_den = np.where((phi_num * (x**2 + y**2 + z**2)) == 0, 1e-12, phi_num * (x**2 + y**2 + z**2))
        dphi_num = x * z * x_dot**2 + (-z) + y * (z * y_dot - y * z_dot)
        phi_dot = dphi_num / dphi_den

        ax_angular.plot(theta, phi, color = color)
        ax_dRange.plot(t_eval, rho, color = color)
        ax_dTheta.plot(t_eval, theta, color = color)
        ax_dPhi.plot(t_eval, phi, color = color)

    ax_angular.set_xlabel('Theta (LU)')
    ax_angular.set_ylabel('Phi (LU)')
    ax_angular.set_title('Theta v. Phi')

    ax_dRange.set_xlabel('Time (TU)')
    ax_dRange.set_ylabel('Range-Rate (LU/TU)')
    ax_dRange.set_title('Time v. Range-Rate')

    ax_dTheta.set_xlabel('Time (TU)')
    ax_dTheta.set_ylabel('Theta-Rate (rad/TU)')
    ax_dTheta.set_title('Time v. Theta-Rate')

    ax_dPhi.set_xlabel('Time (TU)')
    ax_dPhi.set_ylabel('Phi (rad/TU)')
    ax_dPhi.set_title('Time v. Phi')

    fig.suptitle('Spherical Coordinates Phase Plots')

def main():
    if len(sys.argv) < 2:
        print("Format: python CR3BP.py <initial_conditions_file>")
        return
    
    filename = sys.argv[1]
    all_states = initialConditions(filename)

    #Define 
    all_solutions = []
    jac_constants = []

    for state_0, jac_const, period, mu in all_states:
        t0 = 0  # Initial time
        tf = period
        t_eval = np.linspace(0, tf, 1000)

        sol = spi.odeint(modelEOM, state_0, t_eval, args=(mu,))

        all_solutions.append((sol, jac_const, mu, t_eval))
        jac_constants.append(jac_const)

    plotOrbits([(sol, jac_const) for sol, jac_const, mu, t_eval in all_solutions], np.array(jac_constants))
    plotJacobi(all_solutions, np.array(jac_constants))
    plotSpherical(all_solutions, np.array(jac_constants))
    
    plt.show()

if __name__ == "__main__":
    main()