'''
File: .py
Description: The main function for the project: "CR3BP_Trajectory Propogation".
Author: Katherine Korobov
Created: 20 JuMayne 2026
Last Modified: 27 May 2026
'''

import numpy as np 
import matplotlib.pyplot as plt  

def plotOrbits(all_data, jac_constants):
    jac_min = np.min(jac_constants)
    jac_max = np.max(jac_constants)
    cmap = plt.cm.viridis

    fig_rot = plt.figure()
    ax = plt.axes(projection='3d')

    for measurement in all_data:
        color = cmap((measurement.jacobi_const - jac_min) / (jac_max - jac_min) if jac_max > jac_min else 0.5)
        x_rot = measurement.state_vector.x
        y_rot = measurement.state_vector.y
        z_rot = measurement.state_vector.z
        
        if max(z_rot) < 1e-10:
            z_rot = 0

        ax.plot3D(x_rot, y_rot, z_rot, color=color)    

    ax.set_xlabel('X Position (LU)')
    ax.set_ylabel('Y Position (LU)')
    ax.set_zlabel('Z Position (LU)')
    ax.set_title('Visualization of Orbit')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(jac_min, jac_max)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Jacobi Constant (LU\u00b2/TU\u00b2)')
    plt.tight_layout()


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