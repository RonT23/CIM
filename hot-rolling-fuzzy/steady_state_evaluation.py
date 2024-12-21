'''
    Project     : Hot Rolling Fuzzy Control System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 2
    Description : This script evaluates the implementation of the fuzzy algorith as described in the paper "Fuzzy control algorithm for the 
                  prediction of tension variations in hot rolling" - Jong-Yeob Jung, Yong-Taek Im, using the provided (in paper) evaluation 
                  and testing data samples for the steady-state operation.
    Designer    : Ronaldo Tsela
    Date        : 20/12/2024
    Requires    : hot_rolling_fuzzy, matplotlib
'''

import matplotlib.pyplot as plt
from hot_rolling_fuzzy_package.hot_rolling_fuzzy_package.hot_rolling_fuzzy import *

if __name__ == "__main__":
    
    ctrl = HotRollingFuzzyControl()

    # Table 10 (List of input crisp values for testing and evaluation)
    h_in_test = np.array([2.00, 1.53, 3.25, 3.25, 3.25, 3.75, 3.75, 3.50, 4.25, 4.25, 4.25, 4.75, 4.50, 4.75, 5.50, 6.00])
    C_test = np.array([45.0, 60.0, 25.0, 45.0, 55.0, 12.0, 32.0, 43.0, 15.0, 25.0, 40.0, 13.0, 27.0, 40.0, 30.0, 40.0])

    # Table 10 (List of FEM results as expected crisp values for evaluation)
    d_sigma_f_expected = np.array([22.6, 22.6, 16.7, 20.5, 21.5, 18.8, 17.9, 16.8, 18.4, 19.1, 18.6, 14.8, 11.2, 10.9, 12.8, 9.5])
    d_sigma_b_expected = np.array([29.8, 31.5, 25.8, 30.5, 32.3, 25.2, 24.3, 22.7, 25.2, 25.8, 27.7, 25.8, 16.8, 14.9, 16.5, 12.9])

    # Fuzzy results will be stored here
    d_sigma_f_predicted = []
    d_sigma_b_predicted = []

    # Absolute error between the fuzzy results and the expected output
    d_sigma_f_error = []
    d_sigma_b_error = []

    # Loop over the test/ evaluation set
    for i in range(len(h_in_test)):

        # compute the fuzzy results
        ctrl.compute_steady_state(h_in_test[i], C_test[i])

        # get the computed values
        d_sigma_f = ctrl.get_d_sigma_f()
        d_sigma_b = ctrl.get_d_sigma_b()

        # append results
        d_sigma_f_predicted.append(d_sigma_f)
        d_sigma_b_predicted.append(d_sigma_b)

        # compute the absolute error
        d_sigma_f_error.append(abs(d_sigma_f - d_sigma_f_expected[i]) / d_sigma_f_expected[i])
        d_sigma_b_error.append(abs(d_sigma_b - d_sigma_b_expected[i]) / d_sigma_b_expected[i])

    plt.figure(figsize=(8, 6))
    plt.plot(d_sigma_f_expected, 'o-k', label="Expected", markersize=6, linewidth=1.5)
    plt.plot(d_sigma_f_predicted, 'x--b', label="Predicted", markersize=6, linewidth=1.5)
    plt.xlabel("Test Case", fontsize=12)
    plt.ylabel("Front Tension Change (%)", fontsize=12)
    plt.title("Expected vs Predicted Front Tension Change", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(d_sigma_b_expected, 'o-k', label="Expected", markersize=6, linewidth=1.5)
    plt.plot(d_sigma_b_predicted, 'x--b', label="Predicted", markersize=6, linewidth=1.5)
    plt.xlabel("Test Case", fontsize=12)
    plt.ylabel("Back Tension Change (%)", fontsize=12)
    plt.title("Expected vs Predicted Back Tension Change", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.plot(d_sigma_f_error, 'o-g', label="Front Tension", markersize=6, linewidth=1.5)
    plt.plot(d_sigma_b_error, 'o-r', label="Back Tension", markersize=6, linewidth=1.5)
    plt.xlabel("Test Case", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Absolute Normalized Tension Change Error", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
   
    plt.show()