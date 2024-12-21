'''
    Project     : Hot Rolling Fuzzy Control System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 2
    Description : This script is a template of for the use case of the implementation of the fuzzy algorith as described in the paper 
                  "Fuzzy control algorithm for the prediction of tension variations in hot rolling" - Jong-Yeob Jung, Yong-Taek Im for both
                  steady state operation.
    Designer    : Ronaldo Tsela
    Date        : 20/12/2024
    Requires    : hot_rolling_fuzzy
'''
from hot_rolling_fuzzy_package.hot_rolling_fuzzy_package.hot_rolling_fuzzy import *

if __name__ == "__main__":
    
    # Create an object 
    ctrl = HotRollingFuzzyControl()
    
    # Define the crisp input values 
    h_in_test = 3.7
    C_test = 10.7

    # Make the prediction
    ctrl.compute_steady_state(h_in_test, C_test)

    # Read the predicted values
    d_sigma_f = ctrl.get_d_sigma_f()
    d_sigma_b = ctrl.get_d_sigma_b()
    
    # Print the results
    print(f"d_sigma_f = { d_sigma_f }")
    print(f"d_sigma_b = { d_sigma_b }")
