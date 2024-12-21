import matplotlib.pyplot as plt
from hot_rolling_fuzzy_package.hot_rolling_fuzzy_package.hot_rolling_fuzzy import *

if __name__ == "__main__":
    
    # Create an object 
    ctrl = HotRollingFuzzyControl()
    
    # Define the crisp input values
    h_in_test = 3.7
    C_test = 10.7

    # Make the prediction
    d_sigma_f, d_sigma_b = ctrl.compute(h_in_test, C_test)

    # Print the results
    print(f"d_sigma_f = {d_sigma_f}")
    print(f"d_sigma_b = {d_sigma_b}")