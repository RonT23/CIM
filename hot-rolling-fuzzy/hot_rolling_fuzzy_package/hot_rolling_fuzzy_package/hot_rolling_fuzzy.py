"""
    Project     : Hot Rolling Fuzzy Control System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 2
    Description : Reproduction of the fuzzy system described in "Fuzzy control algorithm for the 
                  prediction of tension variations in hot rolling" - Jong-Yeob Jung, Yong-Taek Im.
    Designer    : Ronaldo Tsela
    Date        : 20/12/2024
    Requires    : numpy, math
"""

import numpy as np
import math

"""
    This class implements the Fuzzy Logic System used for hot rolling process control.

    Attributes:
        accuracy (float): The arithmetic accuracy used to represent the input crisp values.
        h_in_min_val (float): The minimum permissible value for the entry thickness input.
        h_in_max_val (float): The maximum permissible value for the entry thickness input.
        C_perc_min_val (float): The minimum permissible value for the carbon equivalent input.
        C_perc_max_val (float): The maximum permissible value for the carbon equivalent input.
        h_in_range (dict): [key = level, value = range] Maps discrete classes (levels) to the input thickness ranges.
        C_perc_range (dict): [key = level, value = range] Maps discrete classes (levels) to the carbon equivalent ranges.
        d_sigma_range (dict): [key = singleton, value = range] Maps singletons to the variation in tension ranges.
        d_sigma_singleton (dict): [key = linguistic variable, value = singleton] Maps linguistic fuzzy variables to singletons for the variation in tension.
        h_in_mf (dict): [key = linguistic variable, value = membership list] Implements a triangular membership function for the entry thickness. Maps linguistic variables to memberships per level.
        C_mf (dict): [key = linguistic variable, value = membership list] Implements a triangular membership function for the carbon equivalent. Maps linguistic variables to memberships per level.
        rules (list): A list of rules. Each entry is a dictionary representing a rule. 
                    Each rule is defined as {if: X1, and: X2, then_1: Y1, then_2: Y2}, where X1 and X2 are the input fuzzy variables, and Y1 and Y2 are the outputs.

    Methods:
        __init__():
            Initializes a new HotRollingFuzzyControl instance with a specified input arithmetic accuracy.
            By default, the accuracy is set to 0.01.

        compute(h_in: float, C_perc: float):
            Predicts the front and back tension variations using fuzzy logic.

            Args:
                h_in (float): The crisp entry thickness input.
                C_perc (float): The crisp carbon equivalent input.

            Returns:
                (float, float): A tuple of two floats representing the computed variations 
                    in front tension (Delta_sigma_f) and back tension (Delta_sigma_b).

        set_h_in_min_val(val: float):
            Sets the minimum limit for the entry thickness.

            Args:
                val (float): The minimum permissible value for the entry thickness.

        set_h_in_max_val(val: float):
            Sets the maximum limit for the entry thickness.

            Args:
                val (float): The maximum permissible value for the entry thickness.

        set_C_perc_min_val(val: float):
            Sets the minimum limit for the carbon equivalent.

            Args:
                val (float): The minimum permissible value for the carbon equivalent.

        set_C_perc_max_val(val: float):
            Sets the maximum limit for the carbon equivalent.

            Args:
                val (float): The maximum permissible value for the carbon equivalent.

              
        classify(x: (float), R: (dict)): 
            Classifies the input values based on the provided range dictionaries.
            
            Args:
                x (float): The input value to classify.
                R (dict):  A reference dictionary in the form {class: range}.
            
            Return:
                (str): The class assigned to the input value if it falls within the range, 
                 otherwise returns None.

        fuzzify(x: (float), mf: (dict)):
            Converts the input into a fuzzy set by assigning appropriate membership values
            based on the provided membership function.

            Args:
                x (float): The input level (classified input). It works as an index to the membership vectors.
                mf (dict): A dictionary of membership functions in the form {linguistic variable: membership list}.
            
            Return: 
                (list) : A list of dictionaries representing the fuzzified values in the form {linguistic variable: membership value}.
        
        infere(fuzzy_input: (list)):
            Evaluates the fuzzy rules and computes the exact fuzzy output values 
            using Mamdani's minimum method.

            Args: 
                fuzzy_input (list): A list of dictionaries representing fuzzy sets.

            Return:
                (float, float) : The exact fuzzy output values.

        defuzzify(fuzzy_output: (float)):
            Computes the actual crisp value from the exact fuzzy value obtained after inference.

            Args:
                fuzzy_output (float): The exact fuzzy value obtained after inference.

            Returns: 
                (float) : The computed crisp value, represented by the first value of the interpolated range 
                        
        interpolate(R1: (list), R2: (list), C1: (int), C2: (int), C12: (float)):
            Performs interpolation between two ranges characterized by discrete class numbers 
            to determine a third interpolated range.

            Args:
                R1 (list): The first range limits [min, max].
                R2 (list): The second range limits [min, max].
                C1 (int): The first class number.
                C2 (int): The second class number.
                C12 (float): The combined class number between the first and second classes.

            Returns:
                (float, float): Returns the lower and upper limits representing the interpolated range limits.
"""

class HotRollingFuzzyControl:
    def __init__(self):

        # Adjustable ranges
        self.accuracy          = 0.01

        self.h_in_min_val      = 1.0
        self.h_in_max_val      = 10.0

        self.C_perc_min_val    = 5.0
        self.C_perc_max_val    = 80.0

        # Table 4 (level <-> crisp ranges)
        self.h_in_range = {
            "1": np.arange(self.h_in_min_val, 3.0, self.accuracy),
            "2": np.arange(3.0, 3.5,self.accuracy),
            "3": np.arange(3.5, 4.0,self.accuracy),
            "4": np.arange(4.0, 4.5,self.accuracy),
            "5": np.arange(4.5, 5.0,self.accuracy),
            "6": np.arange(5.0, self.h_in_max_val, self.accuracy)
        }

        # Table 6 (level <-> crisp ranges)
        self.C_range = {
            "1": np.arange(self.C_perc_min_val, 20.0, self.accuracy),
            "2": np.arange(20.0, 35.0, self.accuracy),
            "3": np.arange(35.0, 50.0, self.accuracy),
            "4": np.arange(50.0, self.C_perc_max_val, self.accuracy)
        }

        # Table 9 (singletons <-> crisp ranges)
        self.d_sigma_range = {
            "1" : np.arange(9.0,  10.99, 0.01),
            "2" : np.arange(11.0, 12.99, 0.01),
            "3" : np.arange(13.0, 14.99, 0.01),
            "4" : np.arange(15.0, 16.99, 0.01),
            "5" : np.arange(17.0, 18.99, 0.01),
            "6" : np.arange(19.0, 20.99, 0.01),
            "7" : np.arange(21.0, 22.99, 0.01),
            "8" : np.arange(23.0, 24.99, 0.01),
            "9" : np.arange(25.0, 26.99, 0.01),
            "10": np.arange(27.0, 28.99, 0.01),
            "11": np.arange(29.0, 30.99, 0.01),
            "12": np.arange(31.0, 32.99, 0.01)
        }

        # Table 9 (linguistic variables <-> singletons)
        self.d_sigma_singleton = {
            "SLL"  : 1,
            "SML"  : 2,
            "SSL"  : 3,
            "SSB"  : 4,
            "SMB"  : 5,
            "SLB"  : 6,
            "BLL"  : 7,
            "BML"  : 8,
            "BSL"  : 9,
            "BSB"  : 10,
            "BMB"  : 11,
            "BLB"  : 12
        }

        # Table 5 (Triangular membership functions)
        self.h_in_mf = {
            "TLL": [1.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            "TML": [0.4, 0.9, 0.2, 0.0, 0.0, 0.0],
            "TSL": [0.0, 0.2, 0.9, 0.6, 0.0, 0.0],
            "TSB": [0.0, 0.2, 0.6, 0.9, 0.1, 0.0],
            "TMB": [0.0, 0.0, 0.0, 0.0, 0.9, 0.5],
            "TLB": [0.0, 0.0, 0.0, 0.0, 0.3, 1.0]
        }

        # Table 7 (Triangular membership functions)
        self.C_mf = {
            "CLL": [1.0, 0.3, 0.0, 0.0],
            "CML": [0.3, 0.9, 0.4, 0.0],
            "CMB": [0.0, 0.3, 0.9, 0.2],
            "CLB": [0.0, 0.0, 0.3, 1.0]
        }

        # Table 8 (Rules: if X1 and X2 then Y1 and Y2)
        self.rules = [
    {'if': 'TLL', 'and': 'CLB', 'then_1': 'BML', 'then_2': 'BLB'}, # rule 1
    {'if': 'TLL', 'and': 'CMB', 'then_1': 'BML', 'then_2': 'BMB'}, # rule 2
    {'if': 'TML', 'and': 'CML', 'then_1': 'SLB', 'then_2': 'BMB'}, # rule 3
    {'if': 'TML', 'and': 'CMB', 'then_1': 'BLL', 'then_2': 'BLB'}, # rule 4
    {'if': 'TML', 'and': 'CLB', 'then_1': 'BML', 'then_2': 'BLB'}, # rule 5
    {'if': 'TSL', 'and': 'CLL', 'then_1': 'SMB', 'then_2': 'BSL'}, # rule 6
    {'if': 'TSL', 'and': 'CML', 'then_1': 'SSB', 'then_2': 'BML'}, # rule 7
    {'if': 'TSL', 'and': 'CMB', 'then_1': 'SSB', 'then_2': 'BLL'}, # rule 8
    {'if': 'TSB', 'and': 'CLL', 'then_1': 'SMB', 'then_2': 'BSL'}, # rule 9
    {'if': 'TSB', 'and': 'CML', 'then_1': 'SLB', 'then_2': 'BSB'}, # rule 10
    {'if': 'TSB', 'and': 'CMB', 'then_1': 'SLB', 'then_2': 'BSB'}, # rule 11
    {'if': 'TMB', 'and': 'CLL', 'then_1': 'SSL', 'then_2': 'BLL'}, # rule 12
    {'if': 'TMB', 'and': 'CML', 'then_1': 'SML', 'then_2': 'SSB'}, # rule 13
    {'if': 'TMB', 'and': 'CMB', 'then_1': 'SML', 'then_2': 'SSL'}, # rule 14
    {'if': 'TLB', 'and': 'CML', 'then_1': 'SML', 'then_2': 'SMB'}, # rule 15
    {'if': 'TLB', 'and': 'CMB', 'then_1': 'SLL', 'then_2': 'SSL'}  # rule 16
]
    
    ##### Main processing function

    def compute(self, h_in, C_perc):

        # Check the input
        if (h_in < self.h_in_min_val) or (h_in > self.h_in_max_val) or (C_perc < self.C_perc_min_val) or (C_perc > self.C_perc_max_val):
            print("[ERROR] HotRollingFuzzyControl:compute : Input is out of range")
            exit(1)
        
        # 1. Classify the crisp input and produce discrete values (levels)
        classified_h_in   = int( self.classify(h_in, self.h_in_range) )
        classified_C_perc = int( self.classify(C_perc, self.C_range)  )

        # 2. Fuzzyfy the discrete input values and return the membership and linguistic variables 
        fuzzy_h_in    = self.fuzzify(classified_h_in - 1, self.h_in_mf)
        fuzzy_C_perc  = self.fuzzify(classified_C_perc - 1, self.C_mf)

        # 3. Infere the fuzzy system rules
        fuzzy_d_sigma_f, fuzzy_d_sigma_b = self.infere([fuzzy_h_in, fuzzy_C_perc])

        # 4. Defuzzify using interpolation scheme to get crisp actual values
        d_sigma_f = self.defuzzify(fuzzy_d_sigma_f)
        d_sigma_b = self.defuzzify(fuzzy_d_sigma_b)

        return d_sigma_f, d_sigma_b

    #####

    ##### Dynamic adjustable input parameters for range limits and accuracy definition
    
    def set_h_in_min_val(self, val):
        if( val > 0):
            self.h_in_min_val = val
        else:
            print("[ERROR] HotRollingFuzzyControl.set_h_in_min_val : values must be positive. ")

    def set_h_in_max_val(self, val):
        if (val > 0):
            self.h_in_max_val = val
        else:
            print("[ERROR] HotRollingFuzzyControl.set_h_in_max_val : values must be positive. ")
        
    def set_C_perc_min_val(self, val):
        if ( val > 0 ): 
            self.C_perc_min_val = val
        else:
            print("[ERROR] HotRollingFuzzyControl.set_C_perc_min_val : values must be positive. ")
        
    def set_C_pers_max_val(self, val):
        if ( val > 0 ):
            self.C_perc_max_val = val
        else:
            print("[ERROR] HotRollingFuzzyControl.set_C_perc_max_val : values must be positive. ")

    def set_arithm_accuracy(self, val):
        if ( val > 0 ):
            self.C_perc_max_val = val
        else:
            print("[ERROR] HotRollingFuzzyControl.set_arithm_accuracy : values must be positive. ")

    #####

    ##### Fuzzy operation functions

    def classify(self, x, R):
        for c, r in R.items():
            if ( r[0] <= x <= r[-1] + (r[1] - r[0]) ):
                return c
        return None

    def fuzzify(self, x, mf):
        fuzzified_dict = {}
        for linguistic_variable, membership in mf.items():
            if membership[x] > 0:
                fuzzified_dict[linguistic_variable] = membership[x]
        return fuzzified_dict

    def infere(self, fuzzy_input):

        yf = 0
        yb = 0

        sum_g  = 0
        sum_Yf = 0
        sum_Yb = 0

        for rule in self.rules:

            # mamdani minimum
            g = min(fuzzy_input[0].get(rule["if"], 0), fuzzy_input[1].get(rule["and"], 0))

            sum_g  += g
            sum_Yf += g * self.d_sigma_singleton.get(rule["then_1"], 0)
            sum_Yb += g * self.d_sigma_singleton.get(rule["then_2"], 0)

        if (sum_g > 0):
            yf = sum_Yf / sum_g
            yb = sum_Yb / sum_g

        return yf, yb
  
    def defuzzify(self, fuzzy_output):
        # get the integer part of the class and the next one
        c1 = math.floor(fuzzy_output)
        c2 = math.floor(fuzzy_output) + 1

        # Compute the limit range of the classes
        min_1 = min( self.d_sigma_range.get( str(c1) , 0) )
        max_1 = max( self.d_sigma_range.get( str(c1) , 0) )
        min_2 = min( self.d_sigma_range.get( str(c2) , 0) )
        max_2 = max( self.d_sigma_range.get( str(c2) , 0) )

        # interpolate the classes
        interpolated_range = self.interpolate([min_1, max_1], [min_2, max_2], c1, c2, fuzzy_output)

        # keep only the first value of the range
        return interpolated_range[0]

    def interpolate(self, R1, R2, C1, C2, C12):
        w = (C12 - C1) / (C2 - C1)
        u = (1 - w) * R1[1] + w * R2[1]
        l = (1 - w) * R1[0] + w * R2[0]
        return l, u

    #####