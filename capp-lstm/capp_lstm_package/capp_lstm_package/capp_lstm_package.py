"""
    Project     : CAPP LSTM-Based System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 3
    Description : Implementation of a CAPP system using an LSTM-based ANN architecture for prediction of up to four 
                  equivalent process chains. 
    Designer    : Ronaldo Tsela
    Date        : 4/1/2025
    Requires    : numpy, ...
"""
import os
import json
import numpy as np
import pandas as pd 

class Dataset:
    def __init__(self, filename):
        self.dataset_path = filename
    
    def load(self):
        with open(self.dataset_path, 'r') as file:
            data_json = json.load(file)

class CappLSTM:
    def __init__(self):
        
        # These will work as lookup tables for the input and output. These are provided by "../../data/part_encoding_and_process_list.json". 
        # Currently these are hardcoded. It would have been better if this class could configure itself.
        self.c_part_geometry = {
            "pure_axisymmetric"                       : 0,
            "axisymmetric_with_prismatic_features"    : 1,
            "prismatic"                               : 2,
            "prismatic_with_axisymmetric_features"    : 3,
            "prismatic_with_freeform_features"        : 4,
            "freeform"                                : 5,
            "unconventional"                          : 6
        }

        self.c_part_holes = {
            "none"              : 0,
            "normal"            : 1,
            "normal_threaded"   : 2,
            "normal_functional" : 3,
            "large"             : 4,
            "large_threaded"    : 5,
            "large_functional"  : 6
        }

        self.c_part_external_threads = {
            "yes"   : 1,
            "no"    : 0
        }

        self.c_part_surface_finish = {
            "rough"     : 0,
            "normal"    : 1,
            "good"      : 2,
            "very_good" : 3
        }

        self.c_part_tolerance = {
            "rough"   : 0,
            "normal"  : 1,
            "medium"  : 2,
            "tight"   : 3
        }

        self.c_part_batch_size = {
            "prototype" : 0,
            "small"     : 1,
            "medium"    : 2,
            "large"     : 3,
            "mass"      : 4
        }

        self.c_available_processes = {
                "Turning"                   : 1,
                "Milling"                   : 2,
                "5-axis Milling"            : 3,
                "SLM"                       : 4,
                "Sand Casting"              : 5,
                "High Pressure Die Casting" : 6,
                "Investment Casting"        : 7,
                "Turning Secondary"         : 8,
                "Milling Secondary"         : 9,
                "Hole Milling"              : 10,
                "5-axis Milling Secondary"  : 11,
                "Thread Milling"            : 12,
                "Tapping"                   : 13,
                "Grinding"                  : 14,
                "5-axis Grinding"           : 15,
                "Superfinishing"            : 16,
                "Drilling"                  : 17,
                "Boring"                    : 18,
                "Reaming"                   : 19,
                "Special Finishing"         : 20
        }

        # The number of available processes
        self.c_n_processes = len(self.c_available_processes)

        # The number of timesteps to predict. This equals the number of unique available processes
        self.c_max_steps = self.c_n_processes

        # Set default values for the training, validation, testing and single input files
        self.training_dataset_path      = "../../data/training.json"
        self.validation_dataset_path    = "../../data/validation.json"
        self.testing_dataset_path       = "../../data/test.json"
        
        self.input_file_path            = "../../data/input.json"
        self.output_file_path           = "../../data/output.json"

        self.scaller_path               = "../../metadata/scaler.ipk"
        self.model_path                 = "../../metadata/model.keras"

    # These setters are used to configure the file directory
    def set_training_dataset_path(self, filename):
        self.training_dataset_path = filename

    def set_validation_dataset_path(self, filename):
        self.validation_dataset_path = filename 

    def set_input_file_path(self, filename):
        self.input_file_path = filename 
    
    def set_output_file_path(self, filename):
        self.output_file_path = filename
    
    def set_test_dataset_path(self, filename):
        self.testing_dataset_path = filename


    ### Related to training
    """
        Function to encode the data from the JSON files that describe a part into numerical values.
        @ part_json: the part configuration in JSON format.
        @ return: the encoded part configuration (geometry, holes, external_threads, surface_finish, tolerance, batch_size).
    """
    def encode_features(self, part):
        
        part_features = part.get('part_features', {})

        # Extract the configuration data from the input
        part_geometry         = self.c_part_geometry[part_features.get('geometry')]
        part_holes            = self.c_part_holes[part_features.get('holes')]
        part_external_threads = self.c_part_external_threads[part_features.get('external_threads')]
        part_surface_finish   = self.c_part_surface_finish[part_features.get('surface_finish')]
        part_tolerance        = self.c_part_tolerance[part_features.get('tolerance')]
        part_batch_size       = self.c_part_batch_size[part_features.get('batch_size')]

        return (
            part_geometry,
            part_holes,
            part_external_threads,
            part_surface_finish,
            part_tolerance,
            part_batch_size
        )


    """
        Encode a list of processes into integer tokens (1..20),
        then pad/truncate to length = max_steps.
        We'll use 0 as the 'pad' index if needed.
    """
    def encode_processes(self, part_processes):
        encoded = []
        for p in part_processes:
            proc_id = self.c_processes.get(p, -1)
            encoded.append(proc_id)
        
        # Truncate if longer
        if len(encoded) > self.c_max_steps:
            encoded = encoded[:self.max_steps]
        
        # Pad if shorter
        while len(encoded) < self.max_steps:
            encoded.append(0) 

        return encoded  # shape (20,)


    """
        Function to load the data from the JSON files and preapre it for further processing creating a pandas dataframe.
        @ filename : the path to the JSON file.
        @ return : a pandas dataframe with the preapred data.
    """
    def load(self, filename):
        data_df = pd.DataFrame(columns=['geometry', 'holes', 'external_threads', 'surface_finish', 'tolerance', 'batch_size'])

        with open(filename, 'r') as file:
            data_json = json.load(file)

        for idx, part in enumerate(data_json):

            # Encode the part features
            feats = encode_features(part)
            data_df.loc[idx, INPUT_FEATURES_COLS] = feats

            process_chains = part.get('process_chains', [])
            for chain_idx in range(4):
                if chain_idx < len(process_chains):
                    encoded_proc = encode_processes(process_chains[chain_idx], max_steps=20)
                else:
                    encoded_proc = [0]*20
                
                chain_str = ",".join(map(str, encoded_proc))
                data_df.loc[idx, f'process_chain_{chain_idx+1}'] = chain_str

        data_df = data_df.reset_index(drop=True)

        return data_df


    def  train(self):
        pass 

    def predict(self):
        pass 
