'''
    Project     : CAPP LSTM-Based System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM Exercise 3
    Description : This script is a template for infering a CAPP LSTM model and predicting a single input.
    Designer    : Ronaldo Tsela
    Date        : 5/1/2025
    Requires    : capp_lstm_package
'''

from capp_lstm_package.capp_lstm_package.capp_lstm import *

if __name__ == "__main__":
    # Create a CAPP object
    capp = CappLSTM()

    # Set directories to the input file
    capp.evaluation_dataset_path = "/home/ronaldo/Desktop/CIM/capp-lstm/data/part_test.json"
   
    # Set the directory from where the model will be loaded
    capp.model_path = "/home/ronaldo/Desktop/CIM/metadata/pre-trained-model.keras"

    # Set the directory from where the input data scaler will be loaded
    capp.scaler_path = "/home/ronaldo/Desktop/CIM/metadata/pre-trained-scaler.ipk"
    
    # Load the model
    capp.load_model()
    
    # Generate the process chains
    results = capp.predict()

    print(f"Process Chain 1: {'->'.join(results[0])}")
    print(f"Process Chain 2: {'->'.join(results[1])}")
    print(f"Process Chain 3: {'->'.join(results[2])}")
    print(f"Process Chain 4: {'->'.join(results[3])}")
