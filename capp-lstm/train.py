'''
    Project     : CAPP LSTM-Based System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM Exercise 3
    Description : This script is a template for training the CAPP model constructed
    Designer    : Ronaldo Tsela
    Date        : 5/1/2025
    Requires    : capp_lstm_package
'''

from capp_lstm_package.capp_lstm_package.capp_lstm import *

if __name__ == "__main__":
    # Create a CAPP object
    capp = CappLSTM()

    # Set directories to the datasets
    capp.training_dataset_path   = "/home/ronaldo/Desktop/CIM/capp-lstm/data/training.json"
    capp.validation_dataset_path = "/home/ronaldo/Desktop/CIM/capp-lstm/data/validation.json"
    capp.evaluation_dataset_path = "/home/ronaldo/Desktop/CIM/capp-lstm/data/validation.json"

    # Set the directory where the model will be stored
    capp.model_path = "/home/ronaldo/Desktop/CIM/capp-lstm/metadata/model.keras"

    # Set the directory where the input data scaler will be stored
    capp.scaler_path = "/home/ronaldo/Desktop/CIM/capp-lstm/metadata/scaler.ipk"

    # Define the model characteristics
    capp.c_num_LSTM_cells = 128
    capp.batch_size = 1         
    capp.training_epochs = 10

    # Create the model
    capp.create_model()

    # View the model 
    print(capp.model.summary())
    
    # Train the model
    capp.train()

    # Display training results 
    capp.training_results()
