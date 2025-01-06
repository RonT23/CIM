'''
    Project     : CAPP LSTM-Based System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM Exercise 3
    Description : This script is a template for infering a CAPP LSTM model and evaluating it.
    Designer    : Ronaldo Tsela
    Date        : 5/1/2025
    Requires    : capp_lstm_package
'''

from capp_lstm_package.capp_lstm_package.capp_lstm import *

if __name__ == "__main__":
    # Create a CAPP object
    capp = CappLSTM()

    # Set directories to the evaluation dataset
    capp.evaluation_dataset_path = "/home/ronaldo/Desktop/CIM/capp-lstm/data/validation.json"
   
    # Set the directory from where the model will be loaded
    capp.model_path = "/home/ronaldo/Desktop/CIM/capp-lstm/metadata/pre-trained-model.keras"

    # Set the directory from where the input data scaler will be loaded
    capp.scaler_path = "/home/ronaldo/Desktop/CIM/capp-lstm/metadata/pre-trained-scaler.ipk"
    
    # Load the model
    capp.load_model()

    # View the model 
    print(capp.model.summary())
    
    # Evaluate the model
    capp.evaluate()

    
    # Display evaluation results
    total_loss = capp.eval_results[0]

    p1_loss = capp.eval_results[1]
    p2_loss = capp.eval_results[2]
    p3_loss = capp.eval_results[3]
    p4_loss = capp.eval_results[4]
    
    p1_acc = capp.eval_results[5]
    p2_acc = capp.eval_results[6]
    p3_acc = capp.eval_results[7]
    p4_acc = capp.eval_results[8]

    print("\n\n")
    print(f"Total Loss : {total_loss}")
    
    print("\n\n")
    print("Evaluation Loss:")
    print(f"\tP1 = {p1_loss}")
    print(f"\tP2 = {p2_loss}")
    print(f"\tP3 = {p3_loss}")
    print(f"\tP4 = {p4_loss}")
    
    print("\n\n")
    print("Evaluation Accuracy:")
    print(f"\tP1 = {p1_acc}")
    print(f"\tP2 = {p2_acc}")
    print(f"\tP3 = {p3_acc}")
    print(f"\tP4 = {p4_acc}")
