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
    capp.model_path = "/home/ronaldo/Desktop/CIM/metadata/pre-trained-model.keras"

    # Set the directory from where the input data scaler will be loaded
    capp.scaler_path = "/home/ronaldo/Desktop/CIM/metadata/pre-trained-scaler.ipk"
    
    # Load the model
    capp.load_model()

    # View the model 
    print(capp.model.summary())
    
    # Evaluate the model
    capp.evaluate()

    # Display evaluation results
    p1_loss = capp.eval_results[0]
    p2_loss = capp.eval_results[2]
    p3_loss = capp.eval_results[3]
    p4_loss = capp.eval_results[4]
    
    p1_acc = capp.eval_results[0]
    p2_acc = capp.eval_results[2]
    p3_acc = capp.eval_results[3]
    p4_acc = capp.eval_results[4]
    
    print("Evaluation Loss")
    print(f"P1 = {p1_loss}")
    print(f"P2 = {p2_loss}")
    print(f"P3 = {p3_loss}")
    print(f"P4 = {p4_loss}")
    
    print("Evaluation Accuracy")
    print(f"P1 = {p1_acc}")
    print(f"P2 = {p2_acc}")
    print(f"P3 = {p3_acc}")
    print(f"P4 = {p4_acc}")