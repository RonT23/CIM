"""
    Project     : CAPP LSTM-Based System - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM Exercise 3
    Description : Implementation of a CAPP system using an LSTM-based neural network architecture for the prediction 
                  of up to four equivalent process chains.
    Designer    : Ronaldo Tsela
    Date        : 4/1/2025
    Dependencies: numpy, sklearn, tensorflow, pandas, joblib, json, matplotlib
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt

"""
  This class implements the Computer Aided Process Planning (CAPP) LSTM-based system for generating
  process chains for part manufacturing. 
        
       # Set default values for the training, validation, testing and single input files
  
  # These will work as lookup tables for the input and output. These are provided by "../../data/part_encoding_and_process_list.json". 
        # Currently these are hardcoded. It would have been better if this class could configure itself dynamically from an external file
        
  Attributes:
    c_part_geometry (dict): Look-up table for part geometry. One-hot integer encoding
    c_part_holes (dict): Look-up table for part holes. One-hot integer encoding.
    c_part_external_threads (dict): Look-up table for external threads. One-hot integer encoding.
    c_part_surface_finish (dict): Look-up table for surface finish. One-hot integer encoding.
    c_part_tolerance (dict): Look-up table for tolerance. One-hot integer encoding.
    c_part_batch_size (dict): Look-up table for batch size. One-hot integer encoding.
    c_available_processes (dict): Look-up table for available processes. One-hot integer encoding.
    c_n_processes (int): Number of available processes.
    c_max_steps (int): The number of timesteps to predict. This equals the number of unique available processes.
    c_feature_columns (list): The features used as input.
    training_dataset_path (str): The path to the JSON dataset used for training.
    validation_dataset_path (str): The path to the JSON dataset used for validation.
    evaluation_dataset_path (str): The path to the JSON dataset used for evaluation.
    input_file_path (str): The path to the input JSON file for a single part process chain prediction.
    scaler_path (str): The path to the input features scaller. When new instance is created the scares is overwritten.
    model_path (str): The path to the model. When new instance is created the model is overwritten.
    eval_results (list) : The results from the evaluation. Results include loss and accuracy per head.
    history : The training history.  
    model : The model instance.   
    scaler : The input scaler instance.    

  Methods:
    __init__():
      Initializes a new CappLSTM object instance.
    
    encode_features(part: (str)):        
      Encodes the data from the JSON files that describe a part into numerical values.

      Args:
        part (str): The part configuration in JSON format.

      Returns:
        (tuple): A tuple containing the encoded part configuration.

    encode_processes(part_processes: (list)):
      Encodes a list of process names into numerical IDs based on available processes.

      Args:
        part_processes (list): A list of str process names representing a process chain.

      Returns:
          list: A list of numerical process IDs representing the encoded process chain, 
                with a fixed length of `c_max_steps`.

    decode_processes(pred_ids, rev_processes):
      Decodes a list of predicted numerical process IDs back into human-readable process names.
      
      Args:
          pred_ids (list): A list of numerical process IDs predicted by the model.
          rev_processes (dict): A dictionary mapping process IDs back to process names.

      Returns:
          (list): A list of process names corresponding to the decoded process chain.
                  If no process chain is found an empty list is returned.

    read_json(filename: (str)):
      Function to read a JSON file and return its contents as a dict.

      Args:
        filename (str): The path to the JSON file to be read.

      Returns:
        (dict): The contents of the JSON file as a dict object.

    load_dataset(filename: (str)):
      Function to load the data from the JSON files and preapre it for further processing creating a pandas dataframe.

      Args:
        filename (str): The path to the JSON file.

      Returns:
        (DataFrame): A pandas dataframe with the preapred data.
    
    create_sequence(seq_str: (str)):
      Converts a sequence string into a list of integers representing a numerical sequence
      with a fixed length by padding with zeros if necessary.

      Args:
          seq_str (str): A comma-separated string of integers representing the sequence.

      Returns:
          (list): A list of integers representing the sequence, padded with zeros to a fixed length.

    
    parse_chain_column(data: (DataFrame), col: (str)):
      Parses a specific column from the input data into sequences of integers and formats them as arrays.

      Args:
          data (pandas.DataFrame): The input data containing multiple rows.
          col (str): The column name containing the sequence strings to be parsed.

      Returns:
          (np.ndarray): A 2D NumPy array where each row is a parsed sequence with a fixed length.

    
    scale_input_data(x: (DataFrame), scaler=None):
          Scales the input data using the provided scaler or the internal scaler.
    If no scaler is provided, fits and transforms the data using the internal scaler.

    Args:
        x (np.ndarray): The input feature data to be scaled.
        scaler (sklearn.preprocessing or None, optional): An external scaler object for transforming the data.
            If None, the internal scaler will be used.

    Returns:
        (np.ndarray, scaller_object): A tupple containing the scaled input data and the scaler object used 
                                      for the transformation.

    load_model():
      Loads a pre-trained Keras model and a saved scaler for input transformation.
      Attempts to load the model and scaler from the specified file paths. If either 
      the model or the scaler is not found, it prints an error message and exits the program.

    create_model():
      Creates a new Keras model architecture with an input layer, a dense hidden layer, 
      an LSTM layer, and four multi-headed TimeDistributed output layers. The model is compiled 
      with the Adam optimizer and uses sparse categorical cross-entropy as the loss function.
    
    train():
      Trains the Keras model on the provided training and validation datasets.
      The input data is scaled using a standard scaler, and output sequences are parsed
      from specific columns in the dataset. The training history is stored for plotting 
      and evaluation purposes.

    training_results():
      Plots the training and validation loss and accuracy for each output head of the model.
      Also prints the final validation accuracy for each output head.
    
    evaluate():
      Evaluates the model on the evaluation dataset to compute loss and accuracy.
     
    predict():
      Performs predictions on input data loaded from a JSON file.
      
      Returns:
          (tuple): Four lists containing decoded process chain labels for each of the four outputs.
                   Labels are converted into human-readable strings instead fo numbers.


"""
class CappLSTM:
    def __init__(self):
        
        self.c_part_geometry = {
            "pure_axisymmetric"                    : 0,
            "axisymmetric_with_prismatic_features" : 1,
            "prismatic"                            : 2,
            "prismatic_with_axisymmetric_features" : 3,
            "prismatic_with_freeform_features"     : 4,
            "freeform"                             : 5,
            "unconventional"                       : 6
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

        self.c_n_processes      = len(self.c_available_processes)
        self.c_max_steps        = self.c_n_processes
        self.c_feature_columns  = ['geometry', 
                                   'holes', 
                                   'external_threads', 
                                   'surface_finish', 
                                   'tolerance', 
                                   'batch_size']

        self.training_dataset_path   = "../../data/training.json"
        self.validation_dataset_path = "../../data/validation.json"
        self.evaluation_dataset_path = "../../data/test.json"
        
        self.input_file_path   = "../../data/input.json"
        self.output_file_path  = "../../data/output.json"

        self.scaler_path  = "../../metadata/scaler.ipk"
        self.model_path   = "../../metadata/model.keras"

        self.c_num_inputs         = len(self.c_feature_columns)
        self.c_num_LSTM_cells     = 16
        self.c_hidden_layer_size  = 32

        self.batch_size       = 1
        self.training_epochs  = 10
        
        self.eval_results = None
        self.history      = None
        self.model        = None
        self.scaler       = None


############# DATA PROCESSING FUNCTIONS ########################
    def encode_features(self, part):
        part_features = part.get('part_features', {})

        # Extract the configuration data from the input and encode it based on the lookup tables
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

    def encode_processes(self, part_processes):
      encoded = []
      for p in part_processes:
        proc_id = self.c_available_processes.get(p, -1)
        encoded.append(proc_id)
        
      # Truncate if paths are longer than 20 steps
      if len(encoded) > self.c_max_steps:
        encoded = encoded[:self.c_max_steps]
        
      # Pad if shorter than 20 steps. Here index 0 implies no operation
      while len(encoded) < self.c_max_steps:
        encoded.append(0) 

      return encoded  # returned shape is (20,)

    def decode_processes(self, pred_ids, rev_processes):
        process_chain = []
        for p in pred_ids:
            if p == 0:
                break
            else:
                process_name = rev_processes.get(p, "UNKNOWN")
                process_chain.append(process_name)
        return process_chain
        
    def read_json(self, filename):
      with open(filename, 'r') as file:
        data_json = json.load(file)
      return data_json

    def load_dataset(self, filename):
      data_df = pd.DataFrame(columns=self.c_feature_columns)

      data_json = self.read_json(filename)

      for idx, part in enumerate(data_json):

        # Encode the part features
        features = self.encode_features(part)
        data_df.loc[idx, self.c_feature_columns] = features

        # Extract the process chains
        process_chains = part.get('process_chains', [])
        for chain_idx in range(4):

            # Encode the process chains or pad if empty
            if chain_idx < len(process_chains):
              encoded_proc = self.encode_processes(process_chains[chain_idx])
            else:
              # 0 implies no process, thus zero vector implies no process chain
              encoded_proc = [0] * self.c_max_steps
            
            chain_str = ",".join(map(str, encoded_proc))
            data_df.loc[idx, f'process_chain_{chain_idx+1}'] = chain_str

        data_df = data_df.reset_index(drop=True)

      return data_df

    def create_sequence(self, seq_str):
      # Get tokens from the sequence string
      tokens = [t.strip() for t in seq_str.split(",")]
      sequence = [int(t) for t in tokens if t != ""]

      # Ensure proper length
      sequence = sequence[:self.c_max_steps]
      while len(sequence) < self.c_max_steps:
        sequence.append(0)

      return sequence

    def parse_chain_column(self, data, col):
      sequences = [self.create_sequence(s) for s in data[col].values]
      return np.array(sequences, dtype="int32")  # returned shape (N, 20)

    def scale_input_data(self, x, scaler=None): 
      if scaler is None: # Fit the newly created scaller
        x_scaled = self.scaler.fit_transform(x)
      else: # Use the already created scaler
        x_scaled = scaler.transform(x)
      return x_scaled, self.scaler

############# END OF DATA PROCESSING FUNCTIONS #################

############ MODE MANAGEMENT FUNCTIONS #########################

    # Inference of an already existing model and metadata
    def load_model(self):
      try: # load model
        self.model = keras.models.load_model(self.model_path)
      except:
        print(f"[ERROR] No Keras models found in {self.model_path}.")
        exit(-1)

      try: # load input scaler
        self.scaler = joblib.load(self.scaler_path)
      except:
        print(f"[ERROR] No Keras scaler found in {self.scaller_path}.")
        exit(-1)

    def create_model(self):   
      # Input layer
      input_layer = keras.Input(shape=(self.c_num_inputs,), name="input layer")
      
      # Fully connected hidden layer
      x = keras.layers.Dense(self.c_hidden_layer_size, activation="relu")(input_layer)
      
      # Repeat the input for each time step
      x = keras.layers.RepeatVector(self.c_max_steps)(x)

      # LSTM layer for sequence generation
      x = keras.layers.LSTM(self.c_num_LSTM_cells, return_sequences=True)(x)

      # Multiheaded output
      p1 = keras.layers.TimeDistributed(keras.layers.Dense(self.c_max_steps + 1, activation="softmax"), name="p1")(x)
      p2 = keras.layers.TimeDistributed(keras.layers.Dense(self.c_max_steps + 1, activation="softmax"), name="p2")(x)
      p3 = keras.layers.TimeDistributed(keras.layers.Dense(self.c_max_steps + 1, activation="softmax"), name="p3")(x)
      p4 = keras.layers.TimeDistributed(keras.layers.Dense(self.c_max_steps + 1, activation="softmax"), name="p4")(x)

      self.model = keras.Model(inputs = input_layer,  outputs = [p1, p2, p3, p4] )

      self.model.compile(
          optimizer = keras.optimizers.Adam(1e-3),
          loss      = "sparse_categorical_crossentropy",
          metrics   = ["accuracy"] * 4
      )

    def train(self):
      
      # Load the training dataset
      training_data   = self.load_dataset(self.training_dataset_path)
      validation_data = self.load_dataset(self.validation_dataset_path)

      # Create a new scaler
      self.scaler = StandardScaler()

      # Create and scale the input data
      x_train = np.array(training_data[self.c_feature_columns].values, dtype=np.float32)
      x_train, self.scaler = self.scale_input_data(x_train)

      x_valid = np.array(validation_data[self.c_feature_columns].values, dtype=np.float32)
      x_valid, _ = self.scale_input_data(x_valid, self.scaler)

      # Create the output sequences
      y1_train = self.parse_chain_column(training_data, "process_chain_1")
      y2_train = self.parse_chain_column(training_data, "process_chain_2")
      y3_train = self.parse_chain_column(training_data, "process_chain_3")
      y4_train = self.parse_chain_column(training_data, "process_chain_4")

      y1_valid = self.parse_chain_column(validation_data, "process_chain_1")
      y2_valid = self.parse_chain_column(validation_data, "process_chain_2")
      y3_valid = self.parse_chain_column(validation_data, "process_chain_3")
      y4_valid = self.parse_chain_column(validation_data, "process_chain_4")

      # Train the model with the given data samples
      self.history = self.model.fit(
                  x               = x_train,
                  y               = [ y1_train, y2_train, y3_train, y4_train ],
                  validation_data = ( x_valid, [y1_valid, y2_valid, y3_valid, y4_valid]),
                  epochs          = self.training_epochs,
                  batch_size      = self.batch_size
      )

      # Export the model and metadata
      self.model.save(self.model_path)
      joblib.dump(self.scaler, self.scaler_path)

    def training_results(self):
      history_dict = self.history.history
      
      train_loss = history_dict['loss'] 
      val_loss = history_dict['val_loss']
      
      # Plot the total training and validation loss
      plt.figure(figsize=(10, 5))
      plt.plot(train_loss, label='Train Loss')
      plt.plot(val_loss, label='Val Loss')
      plt.title('Total Model Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.grid(True)

      plt.show()
      
      # Retrieve the loss and accuracy per output head
      p1_val_loss = history_dict.get('val_p1_loss', None) 
      p2_val_loss = history_dict.get('val_p2_loss', None) 
      p3_val_loss = history_dict.get('val_p3_loss', None) 
      p4_val_loss = history_dict.get('val_p4_loss', None) 

      p1_val_acc = history_dict.get('val_p1_accuracy', None)  
      p2_val_acc = history_dict.get('val_p2_accuracy', None)  
      p3_val_acc = history_dict.get('val_p3_accuracy', None)  
      p4_val_acc = history_dict.get('val_p4_accuracy', None)  

      # Plot the validation loss per output head
      plt.figure(figsize=(10, 5))
      plt.plot(p1_val_loss, label='P1 Validation Loss')
      plt.plot(p2_val_loss, label='P2 Validation Loss')
      plt.plot(p3_val_loss, label='P3 Validation Loss')
      plt.plot(p4_val_loss, label='P2 Validation Loss')
      plt.title('Per-Output Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Value')
      plt.legend()
      plt.grid(True)
      plt.show()

      # Plot the validation accuracy per output head
      plt.figure(figsize=(10, 5))
      plt.plot(p1_val_acc, label='P1 Validation Accuracy')
      plt.plot(p2_val_acc, label='P2 Validation Accuracy')
      plt.plot(p3_val_acc, label='P3 Validation Accuracy')
      plt.plot(p4_val_acc, label='P2 Validation Accuracy')
      plt.title('Per-Output Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Value')
      plt.legend()
      plt.grid(True)
      plt.show()

      print('============================')
      print('\nFinal Validation Accuracy:')
      print(f'P1  : {p1_val_acc[-1]}')
      print(f'P2  : {p2_val_acc[-1]}')
      print(f'P3  : {p3_val_acc[-1]}')
      print(f'P4  : {p4_val_acc[-1]}')
      print(f'Avg : {(p1_val_acc[-1] + p2_val_acc[-1] + p3_val_acc[-1] + p4_val_acc[-1]) / 4.0}')
      print('============================')

    def evaluate(self): 
      # Load evaluation dataset
      evaluation_data = self.load_dataset(self.evaluation_dataset_path)
      
      # Transform evaluation input data
      x_eval = np.array(evaluation_data[self.c_feature_columns].values, dtype=np.float32)
      x_eval, _ = self.scale_input_data(x_eval, self.scaler)

      # Reform the expected output
      y1_eval = self.parse_chain_column(evaluation_data, "process_chain_1")
      y2_eval = self.parse_chain_column(evaluation_data, "process_chain_2")
      y3_eval = self.parse_chain_column(evaluation_data, "process_chain_3")
      y4_eval = self.parse_chain_column(evaluation_data, "process_chain_4")

      self.eval_results = self.model.evaluate(
          x_eval,
          [y1_eval, y2_eval, y3_eval, y4_eval],
          batch_size = 1 
      )

    def predict(self):
      input_data = pd.DataFrame(columns=self.c_feature_columns)
      data_json = self.read_json(self.input_file_path)
      
      for idx, part in enumerate(data_json):
        features = self.encode_features(part)
        input_data.loc[idx, self.c_feature_columns] = features
      
      x = input_data[self.c_feature_columns].values.astype("float32")
      x, _ = self.scale_input_data(x, self.scaler)
      
      p1_pred, p2_pred, p3_pred, p4_pred = self.model.predict(x)

      p1_labels = np.argmax(p1_pred, axis=-1)
      p2_labels = np.argmax(p2_pred, axis=-1)
      p3_labels = np.argmax(p3_pred, axis=-1)
      p4_labels = np.argmax(p4_pred, axis=-1)

      rev_processes = {v: k for k, v in self.c_available_processes.items()}

      return ( 
                self.decode_processes(p1_labels[0], rev_processes), 
                self.decode_processes(p2_labels[0], rev_processes), 
                self.decode_processes(p3_labels[0], rev_processes), 
                self.decode_processes(p4_labels[0], rev_processes)
            )
################################################################