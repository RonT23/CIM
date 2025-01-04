# from google.colab import drive
# drive.mount('/content/drive')

import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_FOLDER = "/content/drive/MyDrive/MSc-Automation Control Systems and Robotics - NTUA - 2024_2025/Computer-Integrated-Manufacturing/ex_3/capp_dataset"
TRAINING_SET_JSON = os.path.join(DATA_FOLDER, 'training.json')
VALIDATION_SET_JSON = os.path.join(DATA_FOLDER, 'validation.json')
INPUT_FILE = os.path.join(DATA_FOLDER, 'part_test.json')

PROCESS_CHAIN_COLS = ['process_chain_1', 'process_chain_2', 'process_chain_3', 'process_chain_4']
INPUT_FEATURES_COLS = ['geometry', 'holes', 'external_threads', 'surface_finish', 'tolerance', 'batch_size']

geometry = {
    "pure_axisymmetric"                       : 0,
    "axisymmetric_with_prismatic_features"    : 1,
    "prismatic"                               : 2,
    "prismatic_with_axisymmetric_features"    : 3,
    "prismatic_with_freeform_features"        : 4,
    "freeform"                                : 5,
    "unconventional"                          : 6
}

holes = {
    "none"              : 0,
    "normal"            : 1,
    "normal_threaded"   : 2,
    "normal_functional" : 3,
    "large"             : 4,
    "large_threaded"    : 5,
    "large_functional"  : 6
}

external_threads = {
    "yes"   : 1,
    "no"    : 0
}

surface_finish = {
    "rough"     : 0,
    "normal"    : 1,
    "good"      : 2,
    "very_good" : 3
}

tolerance = {
    "rough"   : 0,
    "normal"  : 1,
    "medium"  : 2,
    "tight"   : 3
}

batch_size = {
    "prototype" : 0,
    "small"     : 1,
    "medium"    : 2,
    "large"     : 3,
    "mass"      : 4
}

processes = {
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

NUM_PROCESSES = len(processes)

"""
  Function to encode the data from the JSON files that describe a part into numerical values.
  @ part_json: the part configuration in JSON format.
  @ return: the encoded part configuration (geometry, holes, external_threads, surface_finish, tolerance, batch_size).
"""
def encode_features(part):
  part_features = part.get('part_features', {})

  part_geometry         = geometry[part_features.get('geometry')]
  part_holes            = holes[part_features.get('holes')]
  part_external_threads = external_threads[part_features.get('external_threads')]
  part_surface_finish   = surface_finish[part_features.get('surface_finish')]
  part_tolerance        = tolerance[part_features.get('tolerance')]
  part_batch_size       = batch_size[part_features.get('batch_size')]

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
def encode_processes(part_processes, max_steps=20):
    encoded = []
    for p in part_processes:
        proc_id = processes.get(p, -1)
        encoded.append(proc_id)
    
    # Truncate if needed
    if len(encoded) > max_steps:
        encoded = encoded[:max_steps]
    
    # Pad if shorter
    while len(encoded) < max_steps:
        encoded.append(0) 

    return encoded  # length = max_steps => shape (20,)




"""
  Function to load the data from the JSON files and preapre it for further processing creating a pandas dataframe.
  @ filename : the path to the JSON file.
  @ return : a pandas dataframe with the preapred data.
"""
def load_training_data(filename):
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

"""
  Convert a comma-separated string of ints into a list of length max_steps.
  (Already guaranteed in load_training_data, but let's parse again to be sure.)
"""
def create_sequence(seq_str, max_steps=20):

    tokens = [t.strip() for t in seq_str.split(",")]
    seq = [int(t) for t in tokens if t != ""]
    # ensure length = max_steps
    seq = seq[:max_steps]
    while len(seq) < max_steps:
        seq.append(0)
    return seq






# Load and prepare training and validation data
training_data = load_training_data(TRAINING_SET_JSON)
validation_data = load_training_data(VALIDATION_SET_JSON)

# create a x_train that has all the input _featrure cols (6) and len(training_data) rows
x_train = np.array(training_data[INPUT_FEATURES_COLS].values, dtype=np.float32)
x_valid = np.array(validation_data[INPUT_FEATURES_COLS].values, dtype=np.float32)

# Parse the 4 chain columns into integer arrays of shape (N, 19)
# We'll store them as integer-coded sequences => shape (N, 19).
# Later, we'll train with sparse_categorical_crossentropy.

def parse_chain_col(df, col):
    seqs = [create_sequence(s, max_steps=20) for s in df[col].values]
    return np.array(seqs, dtype="int32")  # shape (N,20)




y1_train = parse_chain_col(training_data, "process_chain_1")
y2_train = parse_chain_col(training_data, "process_chain_2")
y3_train = parse_chain_col(training_data, "process_chain_3")
y4_train = parse_chain_col(training_data, "process_chain_4")

y1_valid = parse_chain_col(validation_data, "process_chain_1")
y2_valid = parse_chain_col(validation_data, "process_chain_2")
y3_valid = parse_chain_col(validation_data, "process_chain_3")
y4_valid = parse_chain_col(validation_data, "process_chain_4")





# Scale input features
def scale_input_data(x, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        x_scaled = scaler.transform(x)
    return x_scaled, scaler

x_train_scaled, scaler = scale_input_data(x_train, None)
x_valid_scaled, _       = scale_input_data(x_valid, scaler)


print("x_train_scaled shape:", x_train_scaled.shape)
print("y1_train shape:", y1_train.shape, "(integer-coded sequences)")


# We want 4 outputs, each a 20-step sequence of classes in [0..20],
# hence we have 21 possible classes. We'll use sparse_categorical_crossentropy.

NUM_INPUTS       = x_train_scaled.shape[1]
NUM_TIMESTEPS    = 20
NUM_CLASSES      = 21
NUM_OUTPUTS      = 4 

input_layer = keras.Input(shape=(NUM_INPUTS,), name="input_features")

# 1) Project to some dimension
x = layers.Dense(32, activation="relu")(input_layer)

# 2) Repeat so we get (batch_size, 20, 32)
x = layers.RepeatVector(NUM_TIMESTEPS)(x)

# 3) LSTM => output shape (batch_size, 20, hidden_dim)
x = layers.LSTM(128, return_sequences=True)(x)

# 4) We create 4 output "heads", each TimeDistributed over 19 steps, each softmax(21)
p1 = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax"), name="p1")(x)
p2 = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax"), name="p2")(x)
p3 = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax"), name="p3")(x)
p4 = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax"), name="p4")(x)

model = keras.Model(inputs=input_layer, outputs=[p1, p2, p3, p4])

# Using sparse_categorical_crossentropy => each y is shape (batch_size, 19) with integer labels
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]*4
)

model.summary()

############################################
# 4) Train the Model
############################################
# We must pass the labels as a list [y1_train, y2_train, y3_train, y4_train],
# each shaped (N, 19) with integer codes.

model.fit(
    x=x_train_scaled,
    y=[y1_train, y2_train, y3_train, y4_train],
    validation_data=(x_valid_scaled, [y1_valid, y2_valid, y3_valid, y4_valid]),
    epochs=20,
    batch_size=1
)

############################################
# 5) Evaluate
############################################
eval_results = model.evaluate(
    x_valid_scaled,
    [y1_valid, y2_valid, y3_valid, y4_valid],
    batch_size=1
)
# The first element is the overall loss, then 4 loss values, then 4 accuracies.
# e.g. [total_loss, p1_loss, p2_loss, p3_loss, p4_loss, p1_acc, p2_acc, p3_acc, p4_acc]
print("Evaluation results:", eval_results)

############################################
# 6) Inference on a New Part (INPUT_FILE)
############################################
data_df = pd.DataFrame(columns=INPUT_FEATURES_COLS)

with open(INPUT_FILE, 'r') as file:
    data_json_test = json.load(file)

for idx, part in enumerate(data_json_test):
    feats = encode_features(part)
    data_df.loc[idx, INPUT_FEATURES_COLS] = feats

x_test = data_df[INPUT_FEATURES_COLS].values.astype("float32")
x_test_scaled = scaler.transform(x_test)

p1_pred, p2_pred, p3_pred, p4_pred = model.predict(x_test_scaled)

# Each p?_pred is shape (num_test_samples, 19, 31) => distribution over 31 classes for each of 19 steps
# Convert probabilities to integer labels via argmax
rev_processes = {v: k for k, v in processes.items()}

"""
  Converts a sequence of integer process IDs into a list of process names.
  Stops at '0' (padding) or '30' (process_complete).
"""
def prediction_to_text(pred_ids):

    process_chain = []
    for p in pred_ids:
        if p == 0:
            break
        else:
            process_name = rev_processes.get(p, "UNKNOWN")
            process_chain.append(process_name)
    return process_chain

p1_labels = np.argmax(p1_pred, axis=-1)
p2_labels = np.argmax(p2_pred, axis=-1)
p3_labels = np.argmax(p3_pred, axis=-1)
p4_labels = np.argmax(p4_pred, axis=-1)

print("Predicted chain 1 (text):", prediction_to_text(p1_labels[0]))
print("Predicted chain 2 (text):", prediction_to_text(p2_labels[0]))
print("Predicted chain 3 (text):", prediction_to_text(p3_labels[0]))
print("Predicted chain 4 (text):", prediction_to_text(p4_labels[0]))
