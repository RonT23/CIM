# Computer Aided Process Planing System for Process Chain Generation Using LSTM Networks

## Folder Structure

```bash
    capp-lstm/
        \__ capp_lstm_package/
                \__ capp_lstm_package/
                        \__ __init__.py
                        \__ capp_lstm.py
            
                \__ README.md
                \__ setup.py
        
        \__ data
                \__ expected.json
                \__ part_encoding_and_process_list.json
                \__ part_test.json
                \__ training.json
                \__ validation.json

        \__ metadata
                \__ pre-trained-model.keras
                \__ pre-trained-scaler.ipk

        \__ results
                \__ learning_curves_loss.jpg
                \__ learning_curves_acc.jpg
                \__ learning_curves_total.jpg

        \__ evaluate.py
        \__ predict.py
        \__ train.py
        
        \__ docs/
                \__ capp_lst.pdf
        
        \__ README.md
```

## Description

This repository contains a Computer-Aided Process Planning (CAPP) system built using a Long Short-Term Memory (LSTM) neural network in TensorFlow Keras with Python. The goal of this system is to automatically generate up to four different process chains for manufacturing a given part.

Each part is described by a set of attributes (e.g., geometry, holes, tolerance). These attributes are provided in a JSON-formatted file (see `./data/part_encoding_and_process_list.json`). The system encodes these attributes and processes them to predict the corresponding manufacturing process chains.

The core of the predictor is an LSTM network, which is one of the most effective models for sequence modeling. Since process chain generation depends heavily on sequential dependencies (i.e., the order of processes matters), LSTM is an ideal choice for this task.

The system produces four output process chains, each represented by 4 heads, where each head generates 20 time steps (i.e., a sequence of 20 values). The values are integers in the range [0, 20] where values 1 to 20 represent manufacturing processes and value 0 represents "no operation" (used for padding or when fewer than 20 steps are needed). The predicted sequences are then converted into human-readable strings.

In order for this system to work we made an assumption: Each process can appear at most once in a single process chain. Therefore, the maximum number of processes per chain is 20.

In this repository we provide scripts and data for training, validating, testing, and inferring the model. A pre-trained model and scaler are also included.

## Prerequisites

This package is written in Python, making it largely hardware- and operating system-agnostic, so it should operate similarly across different systems. However, as development and testing were conducted in a Linux-based environment, it is recommend using a Linux environment to run  if possible. Most commands should also work on Windows via Windows Subsystem for Linux (WSL), or by using a virtual machine on a Windows host running one of the many Linux distros available.

The only requirement is Python version 3.7 or higher (for TensorFlow compatibility). Other than that, there are no additional dependencies to use the package except for the additional packages required which will be installed at installation of this package.

## Create Virtual Environment

To keep this package separate from other Python packages on your system, it is best practice to use a virtual environment. To create and activate a virtual environment run:

```bash
   $ python3 -m venv capp_lstm
   $ source ./capp_lstm/bin/activate
```

## Package Installation 

Navigate to the `./capp_lstm_package` directory and run the following command to install the package locally on your computer (within the virtual environment you created).

```bash
    $ pip3 install -e .
```

## Data Samples

The `./data` directory contains the training, validation, and testing data samples, all formatted in JSON. Below is a detailed description of each file:

`part_encoding_and_process_list.json:` Contains the part attribute encodings (e.g., geometry, holes, batch size) and the process list used in this project to describe parts and available manufacturing processes.

`training.json:` Contains multiple part entries used to train the LSTM model. Each entry includes the part's attributes and the corresponding process chains.

`validation.json`: Contains multiple part entries used for validating the model during training to monitor performance and detect overfitting.

`part_test.json`: A single part configuration selected from the validation set, provided as a test input to demonstrate model predictions in inference mode.

`expected.json`: The expected output process chains that the model should generate when using `part_test.json` as input. This serves as a reference for evaluating the model's prediction capabilities in inference mode.

## Metadata

The `./metadata` folder contains a pre-trained model (`pre-trained-model.keras`) and a pre-trained scaler (`pre-trained-scaler.ipk`). These are used to generate the results described in the report.

You can run inference directly using these pre-trained components without needing to retrain the CAPP-LSTM model. Therefore to predict process chains for a part configuration description run the provided `./predict.py` script using the sample input file `part_test.json` from the `./data` directory as input. This will output the predicted process chains which you can evaluate with the `expected.json` provided in `./data` as well.

## Training

If you want to retrain the model (e.g., to improve performance or use new data), you can do so using the `CappLSTM.train()` function provided by the API. To see an example of how to  define system parameters (e.g., hidden layer size, number of epochs) and retrain the model using the training and validation datasets of your choise refer to the provided `./train.py` script.


## Evaluation

You can evaluate the model's performance using either the custom-trained or pre-trained model. This is done by running the `CappLSTM.evaluate()` function, which takes a new evaluation dataset as input. To see an example, refer to the `./evaluate.py` script.

## Results

The `./results` folder contains the learning curves for the pre-trained model. The following plots are available:

* `learning_curves_loss.jpg`: Shows the training and validation loss per epoch.
* `learning_curves_acc.jpg`: Shows the training and validation accuracy per epoch.
* `learning_curves_total.jpg`: Displays the overall performance summary in means of accuracy and loss.

The total accuracy for each of the model’s output heads is computed as the average accuracy across all four heads in the final training epoch. Specifically the model showcased here has 64 LSTM cells and trained for 10 epochs. The average accuracy achieved among all heads is 97.8%. The accuracy per head is as follows:

* P1: 95.9%
* P2: 97.6%
* P3: 99.9%
* P4: 97.8%

## Documentation

More details can be found in `capp_lstm.pdf` located in the `./docs/` folder. For further information, please contact me at `rontsela@mail.ntua.gr` or `ron-tsela@di.uoa.gr`.
