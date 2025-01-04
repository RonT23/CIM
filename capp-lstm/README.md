# Computer Aided Process Planing System for Process Chain Generation Using LSTM Networks

## Folder Structure

```bash
    capp-lstm/
        \__ capp_lstm_package/
            
            \__ capp_lstm_package/
                    \__ __init__.py
                    \__ capp_lstm_package.py
            
            \__ README.md
            \__ setup.py

        \__ results
                \__ 
        \__ training.py
        \__ stats.py
        \__ infere.py
        \__ README.md
        
        \__ docs/
                \__ capp-.pdf
```

## Description

## Prerequisites

This package is written in Python, making it largely hardware- and operating system-agnostic, so it should operate similarly across different systems. However, as development and testing were conducted in a Linux-based environment, it is recommend using a Linux environment to run  if possible. Most commands should also work on Windows via Windows Subsystem for Linux (WSL), or by using a virtual machine on a Windows host running one of the many Linux distros available.

The only requirement is Python version 3.5 or higher. Other than that, there are no additional dependencies to use the package except for the additional packages required which will be installed at installation of this package.

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

## Data Samples Provided

## Training the Model

## Model Evaluation

## Prediction

## Results

## Documentation

More details can be found in `capp_lstm.pdf` located in the `./docs/` folder. For further information, please contact me at `rontsela@mail.ntua.gr` or `ron-tsela@di.uoa.gr`.
