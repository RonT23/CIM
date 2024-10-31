# Petri-Net Simulation

## Folder Structure

```bash
    petri-net-sim/
        
        \__ petri_net_sim_package/
                
                \__ petri_net_sim_package/
                        \__ __init__.py
                        \__ petri_net_sim.py
                
                \__ README.md
                \__ setup.py

        \__ results/
                \__ net_1_simulation_log.json
                \__ net_1_simulation_log.txt
                \__ net_1_structure.json
                \__ net_2_simulation_log.json
                \__ net_2_simulation_log.txt
                \__ net_2_structure.json
                \__ net_3_simulation_log.json
                \__ net_3_simulation_log.txt
                \__ net_3_structure.json
                
        \__ simulations/
                \__ net_1.py
                \__ net_2.py
                \__ net_3.py
        
        \__ README.md
        \__ run.py
        \__ docs/
                \__ petri_net_sim_report.pdf

```

## Description

The Petri Net simulator is a simple Python package for simulating Petri networks. This simulator can also model inhibitor arcs, a feature not defined in classical Petri nets. Additionally, the package includes a module for visualizing the constructed graph. Results exported from the simulator are easy to integrate with other programs, as they use JSON format, which widely adopted also for graph descriptions and simulation results as well as simple comma separated TXT files.

## Prerequisites

This package is written in Python, making it largely hardware- and operating system-agnostic, so it should operate similarly across different systems. However, as development and testing were conducted in a Linux-based environment, it is recommend using a Linux environment to run the simulator. Most commands should also work on Windows via Windows Subsystem for Linux (WSL), or by using a virtual machine on a Windows host running one of the many Linux distros available.

The only requirement is Python version 3.5 or higher. Other than that, there are no additional dependencies to run the simulator. The required packages are installed if they are not already istalled through the package itself.

## Create Virtual Environment

To keep this package separate from other Python packages on your system, it is best practice to use a virtual environment. To create and activate a virtual environment, follow these steps inside the project folder.

```bash
   $ python3 -m venv petri_net_env
   $ source ./petri_net_env/bin/activate
```

## Package Installation 

Navigate to the `./petri_net_sim_package` directory and run the following command to install the package locally on your computer (within the virtual environment you created).

```bash
    $ pip3 install -e .
```

## Define a New Petri Network

Open the `run.py` Python script, which serves as a template for setting up the simulator. Configure the target Petri network for simulation in the section beginning with `##### Define here your Petri network ####` and ending with `##### End of network definitions     ####`. Avoid modifying other parts of the code. To model the Petri network for simulation follow these steps:

1. Set Up the Places

Use `net.add_place(place_name, initial_tokens)` to add a place node to the graph. Place names are typically assigned names as `Pi`, where `i = 1, 2, 3, ... n` for `n` places. If the initial token count is not specified for a place, it defaults to 0.

2. Set Up the Transitions

Use `net.add_transition(transition_name)` to add a transition node to the graph. Transitions are named usually with `Ti`, where `i = 1,2, 3, ... m`, for `m` transitions.

3. Set Up the Arcs

The simulator supports three types of arcs that connect places and transitions: input, output, and inhibitor.

* Use the `net.add_input_arc(transition_name, place_name, weight)` to add an input arc from the specified place to the specified transition with the given weight. If the weight is not defined then the arc will be assigned unit weight. 

* Use `net.add_output_arc(transition_name, place_name, weight)` to add an output arc from the specified transition to the specified place with the specified weight. If the weight is not defined then the arc will be assigned unit weight.

* Use `net.add_inhibitory_arc(transition_name, place_name)` to add an inhibitory arc from the specified place node to the specified transition node. 

4. Set Up the Termination Condition

Assign `target_transition_name` to the desired target transition node and `total_transition_activations` to the total number of activations for that transition. These values will serve as the simulation's termination conditions. Specifically the simulation will run for as many steps as required for either the target transition node is fired for the total transition activation parameter defined or a deadlock occures. In both situations the simulation will terminate.

## Run Simulations

To run the newly configured simulation, simply execute the program as follows:

```bash
    $ python3 run.py
```

In folder `./simulations/` you will find three ready-to-execute scripts that model three Petri networks defined in the requirements of this task. 

### Simulation Output

The simulator upon execution outputs two JSON files: one describes the network's structure, while the other contains the simulation results. These files are then parsed as input to the visualization script, which produces a graphical representation of the changes that occurred in the Petri network throughout all the simulation steps. Each step is graphically visualized with a timestep in between steps set to 1 second. Also the content of the simulation results JSON file is extracted into a more intuitive and easy to work with TXT file. You can find such files in `./results/` folder produced from the three simulation files respectivelly.  

## Documentation

More details can be found in `petri_net_sim_report.pdf` located in the `./docs/` folder. For further information, please contact me at `rontsela@mail.ntua.gr` or `ron-tsela@di.uoa.gr`.
