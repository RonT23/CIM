# Petri-Net Simulation

## Directory Structure 

## Description

## Package Installation 

Go to `./petri_net_sim_package` and run the following command to install the package locally on your computer.

```bash
    $ pip3 install -e .
```

## Configure the Network

Open the `run.py` Python script. This is a template that sets up the simulator. Configure the target Petri network to simulate in the section that starts with `##### Define here your Petri network ####` and terminates with `##### End of network definitions     ####`. Do not change the rest of the code. Follow the following configuration guidelines:

1. Set-Up the Places

Use the `net.add_place(place_name, initial_tokes)` to add a place node to the graph.

2. Set-Up the Transitions

Use the `net.add_transition(transition_name)` to add a transition node to the graph.

3. Set the Arcs

Use the `net.add_input_arc(transition_name, place_name, weight)` to add an input arc from the specified place to the specified transition with the specified weight. Use `net.add_output_arc(transition_name, place_name, weight)` to add an output arc from the specified transition to the specified place with the specified weight. Use `net.add_inhibitory_arc(transition_name, place_name)` to add an inhibitory arc from the specified place node to the specified transition node. 

    target_transition_name = "T3"
    total_transition_activations = 6

4. Set the Termination Condition

Set the value of `target_transition_name` equal to the target transition node and `total_transition_activations` equal to the number of total activations of the specified target trasition. These will operate as simulation termination conditions. 

## Run Simulation

To run the simulation execute the program as follows:
```bash
    $ python3 run.py
```

### Output

The output of the simulator is two `.json` files: one that describes the networks structure and another one that contains the simulation results. These files are then parsed as input to the visualization script which subsequently produces a simple graphical representation of the changes occured on the Petri network for all the steps passed. 

## Documentation

More details can be found in `petri_net_sim_report.pdf` in `./Docs/`. For more information please contact me at: `rontsela@mail.ntua.gr` or `ron-tsela@di.uoa.gr`.
