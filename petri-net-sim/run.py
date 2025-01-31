'''
    Project     : Petri Network Simulator - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 1
    Description : Template simulation script for creating and simulating a simple Petri network using the petri_net_sim package created.
    Designer    : Ronaldo Tsela
    Date        : 30/10/2024
    Requires    : petri_net_sim
'''
from petri_net_sim_package.petri_net_sim import *

if __name__ == "__main__" :
    
    net = PetriNet()

##### Define here your Petri network ####

    # 1. Define the places
    net.add_place("P1", tokens=1)
    net.add_place("P2", tokens=2)
    net.add_place("P3", tokens=0)
    net.add_place("P4", tokens=0)

    # 2. Define the transitions
    net.add_transition("T1")
    net.add_transition("T2")
    net.add_transition("T3")
    
    # 3. Define the arcs per transition
    net.add_input_arc("T1", "P1", weight=1)
    net.add_output_arc("T1", "P3", weight=1)

    net.add_input_arc("T2", "P2", weight=1)
    net.add_output_arc("T2", "P4", weight=1)

    net.add_input_arc("T3", "P3")
    net.add_input_arc("T3", "P4", weight=1)
    net.add_output_arc("T3", "P1", weight=1)
    net.add_output_arc("T3", "P2", weight=1)

    # 4. Define termination conditions
    target_transition_name = "T3"
    total_transition_activations = 6

##### End of network definitions     ####

##### Don't touch these

    net.export_structure("structure.json")
    net.simulate(target_transition_name=target_transition_name, total_transition_activations=total_transition_activations)
    json_to_txt("simulation_log.json", "simulation_results.txt")
    print_txt_to_console("simulation_log.txt")
    visualize_petri_net("structure.json", "simulation_log.json")
    
#### End of program
