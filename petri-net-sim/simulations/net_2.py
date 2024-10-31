'''
    Project     : Petri Network Simulator - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 1
    Description : Simulation script for the given network 2 as defined in the requirements PDF file.
    Designer    : Ronaldo Tsela
    Date        : 30/10/2024
    Requires    : petri_net_sim
'''
from petri_net_sim_package.petri_net_sim_package.petri_net_sim import *

if __name__ == "__main__" :
    
    net = PetriNet()

##### Define here your Petri network ####

    net.add_place("p1",   tokens=1)
    net.add_place("p2",   tokens=0)
    net.add_place("p3",   tokens=0)
    net.add_place("p4",   tokens=0)
    net.add_place("p5",   tokens=0)
    net.add_place("pd",   tokens=0)
    net.add_place("IN",   tokens=1)
    net.add_place("^IN",  tokens=0)
    net.add_place("WIP",  tokens=0)
    net.add_place("^WIP", tokens=1)
    net.add_place("p11",  tokens=1)
    net.add_place("p12",  tokens=0)
    net.add_place("p13",  tokens=0)
    net.add_place("p14",  tokens=0)
    net.add_place("p15",  tokens=0)
    net.add_place("FP",   tokens=0)
    net.add_place("^FP",  tokens=5)
    net.add_place("R",    tokens=1)

    net.add_transition("t1")
    net.add_transition("t2")
    net.add_transition("t3")
    net.add_transition("t4")
    net.add_transition("t5")
    net.add_transition("td")
    net.add_transition("t6")
    net.add_transition("t11")
    net.add_transition("t12")
    net.add_transition("t13")
    net.add_transition("t14")
    net.add_transition("t15")
    net.add_transition("t16")

    # transition t1
    net.add_input_arc("t1", "p1", weight=1)
    net.add_input_arc("t1", "IN", weight=1)
    net.add_input_arc("t1", "R", weight=1)
    net.add_output_arc("t1", "p2", weight=1)
    net.add_output_arc("t1", "^IN", weight=1)

    # transition t2    
    net.add_input_arc("t2", "p2", weight=1)
    net.add_output_arc("t2", "p3", weight=1)
    net.add_output_arc("t1", "R", weight=1)
    
    # transition t3    
    net.add_input_arc("t3", "p3", weight=1)
    net.add_output_arc("t3", "p4", weight=1)

    # transition t4
    net.add_input_arc("t4", "p4", weight=1)
    net.add_input_arc("t4", "R", weight=1)
    net.add_input_arc("t4", "^WIP", weight=1)
    net.add_output_arc("t4", "p5", weight=1)
    net.add_output_arc("t4", "pd", weight=1)
    
    # transition td
    net.add_input_arc("td", "pd", weight=1)
    net.add_output_arc("td", "^WIP", weight=1)

    # transition t5
    net.add_input_arc("t5", "p5", weight=1)
    net.add_input_arc("t5", "^WIP", weight=1)
    net.add_output_arc("t5", "p1", weight=1)
    net.add_output_arc("t5", "WIP", weight=1)
    net.add_output_arc("t5", "R", weight=1)

    # transition t6
    net.add_input_arc("t6", "^IN", weight=1)
    net.add_output_arc("t6", "IN", weight=1)

    # transition t11
    net.add_input_arc("t11", "p11", weight=1)
    net.add_input_arc("t11", "WIP", weight=1)
    net.add_input_arc("t11", "R", weight=1)
    net.add_output_arc("t11", "p12", weight=1)
    net.add_output_arc("t11", "^WIP", weight=1)

    # transition t12
    net.add_input_arc("t12", "p12", weight=1)
    net.add_output_arc("t12", "p13", weight=1)
    net.add_output_arc("t12", "R", weight=1)

    # transition t13
    net.add_input_arc("t13", "p13", weight=1)
    net.add_output_arc("t13", "p14", weight=1)
    
    # transition t14
    net.add_input_arc("t14", "p14", weight=1)
    net.add_input_arc("t14", "R", weight=1)
    net.add_output_arc("t14", "p15", weight=1)

    # transition t15
    net.add_input_arc("t15", "p15", weight=1)
    net.add_input_arc("t15", "^FP", weight=1)
    net.add_output_arc("t15", "FP", weight=1)
    net.add_output_arc("t15", "R", weight=1)
    net.add_output_arc("t15", "p11", weight=1)

    # transition t16
    net.add_input_arc("t16", "FP", weight=1)
    net.add_output_arc("t16", "^FP", weight=1)

    target_transition_name = "t16"
    total_transition_activations = 5

##### End of network definitions     ####

##### Don't touch these

    net.export_structure("../results/net_2_structure.json")
    net.simulate(target_transition_name=target_transition_name, total_transition_activations=total_transition_activations, log_file="../results/net_2_simulation_log.json")
    json_to_txt("../results/net_2_simulation_log.json", "../results/net_2_simulation_log.txt")
    print_txt_to_console("../results/net_2_simulation_log.txt")
    visualize_petri_net("../results/net_2_structure.json", "../results/net_2_simulation_log.json")
    
#### End of program
