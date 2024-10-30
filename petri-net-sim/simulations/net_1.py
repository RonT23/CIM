from petri_net_sim_package.petri_net_sim import *

if __name__ == "__main__" :
    
    net = PetriNet()

##### Define here your Petri network ####

##### End of network definitions     ####

##### Don't touch these

    net.export_structure("net_1_structure.json")
    net.simulate(target_transition_name=target_transition_name, total_transition_activations=total_transition_activations, log_file="net_1_simulation_log.json")
    json_to_txt("net_1_simulation_log.json", "../results/net_1_simulation_log.txt")
    visualize_petri_net("net_1_structure.json", "net_1_simulation_log.json")
#### End of program
