#!/usr/bin/python3
"""
    Project     : Petri Network Simulator Visualizer - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 1
    Description : This is a complementary script to the main Petri Net simulator script, used to visualize the exported results.
    Designer    : Ronaldo Tsela
    Date        : 28/10/2024
    Depedencies : json, networkx, matplotlib
"""
import json
import networkx as nx
import matplotlib.pyplot as plt

"""
    This function visualizes the structure and simulation results of a Petri network exported from the simulator.
    The visualization continues until all steps in the simulation log have been displayed with a delay of 1 second per step.
    
    Args:
        structure_file (str): The path to a JSON file containing the description of the Petri net structure as exported from the simulator.
                              This file contains the places, the transitions, and the arcs used to connect the formers.
        
        simulation_log_file (str): The path to a JSON file containing the simulation log, which records the state of 
                        the Petri net at each step of the simulation.   
"""
def visualize_petri_net(structure_file, simulation_log_file):

    # Load the Petri net structure descritpion file
    with open(structure_file, "r") as file:
        petri_net_data = json.load(file)
    
    # Load the simulation results file
    with open(simulation_log_file, "r") as file:
        simulation_log = json.load(file)
    
    G = nx.DiGraph()
    positions = {}
    
    # Places and transitions are nodes arragned vertically
    for idx, place in enumerate(petri_net_data["places"]):
        G.add_node(place["name"], type="place", tokens=place["tokens"])
        positions[place["name"]] = (0, -idx * 2) 

    for idx, transition in enumerate(petri_net_data["transitions"]):
        G.add_node(transition, type="transition")
        positions[transition] = (2, -idx * 2)

    # Places and transitions have edges described by arcs whome connects them
    arc_labels = {}
    for arc in petri_net_data["arcs"]:
        G.add_edge(arc["from"], arc["to"], arc_type=arc["type"])
        arc_labels[(arc["from"], arc["to"])] = arc.get("weight", 1)

    for step, state in enumerate(simulation_log):

        plt.clf() 
        
        token_labels        = {node: f"{node}\n{state.get(node, 0)}" for node in G.nodes() if G.nodes[node]["type"] == "place"}
        transition_labels   = {node: node for node in G.nodes() if G.nodes[node]["type"] == "transition"}
        combined_labels     = {**token_labels, **transition_labels}

        # Draw nodes and visualize the network
        place_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "place"]
        transition_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "transition"]

        nx.draw_networkx_nodes (G, pos=positions, nodelist=place_nodes, node_color="skyblue", node_size=400)
        nx.draw_networkx_nodes (G, pos=positions, nodelist=transition_nodes, node_color="salmon", node_size=400)
        nx.draw_networkx_labels(G, pos=positions, labels=combined_labels, font_size=10)
        
        input_edges = [(u, v) for u, v, d in G.edges(data=True) if d['arc_type'] == 'input']
        output_edges = [(u, v) for u, v, d in G.edges(data=True) if d['arc_type'] == 'output']
        inhibitory_edges = [(u, v) for u, v, d in G.edges(data=True) if d['arc_type'] == 'inhibitory']
      
        nx.draw_networkx_edges (G, pos=positions, edgelist=input_edges, arrows=True, width=1, arrowstyle='->', edge_color="skyblue")
        nx.draw_networkx_edges (G, pos=positions, edgelist=output_edges, arrows=True, width=1, arrowstyle='->', edge_color="salmon")
        nx.draw_networkx_edges(G, pos=positions, edgelist=inhibitory_edges, arrows=True, width=1, arrowstyle='->', edge_color='black', style='dashed')

        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=arc_labels, font_color='black')
       
        # Check if in this step there is a recorded deadlock
        is_deadlock = "deadlock" in state and state["deadlock"] == True
        total_target_activations = state["Target_Fired"]

        if is_deadlock:
            plt.title(f"Deadlock Detected at Step {step + 1}\nTarget Fired {total_target_activations} Times")
        else:
            plt.title(f"Network State at Step {step + 1}\nTarget Fired {total_target_activations} Times")
        
        # Hold on for a second!
        plt.pause(1)

    if not is_deadlock:
        plt.title(f"Simulation Ended Succesfully at Step {step + 1}")
    
    plt.show()

if __name__ == "__main__":
    visualize_petri_net("structure.json", "simulation_log.json")
