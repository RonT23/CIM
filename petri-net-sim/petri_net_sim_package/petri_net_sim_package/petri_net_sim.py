"""
    Project     : Petri Network Simulator - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 1
    Description : A simple implementation of a Petri Net simulator including inhibitory arcs and visualization of graph.
    Designer    : Ronaldo Tsela
    Date        : 28/10/2024
    Requires    : json, random, os
"""

import random
import json
import os

import networkx as nx
import matplotlib.pyplot as plt


'''
    This class creates and manages places in a Petri net, where each place
    can hold a non-negative number of tokens.

    Attributes:
        name (str): The name of the place.
        tokens_count (int): The current number of tokens in the place.

    Methods:
        __init__(name: str, tokens_count: int):
            Initializes a new Place instance with a specified name and initial token count.
            By default the token count is set to zero.

        add_tokens(tokens_count: int):
            Adds a specified number of tokens to this place.

        remove_tokens(tokens_count: int):
            Removes a specified number of tokens from this place.
'''
class Place:
    def __init__(self, name, tokens_count=0):
        self.name = name
        self.tokens = tokens_count

    def add_tokens(self, tokens_count):
        self.tokens += tokens_count

    def remove_tokens(self, tokens_count):
        self.tokens = max(self.tokens - tokens_count, 0)


'''
    This class creates and manages transitions in a Petri net. Transitions 
    connect places through arcs and control the flow of tokens based on firing 
    conditions. In a classical Petri net, transitions are connected to places 
    through input and output arcs only. Here inhibitory arcs are also 
    modeled.

    Attributes:
        name (str): The name of the transition.
        input_arcs (list): A list of tuples containing input places and their arc weights.
        output_arcs (list): A list of tuples containing output places and their arc weights.
        inhibitory_arcs (list): A list of places that inhibit this transition when they contain tokens.

    Methods:
        __init__(name: str):
            Initializes a Transition instance with a given name.

        add_input_arc(place, weight):
            Adds an input arc to this transition from a specified place.
            
            Args:
                place: The place that serves as input to this transition.
                weight (int): The weight of the arc. By default is 1.

        add_output_arc(place, weight):
            Adds an output arc to this transition to a specified place.
            
            Args:
                place: The place that serves as output for this transition.
                weight (int): The weight of the arc. By default is 1.

        add_inhibitory_arc(place):
            Adds an inhibitory arc that controls this transition by preventing it 
            from firing if the place contains tokens.

            Args:
                place: The place that inhibits this transition.

        is_enabled() -> bool:
            Checks if this transition can fire based on its inputs and inhibitory arcs.

            Returns:
                bool: True if the transition is enabled and can fire; False otherwise.

        fire() -> bool:
            Executes this transition, transferring tokens from input to output places 
            if the transition is enabled.

            Returns:
                bool: True if the transition fired successfully; False otherwise.
'''
class Transition:
    def __init__(self, name):
        self.name = name
        self.input_arcs = []
        self.output_arcs = []
        self.inhibitory_arcs = []

    def add_input_arc(self, place, weight=1):
        self.input_arcs.append((place, weight))

    def add_output_arc(self, place, weight=1):
        self.output_arcs.append((place, weight))

    def add_inhibitory_arc(self, place):
        self.inhibitory_arcs.append(place)

    def is_enabled(self):
        # Ensure each input place has enough tokens based on the arc weights
        for place, weight in self.input_arcs:
            if place.tokens < weight:
                return False

        # Ensure no tokens are in inhibitory places
        for place in self.inhibitory_arcs:
            if place.tokens > 0:
                return False

        return True

    def fire(self):
        if not self.is_enabled():
            return False

        # Remove tokens from input places according to the weights
        for place, weight in self.input_arcs:
            place.remove_tokens(weight)
            
        # Add tokens to output places according to the arc weights
        for place, weight in self.output_arcs:
            place.add_tokens(weight)

        return True

'''
    This class creates and manages a Petri net, allowing the addition of places,
    transitions, and arcs, and simulating the Petri net behavior over a series of 
    iterations defined by the number of total transitions occured on a target transition. 
    The maximum number of places and transitions set by default is 20.

    Methods:
        __init__(name: str, max_place_count: int, max_transition_count: int):
            Initializes a new Petri network instance with a specified limit on place and transition number

        add_place(name: str, tokens: int):
            Adds a new place to the Petri network if it does not exceed the defined limit.

            Args:
                name (str): The name of the place to add.
                tokens (int): The initial number of tokens in the place. Defaults to 0.

        add_transition(name: str):
            Adds a new transition to the Petri net if it does not exceed the defined limit.

            Args:
                name (str): The name of the transition to add.

        add_input_arc(transition_name: str, place_name: str, weight: int):
            Adds an input arc from a place to a transition.

            Args:
                transition_name (str): The name of the transition receiving input from this arc.
                place_name (str): The name of the place that serves as input.
                weight (int): The weight of the arc. By default is 1.

        add_output_arc(transition_name: str, place_name: str, weight: int):
            Adds an output arc from a transition to a place.

            Args:
                transition_name (str): The name of the transition producing output through this arc.
                place_name (str): The name of the place that receives output from this arc.
                weight (int): The weight of the arc. By defaults is 1.

        add_inhibitory_arc(transition_name: str, place_name: str):
            Adds an inhibitory arc from a place to a transition.

            Args:
                transition_name (str): The name of the transition controlled by this arc.
                place_name (str): The name of the place that inhibits this transition.
        
        export_structure(file_name: str):
            Exports the structure of the Petri net to a JSON file for external visualization and processing
            of the network structure.

            Args:
                file_name (str): The name of the file to export the structure to. By default is 'structure.json'.
            
        simulate(target_transition_name: str, total_transition_activations: int, log_file: str):
            Simulates the Petri net until a specified transition has fired a given number of times and records the
            simulation results (Marking, activations and deadlock occurances)

            Args:
                target_transition_name (str): The name of the transition to monitor for the termination condition.
                total_transition_activations (int): The total number of activations of the target transition required to stop the simulation.
                log_file (str): The file path for saving the simulation log. By default is "simulation_log.json".
'''
class PetriNet:
    def __init__(self, max_place_count=20, max_transition_count=20):
        self.places = {}
        self.transitions = {}
        self.arcs = []
        
        self.place_count = 0
        self.transition_count = 0
        
        self.max_place_count = max_place_count
        self.max_transition_count = max_transition_count

    def add_place(self, name, tokens=0):
        if self.place_count >= self.max_place_count :
            print("[ERROR] Reached the place count limit! Place node cannot be added to the network.")
        else:
            self.place_count += 1
            self.places[name] = Place(name, tokens)

    def add_transition(self, name):
        if self.transition_count >= self.max_transition_count :
            print("[ERROR] Reached the transition count limit! Transition node cannot be added to the network.")
        else:
            self.transition_count += 1    
            self.transitions[name] = Transition(name)

    def add_input_arc(self, transition_name, place_name, weight=1):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_input_arc(place, weight)
        self.arcs.append({"from": place_name, "to":transition_name, "type": "input", "weight": weight})

    def add_output_arc(self, transition_name, place_name, weight=1):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_output_arc(place, weight)
        self.arcs.append({"from": transition_name, "to": place_name, "type": "output", "weight": weight})

    def add_inhibitory_arc(self, transition_name, place_name):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_inhibitory_arc(place)
        self.arcs.append({"from": place_name, "to": transition_name, "type": "inhibitory"})

    def export_structure(self, file_name="structure.json"):
        
        # Delete the old versions of the structure configuration file if it exists
        try:
            os.remove(file_name)
        except:
            pass
            
        petri_net_struct_data = {
            "places": [{"name": place.name, "tokens": place.tokens} for place in self.places.values()],
            "transitions": list(self.transitions.keys()),
            "arcs": self.arcs 
        }

        with open(file_name, "w") as file:
            json.dump(petri_net_struct_data, file, indent=4)

    def simulate(self, target_transition_name, total_transition_activations, log_file="simulation_log.json"):
        
        target_transition = self.transitions[target_transition_name]
        target_activation_count = 0
        steps = 0
        log = []

        # Delete the old versions of the lof file if it already exists
        try:
            os.remove(log_file)
        except:
            pass
            
        while target_activation_count < total_transition_activations:
            
            enabled_transitions = [t for t in self.transitions.values() if t.is_enabled()]
            
            # Check for deadlocks
            if not enabled_transitions:
                deadlock_state = {place.name: place.tokens for place in self.places.values()}
                log.append({"step": steps, "deadlock": True, "state": deadlock_state})
                
                # Stop simulation, not a thing we can do here now on...
                break  

            # Chose randomply one of the enabled transitions adn fire it
            chosen_transition = random.choice(enabled_transitions)
            if chosen_transition.fire():
                steps += 1

                # If the choosen transition aligns with the target one then increase the termination index counter
                if chosen_transition == target_transition:
                    target_activation_count += 1
                
                step_state = {place.name: place.tokens for place in self.places.values()}
                step_state["Target_Fired_Count"] = target_activation_count
                step_state["Activated_Transition"] = chosen_transition.name
                log.append(step_state)
            
            print(f"[INFO] step {steps} \ Target transition activations: {target_activation_count}/{total_transition_activations}")

        # Create the output file and export the simualtion results
        with open(log_file, "w") as file:
            json.dump(log, file)

        if target_activation_count < total_transition_activations:
            print(f"[ERROR] Simulation ended due to deadlock after {steps} steps.")
        else:
            print(f"[STATUS] Simulation completed successfully in {steps} steps.")

'''
    This function visualizes the structure and simulation results of a Petri network exported from the simulator.
    The visualization continues until all steps in the simulation log have been displayed with a timestep of 1 second per step.
    
    Args:
        structure_file (str): The path to a JSON file containing the description of the Petri net structure as exported from the simulator.
                              This file contains the places, the transitions, and the arcs used to connect the formers.
        
        simulation_log_file (str): The path to a JSON file containing the simulation log filr, which records the state of 
                                    the Petri net at each step of the simulation.   
'''
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

    total_target_activations = 0

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
        
        try:
            total_target_activations = state["Target_Fired_Count"]
        except:
            pass 

        if is_deadlock:
            plt.title(f"Deadlock Detected at Step {step + 1}\nTarget Fired {total_target_activations} Times")
        else:
            plt.title(f"Network State at Step {step + 1}\nTarget Fired {total_target_activations} Times")
        
        # Hold on for a second!
        plt.pause(1)

    if not is_deadlock:
        plt.title(f"Simulation Ended Succesfully at Step {step + 1}")
    
    plt.show()


'''
    This function reads a JSON file containing the state of a Petri network simulation and exports the relevant data
    into a comma-separated TXT file for easier reading. It ensures that the 'Target_Fired' field is placed at the end of each row
    in the output file, whether or not a deadlock entry is encountered.
    
    Args:
        json_file (str): The path to the input JSON file containing the simulation data, which may include both 
                         normal state entries and deadlock entries.
                         
        txt_file (str): The path to the output TXT file where the processed simulation data will be saved. 
                        The output will be in a comma-separated format for easier readability.
'''
def json_to_txt(json_file, txt_file):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Open the TXT file for writing
    with open(txt_file, 'w') as file:
        if data:
            # Determine the header, ensuring 'Target_Fired' is last
            header = list(data[0].keys())
            if 'Target_Fired' in header:
                header.remove('Target_Fired')
                header.append('Target_Fired')
            file.write(', '.join(header) + '\n')

            for entry in data:
                # Handle deadlock entry separately if present
                if 'deadlock' in entry:
                    state = entry.get('state', {})
                    # Move 'Target_Fired' to the end for deadlock states
                    row = ', '.join(str(state.get(key, 0)) for key in header if key in state)
                else:
                    # Regular entries, with 'Target_Fired' at the end
                    row = ', '.join(str(entry.get(key, 0)) for key in header)
                
                file.write(row + '\n')

'''
    This function reads a comma-separated TXT file containing the processed Petri network simulation data 
    and prints the data line-by-line to the console for easy viewing.
    
    Args:
        txt_file (str): The path to the input TXT file containing the simulation data in comma-separated format.
'''
def print_txt_to_console(txt_file):
    with open(txt_file, 'r') as file:
        for line in file:
            print(line.strip())
