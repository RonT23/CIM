#!/usr/bin/python3
"""
    Project     : Petri Network Simulator - MSc Autonomous Control Systems and Robotics - NTUA Fall 2024-2025 - CIM ex. 1
    Description : A simple implementation of a Petri Net simulator including inhibitory arcs and visualization.
    Designer    : Ronaldo Tsela
    Date        : 28/10/2024
    Depedencies : json, random
"""

import random
import json

"""
    This class creates and manages places in a Petri net, where each place
    can hold a non-negative number of tokens.

    Attributes:
        name (str): The name of the place.
        tokens_count (int): The current number of tokens in the place.

    Methods:
        __init__(name: str, tokens_count: int = 0):
            Initializes a new Place instance with a specified name and initial token count.
            By default the token count is set to zero.

        add_tokens(tokens_count: int):
            Adds a specified number of tokens to this place.

        remove_tokens(tokens_count: int):
            Removes a specified number of tokens from this place.
"""

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
    included.

    Attributes:
        name (str): The name of the transition.
        input_arcs (list): A list of tuples containing input places and their arc weights.
        output_arcs (list): A list of tuples containing output places and their arc weights.
        inhibitory_arcs (list): A list of places that inhibit this transition when they contain tokens.

    Methods:
        __init__(name: str):
            Initializes a Transition instance with a given name.

        add_input_arc(place, weight=1):
            Adds an input arc to this transition from a specified place.
            
            Args:
                place: The place that serves as input to this transition.
                weight (int): The weight of the arc. Default is 1.

        add_output_arc(place, weight=1):
            Adds an output arc to this transition to a specified place.
            
            Args:
                place: The place that serves as output for this transition.
                weight (int): The weight of the arc. Default is 1.

        add_inhibitory_arc(place):
            Adds an inhibitory arc that controls this transition by preventing it 
            from firing if the place contains tokens.

            Args:
                place: The place that inhibits this transition.

        is_enabled() -> bool:
            Checks if this transition can fire based on its input and inhibitory arcs.

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

    Methods:
        add_place(name: str, tokens: int = 0):
            Adds a new place to the Petri net.

            Args:
                name (str): The name of the place to add.
                tokens (int): The initial number of tokens in the place. Defaults to 0.

        add_transition(name: str):
            Adds a new transition to the Petri net.

            Args:
                name (str): The name of the transition to add.

        add_input_arc(transition_name: str, place_name: str, weight: int = 1):
            Adds an input arc from a place to a transition.

            Args:
                transition_name (str): The name of the transition receiving input from this arc.
                place_name (str): The name of the place that serves as input.
                weight (int): The weight of the arc. Defaults to 1.

        add_output_arc(transition_name: str, place_name: str, weight: int = 1):
            Adds an output arc from a transition to a place.

            Args:
                transition_name (str): The name of the transition providing output through this arc.
                place_name (str): The name of the place that receives output from this arc.
                weight (int): The weight of the arc. Defaults to 1.

        add_inhibitory_arc(transition_name: str, place_name: str):
            Adds an inhibitory arc from a place to a transition.

            Args:
                transition_name (str): The name of the transition controlled by this arc.
                place_name (str): The name of the place that inhibits this transition.
        
        export_structure(file_name: str):
            Exports the structure of the Petri net to a JSON file for external visualization.

            Args:
                file_name (str): The name of the file to export the structure to. Default is 'structure.json'.
            
        simulate(target_transition_name: str, total_transition_activations: int, log_file: str):
            Simulates the Petri net until a specified transition has fired a set number of times.

            Args:
                target_transition_name (str): The name of the transition to monitor for the termination condition.
                total_transition_activations (int): The total number of activations of the target transition required to stop the simulation.
                log_file (str): The file path for saving the simulation log. Default value is "simulation_log.json".
'''
class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []

    def add_place(self, name, tokens=0):
        self.places[name] = Place(name, tokens)

    def add_transition(self, name):
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

        petri_net_struct_data = {
            "places": [{"name": place.name, "tokens": place.tokens} for place in self.places.values()],
            "transitions": list(self.transitions.keys()),
            "arcs": self.arcs 
        }

        with open(file_name, "w") as file:
            json.dump(petri_net_struct_data, file, indent=4)

        print(f"Petri net structure exported to {file_name}")

    def simulate(self, target_transition_name, total_transition_activations, log_file="simulation_log.json"):
        
        target_transition = self.transitions[target_transition_name]
        target_activation_count = 0
        steps = 0
        log = []

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
                log.append(step_state)
        
        # Create the output file
        with open(log_file, "w") as file:
            json.dump(log, file)

        # Final message if deadlock prevented completion
        if target_activation_count < total_transition_activations:
            print(f"Simulation ended due to deadlock after {steps} steps. Target transition activations: {target_activation_count}/{total_transition_activations}")
        else:
            print(f"Simulation completed successfully in {steps} steps.")


if __name__ == "__main__" :
    
    net = PetriNet()
    
    net.add_place("P1", tokens=1)
    net.add_place("P2", tokens=0)
    net.add_place("P3", tokens=2)
    net.add_place("P4", tokens=0)

    net.add_transition("T1")
    net.add_transition("T2")
    net.add_transition("T3")
    
    net.add_input_arc("T1", "P1", weight=1)
    net.add_output_arc("T1", "P3", weight=1)

    net.add_input_arc("T2", "P2", weight=1)
    net.add_output_arc("T2", "P4", weight=1)

    net.add_input_arc("T3", "P3", weight=1)
    net.add_input_arc("T3", "P4", weight=1)
    net.add_output_arc("T3", "P1", weight=1)
    net.add_output_arc("T3", "P2", weight=1)
    
    net.export_structure("structure.json")

    net.simulate(target_transition_name="T3", total_transition_activations=3)
