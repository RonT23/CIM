import random
import json

class Place:
    def __init__(self, name, tokens=0):
        self.name = name
        self.tokens = tokens

    def add_tokens(self, count):
        self.tokens += count

    def remove_tokens(self, count):
        self.tokens = max(self.tokens - count, 0)

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
        for place, weight in self.input_arcs:
            if place.tokens < weight:
                return False
        for place in self.inhibitory_arcs:
            if place.tokens > 0:
                return False
        return True

    def fire(self):
        if not self.is_enabled():
            return False
        for place, weight in self.input_arcs:
            place.remove_tokens(weight)
        for place, weight in self.output_arcs:
            place.add_tokens(weight)
        return True

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}

    def add_place(self, name, tokens=0):
        self.places[name] = Place(name, tokens)

    def add_transition(self, name):
        self.transitions[name] = Transition(name)

    def add_input_arc(self, transition_name, place_name, weight=1):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_input_arc(place, weight)

    def add_output_arc(self, transition_name, place_name, weight=1):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_output_arc(place, weight)

    def add_inhibitory_arc(self, transition_name, place_name):
        transition = self.transitions[transition_name]
        place = self.places[place_name]
        transition.add_inhibitory_arc(place)

    def simulate(self, target_transition_name, C_N, log_file="simulation_log.json"):
        target_transition = self.transitions[target_transition_name]
        target_activation_count = 0
        steps = 0
        log = []

        while target_activation_count < C_N:
            enabled_transitions = [t for t in self.transitions.values() if t.is_enabled()]
            if not enabled_transitions:
                break
            chosen_transition = random.choice(enabled_transitions)
            if chosen_transition.fire():
                steps += 1
                if chosen_transition == target_transition:
                    target_activation_count += 1
                step_state = {place.name: place.tokens for place in self.places.values()}
                log.append(step_state)
        
        with open(log_file, "w") as file:
            json.dump(log, file)

# Example Usage
net = PetriNet()
net.add_place("P1", tokens=1)
net.add_place("P2", tokens=0)
net.add_place("P3", tokens=0)
net.add_place("P4", tokens=0)
net.add_transition("T1")
net.add_transition("T2")
net.add_input_arc("T1", "P1", weight=1)
net.add_output_arc("T1", "P2", weight=1)
net.add_input_arc("T2", "P2", weight=1)
net.add_output_arc("T2", "P3", weight=1)
net.add_output_arc("T2", "P4", weight=1)
net.add_inhibitory_arc("T2", "P1")
net.simulate(target_transition_name="T2", C_N=3)
