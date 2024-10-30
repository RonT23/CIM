import random

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
        self.input_arcs = []       # Regular input arcs (place, weight)
        self.output_arcs = []      # Output arcs (place, weight)
        self.inhibitory_arcs = []  # Inhibitory arcs (places that should have 0 tokens for transition to fire)

    def add_input_arc(self, place, weight=1):
        self.input_arcs.append((place, weight))

    def add_output_arc(self, place, weight=1):
        self.output_arcs.append((place, weight))

    def add_inhibitory_arc(self, place):
        self.inhibitory_arcs.append(place)

    def is_enabled(self):
        # Check regular input arcs: verify each place has enough tokens to satisfy weight
        for place, weight in self.input_arcs:
            if place.tokens < weight:
                return False
        # Check inhibitory arcs: all inhibitory places must have 0 tokens
        for place in self.inhibitory_arcs:
            if place.tokens > 0:
                return False
        return True

    def fire(self):
        if not self.is_enabled():
            return False
        # Consume tokens from input places based on arc weights
        for place, weight in self.input_arcs:
            place.remove_tokens(weight)
        # Add tokens to output places based on arc weights
        for place, weight in self.output_arcs:
            place.add_tokens(weight)
        return True

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}

    def add_place(self, name, tokens=0):
        place = Place(name, tokens)
        self.places[name] = place

    def add_transition(self, name):
        transition = Transition(name)
        self.transitions[name] = transition

    def add_input_arc(self, transition_name, place_name, weight=1):
        if transition_name in self.transitions and place_name in self.places:
            transition = self.transitions[transition_name]
            place = self.places[place_name]
            transition.add_input_arc(place, weight)

    def add_output_arc(self, transition_name, place_name, weight=1):
        if transition_name in self.transitions and place_name in self.places:
            transition = self.transitions[transition_name]
            place = self.places[place_name]
            transition.add_output_arc(place, weight)

    def add_inhibitory_arc(self, transition_name, place_name):
        if transition_name in self.transitions and place_name in self.places:
            transition = self.transitions[transition_name]
            place = self.places[place_name]
            transition.add_inhibitory_arc(place)

    def simulate(self, target_transition_name, C_N):
        if target_transition_name not in self.transitions:
            print(f"Transition {target_transition_name} not found in the Petri net.")
            return

        target_transition = self.transitions[target_transition_name]
        target_activation_count = 0  # Track activations of the target transition
        steps = 0                    # Count total simulation steps

        while target_activation_count < C_N:
            enabled_transitions = [t for t in self.transitions.values() if t.is_enabled()]

            if not enabled_transitions:
                print("No enabled transitions. Simulation ended.")
                break

            # Randomly select a transition to fire
            chosen_transition = random.choice(enabled_transitions)
            if chosen_transition.fire():
                steps += 1
                print(f"Step {steps}: Transition '{chosen_transition.name}' fired.")
                
                # Increment counter if the chosen transition is the target
                if chosen_transition == target_transition:
                    target_activation_count += 1
                    print(f"Target Transition '{target_transition.name}' has fired {target_activation_count} times.")

            # Print current tokens in each place
            print("Current Tokens in Places:")
            for name, place in self.places.items():
                print(f"Place {name}: {place.tokens} tokens")

        print(f"\nSimulation finished after {steps} steps.")
        print(f"Target transition '{target_transition.name}' fired {target_activation_count} times.")

# Example Usage
# Define the Petri net structure and simulate it

net = PetriNet()
# Add places
net.add_place("P1", tokens=1)
net.add_place("P2", tokens=1)
net.add_place("P3", tokens=1)

# Add transitions
net.add_transition("T1")
net.add_transition("T2")

# Define arcs for T1
net.add_input_arc("T1", "P1", weight=1)
net.add_input_arc("T1", "P2", weight=1)
net.add_output_arc("T1", "P3", weight=1)

# Define arcs for T2, with an inhibitory arc on P3
net.add_input_arc("T2", "P3", weight=1)
net.add_output_arc("T2", "P1", weight=1)
net.add_output_arc("T2", "P2", weight=1)

# net.add_inhibitory_arc("T2", "P1")  # T2 cannot fire if P1 has tokens

# Run the simulation, targeting T2 for 3 activations
net.simulate(target_transition_name="T2", C_N=3)
