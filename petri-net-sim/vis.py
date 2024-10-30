import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_petri_net(log_file):
    with open(log_file, "r") as file:
        simulation_log = json.load(file)
    
    # Initialize graph
    G = nx.DiGraph()
    positions = {}
    
    # Add nodes for places and transitions
    places = [p for p in simulation_log[0].keys()]
    for idx, place in enumerate(places):
        G.add_node(place, type="place", tokens=simulation_log[0][place])
        positions[place] = (0, -idx * 2)  # Arrange places vertically

    transitions = ["T1", "T2"]  # Known transitions; adapt if needed
    for idx, transition in enumerate(transitions):
        G.add_node(transition, type="transition")
        positions[transition] = (2, -idx * 2)  # Arrange transitions vertically

    # Define the arcs manually (or dynamically if using an extended structure)
    G.add_edge("P1", "T1")
    G.add_edge("T1", "P2")
    G.add_edge("P2", "T2")
    G.add_edge("T2", "P3")
    G.add_edge("T2", "P4")

    # Visualize each step in simulation
    for step, state in enumerate(simulation_log):
        plt.clf()  # Clear previous step
        token_labels = {node: f"{node}\n{state[node]} tokens" if node in state else node
                        for node in G.nodes()}
        
        # Draw places and transitions
        place_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "place"]
        transition_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "transition"]
        
        nx.draw_networkx_nodes(G, pos=positions, nodelist=place_nodes, node_color="skyblue", node_size=1500)
        nx.draw_networkx_nodes(G, pos=positions, nodelist=transition_nodes, node_color="salmon", node_size=1000)
        nx.draw_networkx_labels(G, pos=positions, labels=token_labels, font_size=10)
        nx.draw_networkx_edges(G, pos=positions, edgelist=G.edges(), arrows=True)
        
        plt.title(f"Petri Net Simulation Step {step + 1}")
        plt.pause(1)  # Pause to view each step (adjust as needed)

    plt.show()

# Example call to visualize
visualize_petri_net("simulation_log.json")
