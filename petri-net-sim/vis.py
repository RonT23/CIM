
import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_petri_net(structure_file, log_file):

    # Load the Petri net structure descritpion file
    with open(structure_file, "r") as file:
        petri_net_data = json.load(file)
    
    # Load the simulation results file
    with open(log_file, "r") as file:
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

    # Places and transitions have edges described by arcs
    for arc in petri_net_data["arcs"]:
        G.add_edge(arc["from"], arc["to"])

    # Visualize the simulation
    for step, state in enumerate(simulation_log):
        # Clear previous step
        plt.clf() 
        
        # Check if in this step there is a recorded deadlock
        is_deadlock = "deadlock" in state and state["deadlock"] == True
        
        # Update token labels
        token_labels = {node: f"{node}\n{state.get(node, 0)} tokens" for node in G.nodes()}
        
        # Draw nodes
        place_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "place"]
        transition_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "transition"]
        
        nx.draw_networkx_nodes(G, pos=positions, nodelist=place_nodes, node_color="blue", node_size=1500)
        nx.draw_networkx_nodes(G, pos=positions, nodelist=transition_nodes, node_color="red", node_size=1000)
        nx.draw_networkx_labels(G, pos=positions, labels=token_labels, font_size=10)
        nx.draw_networkx_edges(G, pos=positions, edgelist=G.edges(), arrows=True)

        if is_deadlock:
            plt.title(f"Deadlock Detected at Step {step + 1}")
        else:
            plt.title(f"Network State at Step {step + 1}")
        
        # Hold on for better visualization
        plt.pause(1)

    
    plt.show()

if __name__ == "__main__":
    visualize_petri_net("structure.json", "simulation_log.json")
