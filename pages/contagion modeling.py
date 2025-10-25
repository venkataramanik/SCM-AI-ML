import networkx as nx
import matplotlib.pyplot as plt
import random

# ======================================================================
# SUPPLY CHAIN CONTAGION MODEL: DEMO FOR BUSINESS AUDIENCE
# ======================================================================

def create_supply_chain_network():
    """
    BUSINESS EXPLANATION: Defines the 'neighborhood' of companies.
    
    This function creates a map (a graph) of all the critical partners.
    Nodes are individual companies (e.g., Core Firm, Suppliers, Logistics).
    Edges are the dependencies (the contracts, payments, or physical goods flow).
    """
    G = nx.DiGraph()
    
    # Nodes: Critical players in the supply chain
    nodes = ['A (Core)', 'S1', 'S2', 'S3', 'B1', 'B2', 'L (Logistics)', 'F (Financier)']
    G.add_nodes_from(nodes)
    
    # Edges: Operational and Financial Links
    # S -> A: Suppliers feed the Core Firm
    G.add_edge('S1', 'A (Core)', type='Goods')
    G.add_edge('S2', 'A (Core)', type='Goods')
    G.add_edge('S3', 'A (Core)', type='Goods')
    
    # A -> B: Core Firm supplies the Buyers
    G.add_edge('A (Core)', 'B1', type='Goods')
    G.add_edge('A (Core)', 'B2', type='Goods')

    # Shared Dependencies: The hidden links that spread risk
    # L affects S1, S2 affects L, F affects A and S3 (Financial risk)
    G.add_edge('L (Logistics)', 'S1', type='Service') 
    G.add_edge('S2', 'L (Logistics)', type='Service') 
    G.add_edge('F (Financier)', 'A (Core)', type='Credit') 
    G.add_edge('F (Financier)', 'S3', type='Credit')

    return G, nodes

# ----------------------------------------------------------------------

def simulate_contagion(G, initial_infected, infection_prob=0.3, recovery_prob=0.2, steps=10):
    """
    BUSINESS EXPLANATION: Runs the 'Domino Effect' Simulation (SIR Model).
    
    This function models how a disruption spreads using an epidemiological
    framework (SIR: Susceptible-Infected-Recovered).
    
    - Susceptible (0 / Blue): Healthy, but exposed to risk.
    - Infected (1 / Red): Currently disrupted, spreading the failure.
    - Recovered (2 / Green): Contained the risk (e.g., found a new supplier).
    
    infection_prob: Likelihood a risk spreads to a connected partner.
    recovery_prob: Likelihood a company can mitigate the risk and become stable.
    """
    state = {node: 0 for node in G.nodes} # Initialize all as Susceptible
    
    # Set the Initial Shock
    for node in initial_infected:
        if node in state:
            state[node] = 1 

    history = {t: state.copy() for t in range(steps)}

    for t in range(1, steps):
        new_state = state.copy()
        
        for node in G.nodes:
            # 1. Recovery: Companies successfully execute mitigation plans
            if state[node] == 1:
                if random.random() < recovery_prob:
                    new_state[node] = 2 
                    
            # 2. Infection: Companies catch the disruption from a neighbor
            elif state[node] == 0:
                infected_neighbors = 0
                # Check neighbors both upstream and downstream
                for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
                    if state[neighbor] == 1:
                        infected_neighbors += 1
                
                # If exposed and unlucky, the risk spreads
                if infected_neighbors > 0 and random.random() < infection_prob:
                    new_state[node] = 1 

        state = new_state
        history[t] = state.copy()
        
        # Stop simulation if the crisis is contained
        if all(s != 1 for s in state.values()):
            print(f"Contagion contained at Time Step {t}.")
            break
            
    return history

# ----------------------------------------------------------------------

def visualize_step(G, state, pos, step_num):
    """
    BUSINESS EXPLANATION: Creates a visual map of the crisis spread.
    
    This shows the state of the network at a specific point in time,
    making the systemic risk immediately clear.
    """
    
    # Define colors for the three risk states
    color_map = {0: 'lightblue', 1: 'red', 2: 'lightgreen'} 
    node_colors = [color_map[state[node]] for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors, 
            node_size=3000, 
            font_size=10, 
            font_weight='bold',
            arrowsize=20)
    
    # Add descriptive title and legend
    plt.title(f"SC Contagion Analysis: Systemic Risk Spread - Time Step {step_num}")
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Susceptible (Healthy/Exposed)', 
                   markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Infected (Disrupted/Failing)', 
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Recovered (Mitigated/Stable)', 
                   markerfacecolor='lightgreen', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Setup the Network
    G, nodes = create_supply_chain_network()
    pos = nx.spring_layout(G, seed=42) 

    # 2. Define the Initial SCXM Issue
    # SCXM Issue: Supplier S2 has an operational failure (e.g., equipment breakdown).
    initial_infected_node = ['S2'] 
    
    print(f"SIMULATION START: A major operational failure begins at: {initial_infected_node}")
    
    # 3. Run the Simulation with Parameters
    # Risk spreads easily (0.4) but mitigation is slow (0.1)
    history = simulate_contagion(
        G, 
        initial_infected_node, 
        infection_prob=0.4, 
        recovery_prob=0.1,  
        steps=5 
    )

    # 4. Visualize the Results at each critical step
    for step, state in history.items():
        visualize_step(G, state, pos, step)
             
    print("\nSIMULATION END: The results show how critical S2's position is for systemic risk.")
    
# ======================================================================
# TOOLS USED:
# 1. networkx (Graph Theory): Used to map the relationships (nodes and edges).
# 2. Matplotlib (Visualization): Used to plot the maps and show the spread.
# 3. SIR Model (Contagion Logic): The mathematical model defining how failure spreads and stops.
# ======================================================================
