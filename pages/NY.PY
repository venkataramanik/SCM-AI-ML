import networkx as nx
import random
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_PORTS = 15  # Number of nodes in our global network
CRITICAL_PORT_COLOR = 'red'
DEFAULT_COLOR = 'skyblue'
DISRUPTED_COLOR = 'orange'
MAX_DISRUPTION_LEVEL = 10  # Max stress level a node can reach

# --- Functions ---

def create_global_network(num_ports):
    """Creates a random, well-connected graph representing global ports."""
    G = nx.fast_gnp_random_graph(num_ports, 0.35, seed=42)
    # Rename nodes to look like major ports
    port_names = [
        "Shanghai", "Singapore", "Rotterdam", "LA/LB", "Newark",
        "Busan", "Hamburg", "Jebel Ali", "Qingdao", "Hong Kong",
        "Felixstowe", "Mumbai", "Vancouver", "Panama", "Suez"
    ]
    mapping = {i: port_names[i] for i in range(num_ports)}
    G = nx.relabel_nodes(G, mapping)

    # Add initial attributes
    for u, v in G.edges():
        # Edge weight represents normal transit time (in days)
        G[u][v]['time'] = random.randint(7, 30)

    for node in G.nodes():
        G.nodes[node]['stress'] = 0      # Disruption stress level (0 = normal)
        G.nodes[node]['disrupted'] = False
        G.nodes[node]['propagation_time'] = float('inf') # Time for disruption to reach
        G.nodes[node]['color'] = DEFAULT_COLOR

    return G

def find_critical_port(G):
    """Uses Betweenness Centrality to find the most influential port."""
    # Betweenness Centrality measures how often a node appears on the shortest path
    # between other pairs of nodes. High centrality means high leverage/criticality.
    centrality = nx.betweenness_centrality(G)
    critical_port = max(centrality, key=centrality.get)
    return critical_port, centrality

def simulate_contagion(G, start_port, initial_stress=5):
    """Simulates the spread of the container shortage disruption."""
    G.nodes[start_port]['stress'] = initial_stress
    G.nodes[start_port]['disrupted'] = True
    G.nodes[start_port]['propagation_time'] = 0
    G.nodes[start_port]['color'] = CRITICAL_PORT_COLOR

    # Use a Queue-like list for breadth-first propagation
    queue = [start_port]
    
    # 1. Contagion Propagation
    # The shortage spreads to neighbors, causing a delay (stress) proportional to distance
    while queue:
        current_node = queue.pop(0)
        
        for neighbor in G.neighbors(current_node):
            if not G.nodes[neighbor]['disrupted']:
                # The "sneeze" spreads (contagion)
                G.nodes[neighbor]['disrupted'] = True
                
                # Propagation Time: Shortest path time from the source
                # The delay accumulates based on the original transit time
                time_to_reach = G.nodes[current_node]['propagation_time'] + G[current_node][neighbor]['time']
                G.nodes[neighbor]['propagation_time'] = min(G.nodes[neighbor]['propagation_time'], time_to_reach)
                
                # Node Stress: A measure of the cumulative container backlog
                # Stress decreases the further away you are, but still exists
                G.nodes[neighbor]['stress'] = max(1, G.nodes[current_node]['stress'] - random.randint(1, 2))
                G.nodes[neighbor]['color'] = DISRUPTED_COLOR
                
                queue.append(neighbor)

    # 2. Total Supply Chain Impact
    # The disruption increases transit time on all outgoing edges from stressed nodes
    total_transit_time_increase = 0
    for u, v, data in G.edges(data=True):
        original_time = data['time']
        
        # Delay factor is proportional to the stress of the originating port (u)
        stress_factor = G.nodes[u]['stress'] / MAX_DISRUPTION_LEVEL
        
        # New time = Original time + (Original time * Stress Factor * Random Multiplier)
        delay = original_time * stress_factor * random.uniform(0.5, 1.5)
        new_time = original_time + delay
        
        G[u][v]['new_time'] = new_time
        total_transit_time_increase += delay

    return total_transit_time_increase

def visualize_network(G, title):
    """Draws the network graph with stress levels and colors."""
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [500 + G.nodes[node]['stress'] * 200 for node in G.nodes()]
    labels = {node: f"{node}\n(S:{G.nodes[node]['stress']})" for node in G.nodes()}

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

# --- Main Execution ---
G = create_global_network(NUM_PORTS)
critical_port, centrality = find_critical_port(G)

print("=" * 80)
print("             Global Supply Chain Contagion Simulation")
print("=" * 80)
print(f"Network Nodes (Ports): {NUM_PORTS}")
print("Analysis: Betweenness Centrality (identifies critical flow points)")

# 1. Network Centrality Analysis
print("\n--- Centrality Ranking ---")
sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
for port, score in sorted_centrality[:5]:
    print(f"  {port:<12}: {score:.4f} (Centrality Score)")

print(f"\n**The 'Sneeze' Port (Highest Centrality): {critical_port}** ")
print(f"Initial Stress Level: 5 (representing severe container shortage/labor strike)")

# 2. Simulation and Contagion Modeling
total_delay = simulate_contagion(G, critical_port, initial_stress=5)

# 3. Results Summary
print("\n--- Contagion Results ---")
print(f"Total cumulative increase in global transit time: **{total_delay:.2f} days**")

print("\n| Port Name      | Stress Level | Propagation Time (Days) |")
print("|----------------|--------------|-------------------------|")
# Display results sorted by propagation time
results = sorted([(node, G.nodes[node]['stress'], G.nodes[node]['propagation_time']) 
                  for node in G.nodes()], key=lambda x: x[2])

for port, stress, p_time in results:
    time_str = f"{p_time:.2f}" if p_time != float('inf') else "N/A"
    print(f"| {port:<14} | {stress:<12} | {time_str:<23} |")
print("-" * 48)

# 4. Visualization
visualize_network(G, f"Contagion from '{critical_port}' (Node Stress and Propagation)")
