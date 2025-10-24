# app.py
# When a Port Sneezes, the World Catches a Cold — Simple Ripple Demo (Streamlit)
# ------------------------------------------------------------------------------
# What this shows (in plain English):
# - A small container shortage at one port can "push" pressure into nearby ports.
# - That pressure spreads along common trade lanes and gathers at key hubs.
# - You can adjust the seed port, how much pressure spreads each step, and how many steps to watch.
#
# Requirements (requirements.txt):
# streamlit
# networkx
# matplotlib
# pandas
# numpy
#
# Run: streamlit run app.py

from typing import Dict, Tuple
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# 1) Small, clear data model
# -----------------------------
def build_network() -> nx.Graph:
    ports = [
        "Shanghai", "Singapore", "Los Angeles", "Long Beach", "Rotterdam", "Antwerp",
        "Shenzhen", "Ningbo", "Busan", "Dubai", "Hamburg", "Felixstowe",
        "Tanjung Pelepas", "Valencia", "New York/New Jersey"
    ]
    edges = [
        ("Shanghai", "Singapore"),
        ("Shanghai", "Busan"),
        ("Shanghai", "Ningbo"),
        ("Shenzhen", "Singapore"),
        ("Shenzhen", "Busan"),
        ("Ningbo", "Busan"),
        ("Singapore", "Tanjung Pelepas"),
        ("Singapore", "Dubai"),
        ("Dubai", "Rotterdam"),
        ("Dubai", "Antwerp"),
        ("Rotterdam", "Antwerp"),
        ("Rotterdam", "Hamburg"),
        ("Rotterdam", "Felixstowe"),
        ("Antwerp", "Hamburg"),
        ("Felixstowe", "Valencia"),
        ("Valencia", "Rotterdam"),
        ("Los Angeles", "Long Beach"),
        ("Los Angeles", "Shanghai"),
        ("Los Angeles", "Singapore"),
        ("Long Beach", "Shanghai"),
        ("Long Beach", "Singapore"),
        ("Los Angeles", "New York/New Jersey"),
        ("New York/New Jersey", "Rotterdam"),
        ("New York/New Jersey", "Felixstowe"),
    ]
    G = nx.Graph()
    G.add_nodes_from(ports)
    G.add_edges_from(edges)
    return G

def network_importance(G: nx.Graph) -> Dict[str, float]:
    # Plain-language: "How central is a port in common routes?"
    return nx.betweenness_centrality(G, normalized=True)

# -----------------------------------------
# 2) Simple ripple (no math jargon)
# -----------------------------------------
def simulate(
    G: nx.Graph,
    seed: str,
    steps: int,
    spread: float,
    decay: float = 0.15,   # gentle recovery each step
    layout_seed: int = 3
):
    ports = list(G.nodes)
    pos = nx.spring_layout(G, seed=layout_seed)

    # Start: only the seed port is pressured
    stress = {p: 0.0 for p in ports}
    stress[seed] = 1.0

    history = [{"time": 0, **stress}]

    for t in range(1, steps + 1):
        # 1) natural recovery toward zero
        new_stress = {p: max(0.0, s * (1 - decay)) for p, s in stress.items()}
        # 2) spillover from neighbors (average neighbor stress * spread)
        for p in ports:
            neigh = list(G.neighbors(p))
            if not neigh:
                continue
            inflow = sum(stress[n] for n in neigh) / len(neigh)
            new_stress[p] = min(1.0, new_stress[p] + spread * inflow)
        stress = new_stress
        history.append({"time": t, **stress})

    df = pd.DataFrame(history).melt(id_vars=["time"], var_name="port", value_name="stress")
    return df, pos

# -----------------------------------------
# 3) Small, clear charts (matplotlib only)
# -----------------------------------------
def plot_network(G: nx.Graph, df: pd.DataFrame, pos: Dict[str, Tuple[float, float]], seed: str):
    latest = int(df["time"].max())
    final = df[df["time"] == latest].set_index("port")["stress"].to_dict()

    fig = plt.figure(figsize=(9, 6))
