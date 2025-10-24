# app.py
# When a Port Sneezes, the World Catches a Cold — Interactive Ripple Demo (Streamlit)
# ------------------------------------------------------------------------------
# Purpose:
#   Show how a local container shortage at one port can ripple through trade lanes.
#   Focus is on clarity and business interpretation (no ML/AI jargon).
#
# What it does:
#   - Builds a small network of major ports and common routes
#   - Seeds a shortage at one port and simulates how "pressure" spreads to neighbors
#   - Lets you adjust spread strength, recovery speed, duration, and temporary closure
#   - Produces two charts (final ripple map, global pressure timeline) and downloadable CSVs
#
# Requirements (add to requirements.txt):
# streamlit
# networkx
# matplotlib
# pandas
# numpy
#
# Run:
#   streamlit run app.py

from typing import Dict, Tuple
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ----------------------
# Model & Utility funcs
# ----------------------

def build_network() -> nx.Graph:
    ports = [
        "Shanghai", "Singapore", "Los Angeles", "Long Beach", "Rotterdam", "Antwerp",
        "Shenzhen", "Ningbo", "Busan", "Dubai", "Hamburg", "Felixstowe",
        "Tanjung Pelepas", "Valencia", "New York/New Jersey"
    ]
    # Illustrative trade lanes (toy model for demonstration)
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


def compute_centrality(G: nx.Graph) -> Dict[str, float]:
    # Betweenness centrality indicates how often a port lies on shortest routes (a proxy for network importance).
    return nx.betweenness_centrality(G, normalized=True)


def simulate(
    G: nx.Graph,
    seed: str = "Shanghai",
    steps: int = 12,
    spread: float = 0.45,
    decay: float = 0.15,
    closure_steps: int = 0,
    layout_seed: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Simulates shortage pressure over a number of steps.
    - spread: portion of neighbor pressure that spills over each step (0..1)
    - decay: natural recovery percentage each step (0..1)
    - closure_steps: for the first N steps, remove all edges from the seed port
    Returns:
      df: long-form DataFrame [time, port, stress]
      pos: fixed layout positions for plotting
    """
    ports = list(G.nodes)
    pos = nx.spring_layout(G, seed=layout_seed)

    # Initial pressure: only the seed is stressed at t=0
    stress = {p: 0.0 for p in ports}
    stress[seed] = 1.0

    seed_edges = list(G.edges(seed))
    closure_window = set(range(1, closure_steps + 1))

    history = [{"time": 0, **stress}]

    for t in range(1, steps + 1):
        removed = []
        if t in closure_window:
            for e in seed_edges:
                if G.has_edge(*e):
                    G.remove_edge(*e)
                    removed.append(e)

        # Recovery toward zero
        new_stress = {p: max(0.0, s * (1 - decay)) for p, s in stress.items()}

        # Neighbor spillover
        for p in ports:
            neigh = list(G.neighbors(p))
            if not neigh:
                continue
            inflow = sum(stress[n] for n in neigh) / len(neigh)
            new_stress[p] = min(1.0, new_stress[p] + spread * inflow)

        # Keep the seed pressured during closure to reflect ongoing local issue
        if t in closure_window:
            new_stress[seed] = 1.0

        stress = new_stress
        history.append({"time": t, **stress})

        # Restore removed edges after each step
        if removed:
            G.add_edges_from(removed)

    df = pd.DataFrame(history).melt(id_vars=["time"], var_name="port", value_name="stress")
    return df, pos


def plot_network(G: nx.Graph, df: pd.DataFrame, pos: Dict[str, Tuple[float, float]], seed: str):
    latest_t = int(df["time"].max())
    final = df[df["time"] == latest_t].set_index("port")["stress"].to_dict()

    fig = plt.figure(figsize=(9, 7))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=[3000 if p == seed else 1800 for p in G.nodes],
        node_color=[final.get(p, 0.0) for p in G.nodes],  # default colormap (no manual colors)
        font_size=8
    )
    plt.title(f"Final Ripple Map (Seed: {seed})")
    plt.tight_layout()
    return fig


def plot_global_timeline(df: pd.DataFrame):
    global_stress = df.groupby("time")["stress"].sum().reset_index()

    fig = plt.figure(figsize=(9, 4))
    plt.plot(global_stress["time"], global_stress["stress"], marker="o")
    plt.title("Global Pressure Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Total Pressure Across Ports")
    plt.tight_layout()
    return fig, global_stress


def make_rankings(df: pd.DataFrame, centrality: Dict[str, float]):
    peak = df.groupby("port")["stress"].max().sort_values(ascending=False).reset_index()
    peak.columns = ["port", "peak_stress"]

    cent = pd.Series(centrality, name="betweenness").reset_index().rename(columns={"index": "port"})
    ranks = peak.merge(cent, on="port", how="left")
    ranks["rank_peak"] = ranks["peak_stress"].rank(ascending=False, method="min").astype(int)
    ranks["rank_centrality"] = ranks["betweenness"].rank(ascending=False, method="min").astype(int)
    return peak, cent, ranks


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def linkedin_blurb(seed: str, spread: float, decay: float, closure_steps: int, ranks: pd.DataFrame) -> str:
    top3 = ranks.sort_values(["rank_peak", "rank_centrality"]).head(3)["port"].tolist()
    top3_txt = ", ".join(top3) if top3 else "multiple hubs"
    return f"""One disruption can travel far.

We simulated a small container shortage at one port and tracked how pressure spread along common trade lanes.
Seed: {seed} | Spread: {spread} | Recovery: {decay} | Temporary closure steps: {closure_steps}

Observations:
• Pressure clustered at central hubs: {top3_txt}.
• Local constraints can raise lead times and costs across distant lanes.

Actions:
• Pre-position empties near key hubs.
• Trigger reroutes when pressure crosses thresholds.
• Avoid over-reliance on a single port or corridor."""


# --------------
# Streamlit App
# --------------

st.set_page_config(page_title="Port Ripple Demo", layout="wide")

st.title("When a Port Sneezes, the World Catches a Cold")
st.caption("A simple network demonstration of how local shortages can ripple through trade lanes.")

# Controls
with st.sidebar:
    st.header("Controls")
    G = build_network()
    centrality = compute_centrality(G)

    seed_port = st.selectbox("Seed Port", sorted(G.nodes), index=sorted(G.nodes).index("Shanghai"))
    steps = st.slider("Time Steps", min_value=5, max_value=30, value=12, step=1)
    spread = st.slider("Spread Strength (0..1)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    decay = st.slider("Recovery per Step (0..1)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    closure_steps = st.slider("Temporary Closure of Seed (steps)", min_value=0, max_value=10, value=0, step=1)
    layout_seed = st.number_input("Layout Seed (for stable node placement)", min_value=0, max_value=9999, value=3, step=1)

# Run simulation
df, pos = simulate(
    G.copy(),
    seed=seed_port,
    steps=steps,
    spread=spread,
    decay=decay,
    closure_steps=closure_steps,
    layout_seed=layout_seed,
)

# Charts
c1, c2 = st.columns([1.08, 0.92])

with c1:
    st.subheader("Final Ripple Map")
    fig_net = plot_network(G, df, pos, seed_port)
    st.pyplot(fig_net, clear_figure=True)

with c2:
    st.subheader("Global Pressure Timeline")
    fig_tl, global_stress = plot_global_timeline(df)
    st.pyplot(fig_tl, clear_figure=True)

# Tables
peak, cent, ranks = make_rankings(df, centrality)

st.subheader("Peak Pressure by Port")
st.dataframe(peak, use_container_width=True)

st.subheader("Network Importance (Betweenness Centrality)")
st.dataframe(cent.sort_values("betweenness", ascending=False), use_container_width=True)

st.subheader("Combined View (Peak vs Centrality)")
st.dataframe(ranks.sort_values(["rank_peak", "rank_centrality"]), use_container_width=True)

# Downloads
st.markdown("### Downloads")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.download_button("Stress History CSV", data=df_to_csv_bytes(df), file_name="stress_history.csv")
with colB:
    st.download_button("Peak Ranking CSV", data=df_to_csv_bytes(peak), file_name="ranking_peak_stress.csv")
with colC:
    st.download_button("Centrality Ranking CSV", data=df_to_csv_bytes(cent), file_name="ranking_centrality.csv")
with colD:
    st.download_button("Combined Ranking CSV", data=df_to_csv_bytes(ranks), file_name="ranking_combined.csv")

# LinkedIn blurb
st.markdown("### Summary Blurb")
st.code(linkedin_blurb(seed_port, spread, decay, closure_steps, ranks), language="markdown")

# Plain-English notes
st.markdown("---")
st.markdown("#### How to read this")
st.markdown(
    "- Each port is a junction. A local shortage creates pressure that can spill into neighbors over shared routes.\n"
    "- Over a few steps, the effect can spread widely, especially through central hubs.\n"
    "- Use this to identify points where rerouting or extra containers can limit knock-on effects."
)

# Machine-readable export
summary = {
    "seed": seed_port,
    "steps": steps,
    "spread": spread,
    "decay": decay,
    "closure_steps": closure_steps,
    "top_impacted_by_peak": peak.head(5).to_dict(orient="records"),
}
st.download_button(
    "Download Summary (JSON)",
    data=json.dumps(summary, indent=2).encode("utf-8"),
    file_name="summary.json",
    mime="application/json",
)

st.caption("Demonstration only. Uses default matplotlib styles, one chart per figure.")
