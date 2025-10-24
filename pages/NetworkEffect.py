# app.py
# When a Port Sneezes, the World Catches a Cold — Interactive Ripple Demo (Streamlit)
# ------------------------------------------------------------------------------
# Order:
#   0) Simulation Controls (TOP — users tweak here)
#   1) Summary Blurb (auto-updates from controls)
#   2) How to Read This
#   3) Tools Used
#   4) Why This Matters (Business Value)
#   5) Charts (Final Ripple Map, Global Pressure Timeline)
#   6) Tables (Peak, Centrality, Combined)
#   7) Downloads (CSVs + JSON + Blurb)
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


# ----------------------
# Model & Utility funcs
# ----------------------

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


def compute_centrality(G: nx.Graph) -> Dict[str, float]:
    # Betweenness centrality as a proxy for network importance (“super-spreader” potential)
    return nx.betweenness_centrality(G, normalized=True)


def simulate(
    G: nx.Graph,
    seed: str = "Shanghai",
    steps: int = 12,
    spread: float = 0.45,
    decay: float = 0.15,
    closure_steps: int = 0,
    layout_seed: int = 3,
):
    """
    Simulates shortage pressure over 'steps'.
    - spread: neighbor spillover strength (0..1)
    - decay: natural recovery each step (0..1)
    - closure_steps: temporarily remove all edges from seed for the first N steps
    Returns:
      df: long DataFrame [time, port, stress]
      pos: fixed layout positions for plotting
    """
    ports = list(G.nodes)
    pos = nx.spring_layout(G, seed=layout_seed)

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

        # Keep seed pressured during closure window
        if t in closure_window:
            new_stress[seed] = 1.0

        stress = new_stress
        history.append({"time": t, **stress})

        # Restore edges after step
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
        font_size=8,
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


def summary_blurb(seed: str, spread: float, decay: float, closure_steps: int, ranks: pd.DataFrame) -> str:
    top3 = ranks.sort_values(["rank_peak", "rank_centrality"]).head(3)["port"].tolist()
    top3_txt = ", ".join(top3) if top3 else "multiple hubs"
    return f"""**Summary Blurb**

**One disruption can travel far.**

We simulated a small container shortage at one port and tracked how pressure spread along common trade lanes.  
**Seed:** {seed} &nbsp;&nbsp;|&nbsp;&nbsp; **Spread:** {spread} &nbsp;&nbsp;|&nbsp;&nbsp; **Recovery:** {decay} &nbsp;&nbsp;|&nbsp;&nbsp; **Temporary closure steps:** {closure_steps}

**Observations**  
• Pressure clustered at central hubs: {top3_txt}.  
• Local constraints can raise lead times and costs across distant lanes.

**Actions**  
• Pre-position empties near key hubs.  
• Trigger reroutes when pressure crosses thresholds.  
• Avoid over-reliance on a single port or corridor.
"""


def how_to_read_section() -> str:
    return """**How to Read This**

• Each port is a junction. A local shortage creates pressure that can spill into neighbors over shared routes.  
• Over a few steps, the effect can spread widely, especially through central hubs.  
• Use this to identify points where rerouting or extra containers can limit knock-on effects.
"""


def tools_used_section() -> str:
    return """**Tools Used (Under the Hood)**

• **Network Graph:** Ports are nodes; common trade lanes are edges.  
• **Ripple Simulation:** Each step, a portion of a port’s pressure spills into its neighbors; all ports also recover a bit each step.  
• **Centrality Ranking:** Betweenness centrality highlights “network-important” ports that often sit on key routes.  
• **Peak Pressure Ranking:** Identifies which ports experienced the highest pressure at any time.  
• **Scenario Toggles:** Change the seed port, spread, recovery, duration, and temporary closure to see different outcomes.  
• **Data Exports:** Download history, rankings, and a machine-readable summary.
"""


def why_it_matters_section() -> str:
    return """**Why This Matters (Business Value)**

• **Early Warning:** Monitor critical junctions where local issues can become network-wide problems.  
• **Reroute Playbooks:** Define thresholds that automatically trigger alternate ports or modes.  
• **Right-Sizing Buffers:** Place empty containers, chassis, and yard capacity where they reduce global risk the most.  
• **Supplier & Lane Diversification:** Avoid over-concentration on a few hubs or single corridors.  
• **Customer Communication:** Proactively adjust ETAs and set expectations when upstream pressure rises.
"""


# --------------------
# Streamlit page setup
# --------------------

st.set_page_config(page_title="Port Ripple Demo", layout="wide")
st.title("When a Port Sneezes, the World Catches a Cold")

# 0) SIMULATION CONTROLS (TOP)
st.markdown("**Simulation Controls**")
with st.form("controls_form", clear_on_submit=False):
    G = build_network()
    centrality = compute_centrality(G)
    cols = st.columns([1.2, 1, 1, 1, 1, 1])
    with cols[0]:
        seed_port = st.selectbox("Seed Port", sorted(G.nodes), index=sorted(G.nodes).index("Shanghai"))
    with cols[1]:
        steps = st.number_input("Time Steps", min_value=5, max_value=60, value=12, step=1)
    with cols[2]:
        spread = st.number_input("Spread Strength (0..1)", min_value=0.0, max_value=1.0, value=0.45, step=0.05, format="%.2f")
    with cols[3]:
        decay = st.number_input("Recovery per Step (0..1)", min_value=0.0, max_value=1.0, value=0.15, step=0.05, format="%.2f")
    with cols[4]:
        closure_steps = st.number_input("Seed Closure (steps)", min_value=0, max_value=20, value=0, step=1)
    with cols[5]:
        layout_seed = st.number_input("Layout Seed", min_value=0, max_value=9999, value=3, step=1)

    c_run, c_reset = st.columns([0.2, 0.8])
    submitted = c_run.form_submit_button("Run Simulation")
    reset = c_reset.form_submit_button("Reset to Defaults")

if reset:
    seed_port, steps, spread, decay, closure_steps, layout_seed = "Shanghai", 12, 0.45, 0.15, 0, 3

# Compute once based on current/last submitted values
if "last_params" not in st.session_state or submitted or reset:
    st.session_state.last_params = dict(
        seed=seed_port, steps=int(steps), spread=float(spread),
        decay=float(decay), closure_steps=int(closure_steps),
        layout_seed=int(layout_seed)
    )

params = st.session_state.last_params
df, pos = simulate(
    G.copy(),
    seed=params["seed"],
    steps=params["steps"],
    spread=params["spread"],
    decay=params["decay"],
    closure_steps=params["closure_steps"],
    layout_seed=params["layout_seed"],
)
peak, cent, ranks = make_rankings(df, centrality)

# 1) SUMMARY BLURB (auto-updates from controls)
st.markdown(summary_blurb(params["seed"], params["spread"], params["decay"], params["closure_steps"], ranks))

# 2) HOW TO READ
st.markdown(how_to_read_section())

# 3) TOOLS USED
st.markdown(tools_used_section())

# 4) WHY THIS MATTERS
st.markdown(why_it_matters_section())

st.markdown("---")

# 5) CHARTS
c1, c2 = st.columns([1.08, 0.92])

with c1:
    st.subheader("Final Ripple Map")
    fig_net = plot_network(G, df, pos, params["seed"])
    st.pyplot(fig_net, clear_figure=True)

with c2:
    st.subheader("Global Pressure Timeline")
    fig_tl, global_stress = plot_global_timeline(df)
    st.pyplot(fig_tl, clear_figure=True)

# 6) TABLES
st.subheader("Peak Pressure by Port")
st.dataframe(peak, use_container_width=True)

st.subheader("Network Importance (Betweenness Centrality)")
st.dataframe(cent.sort_values("betweenness", ascending=False), use_container_width=True)

st.subheader("Combined View (Peak vs Centrality)")
st.dataframe(ranks.sort_values(["rank_peak", "rank_centrality"]), use_container_width=True)

# 7) DOWNLOADS
st.markdown("### Downloads")
colA, colB, colC, colD, colE = st.columns(5)
with colA:
    st.download_button("Stress History CSV", data=df_to_csv_bytes(df), file_name="stress_history.csv")
with colB:
    st.download_button("Peak Ranking CSV", data=df_to_csv_bytes(peak), file_name="ranking_peak_stress.csv")
with colC:
    st.download_button("Centrality Ranking CSV", data=df_to_csv_bytes(cent), file_name="ranking_centrality.csv")
with colD:
    st.download_button("Combined Ranking CSV", data=df_to_csv_bytes(ranks), file_name="ranking_combined.csv")

summary_json = {
    "seed": params["seed"],
    "steps": params["steps"],
    "spread": params["spread"],
    "decay": params["decay"],
    "closure_steps": params["closure_steps"],
    "top_impacted_by_peak": peak.head(5).to_dict(orient="records"),
}
with colE:
    st.download_button(
        "Summary (JSON)",
        data=json.dumps(summary_json, indent=2).encode("utf-8"),
        file_name="summary.json",
        mime="application/json",
    )

st.download_button(
    "Summary Blurb (Markdown)",
    data=summary_blurb(params["seed"], params["spread"], params["decay"], params["closure_steps"], ranks).encode("utf-8"),
    file_name="summary_blurb.md",
    mime="text/markdown",
)

st.caption("Demonstration only. Uses default matplotlib styles; one chart per figure.")
