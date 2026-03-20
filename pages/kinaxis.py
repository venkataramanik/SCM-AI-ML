import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="ERP-Planning Integration Map", layout="wide")

st.title("🔗 ERP to Planning Integration Architect")
st.subheader("SAP/Oracle Integrated with Kinaxis/o9 Solutions")

# --- SIDEBAR: INTEGRATION CONTROLS ---
st.sidebar.header("Integration Settings")
int_tool = st.sidebar.selectbox(
    "Recommended Integration Tool",
    ["SAP Integration Suite (BTP)", "Oracle Integration Cloud (OIC)", "MuleSoft", "Informatica Cloud", "Workato"]
)

refresh_rate = st.sidebar.select_slider(
    "Data Sync Frequency",
    options=["Real-time (Webhooks)", "Hourly (API)", "Daily (Batch)", "Weekly"]
)

# --- MAIN INTERACTIVE DIAGRAM ---
st.write("### 1. Architectural Data Flow")
st.info("Click the buttons below to highlight specific data paths.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬆️ Inbound: Master Data"):
        view = "inbound_master"
    else:
        view = "full"

with col2:
    if st.button("🔄 Transactional Sync"):
        view = "transactional"
    else:
        view = "full"

with col3:
    if st.button("⬇️ Outbound: Commit to ERP"):
        view = "outbound"
    else:
        view = "full"

# Graphviz logic based on selection
def generate_graph(highlight):
    colors = {"erp": "#1f77b4", "planning": "#ff7f0e", "highlight": "#d62728"}
    
    dot_code = f"""
    digraph G {{
        rankdir=LR;
        node [shape=box, style=filled, fontname="Helvetica"];
        
        subgraph cluster_0 {{
            label = "ERP System (SAP S/4HANA / Oracle)";
            style=dashed;
            ERP_Master [label="Master Data\\n(Items, BOMs, Orgs)", fillcolor="{colors['highlight'] if highlight=='inbound_master' else colors['erp']}", fontcolor="white"];
            ERP_Trans [label="Transactions\\n(PO, SO, Inventory)", fillcolor="{colors['highlight'] if highlight=='transactional' else colors['erp']}", fontcolor="white"];
            ERP_Exec [label="Execution\\n(Production Orders)", fillcolor="{colors['highlight'] if highlight=='outbound' else colors['erp']}", fontcolor="white"];
        }}

        Middle [label="{int_tool}", shape=diamond, fillcolor="#eeeeee"];

        subgraph cluster_1 {{
            label = "Planning Engine (Kinaxis / o9)";
            style=dashed;
            Plan_Model [label="Digital Twin /\\nSupply Model", fillcolor="{colors['planning']}", fontcolor="white"];
            Plan_Solve [label="Constraint Solver /\\nScenario", fillcolor="{colors['highlight'] if highlight=='transactional' else colors['planning']}", fontcolor="white"];
            Plan_Commit [label="Confirmed Plan", fillcolor="{colors['highlight'] if highlight=='outbound' else colors['planning']}", fontcolor="white"];
        }}

        ERP_Master -> Middle [label="Weekly/Batch"];
        ERP_Trans -> Middle [label="Near Real-Time"];
        Middle -> Plan_Model;
        Middle -> Plan_Solve;
        Plan_Commit -> Middle [label="Actionable Recs"];
        Middle -> ERP_Exec [label="API Push", color="{colors['highlight'] if highlight=='outbound' else 'black'}"];
    }}
    """
    return dot_code

st.graphviz_chart(generate_graph(view))

# --- DATA FEEDS TABLE ---
st.write("### 2. Integration Data Catalog")

data_feeds = {
    "Direction": ["Inbound", "Inbound", "Inbound", "Outbound", "Outbound"],
    "Data Object": ["Material Master / BOM", "On-Hand Inventory", "Sales Orders", "Purchase Reqs", "Planned Orders"],
    "Frequency": ["Daily/Weekly", "Hourly", "Near Real-Time", "Intraday/On-Demand", "Daily"],
    "Mechanism": ["OData / IDoc", "REST API", "Webhooks / Event Mesh", "REST API", "BAPI / SOAP"],
    "Criticality": ["High (Master)", "High (Planning)", "Medium", "Critical (Execution)", "High"]
}

df = pd.DataFrame(data_feeds)
st.table(df)

# --- TECHNICAL RECOMMENDATIONS ---
st.divider()
st.write("### 3. Implementation Blueprint")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    **Bidirectional Strategy:**
    * **ERP to Planning:** Focus on 'Net-Change' logic to reduce payload size.
    * **Planning to ERP:** Use a 'Staging/Draft' area in the ERP before auto-releasing orders.
    """)

with c2:
    st.markdown(f"""
    **Tooling Strategy ({int_tool}):**
    * **Standard Adapters:** Use pre-built connectors (e.g., Kinaxis SAP Connector) to avoid custom ABAP/SQL.
    * **Error Handling:** Implement dead-letter queues for failed API calls back to ERP.
    """)

st.success(f"Current Dashboard configured for **{refresh_rate}** refresh cycles.")
