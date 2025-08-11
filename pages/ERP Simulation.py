# agentic_scm_app.py
# Streamlit Agentic SCM Simulator — Multi‑agent demo for PLAN / SOURCE / MAKE / DELIVER
# Run:  streamlit run agentic_scm_app.py

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Agentic SCM Simulator", layout="wide")

# ---------------------------
# Utilities & Random Seed
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------
# Environment & State
# ---------------------------
@dataclass
class Supplier:
    supplier_id: str
    lead_time_days: int
    on_time_prob: float
    country: str

@dataclass
class TransportLane:
    lane_id: str
    mode: str  # "Ocean" or "Air"
    transit_days: int
    expedite_cut: int  # days reduced if expedited (Ocean->Air effect)

@dataclass
class InboundPO:
    po_id: str
    item_id: str
    qty: int
    eta_day: int           # absolute sim day when arrives (1..H)
    lane_id: str
    expedited: bool = False

@dataclass
class SalesOrder:
    so_id: str
    item_id: str
    qty: int
    day_created: int
    due_day: int
    shipped_qty: int = 0
    ship_day: Optional[int] = None

@dataclass
class WorkCenter:
    wc_id: str
    daily_capacity_units: int

@dataclass
class Env:
    # Inventory: item -> units (single location)
    inv: Dict[str, int]
    # BOM: FG -> {RM: qty_per}
    bom: Dict[str, Dict[str, float]]
    # Work centers (one-step per unit for demo simplicity)
    wc: Dict[str, WorkCenter]
    # Suppliers and lanes
    suppliers: Dict[str, Supplier]
    lanes: Dict[str, TransportLane]

    # Demand model
    base_demand: Dict[str, int]  # expected daily mean demand per FG
    demand_sigma: int            # noise
    seasonality_amp: float       # weekly seasonality amplitude

    # Orders & inbound pipeline
    orders: List[SalesOrder] = field(default_factory=list)
    inbound: List[InboundPO] = field(default_factory=list)

    # Pricing / cost
    sales_price: Dict[str, float] = field(default_factory=lambda: {"FG-100": 120.0, "FG-200": 140.0})
    rm_cost: Dict[str, float] = field(default_factory=lambda: {"RM-10": 10.0, "RM-20": 5.0, "RM-30": 20.0})
    logistics_cost_accum: float = 0.0
    revenue_accum: float = 0.0
    cogs_accum: float = 0.0

    # Metrics history
    day_log: List[Dict] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)

# ---------------------------
# Agents
# ---------------------------
class DemandSensingAgent:
    """Adjusts demand forecast using EWMA + optional news shock."""
    def __init__(self, ewma_alpha: float, news_shock_pct: float):
        self.alpha = ewma_alpha
        self.news = news_shock_pct / 100.0
        self.last_forecast = {}

    def propose(self, t: int, env: Env, observed_orders: Dict[str, int]) -> Dict[str, int]:
        forecast = {}
        for fg, base in env.base_demand.items():
            hist = observed_orders.get(fg, base)
            prev = self.last_forecast.get(fg, base)
            ewma = self.alpha * hist + (1 - self.alpha) * prev
            season = 1.0 + env.seasonality_amp * math.sin(2 * math.pi * t / 7.0)
            forecast_qty = max(0, int(ewma * season * (1 + self.news)))
            forecast[fg] = forecast_qty
        self.last_forecast = forecast
        return forecast

class InventoryPlannerAgent:
    """(s, S) policy based on sensed forecast and desired service level."""
    def __init__(self, cycle_days: int, z_service: float, cov: float, ss_multiplier: float):
        self.cycle = cycle_days
        self.z = z_service         # e.g., 1.645 ~ 95%
        self.cov = cov             # coefficient of variation for demand
        self.ss_mult = ss_multiplier

    def propose(self, t: int, env: Env, forecast: Dict[str, int]) -> List[Tuple[str, int]]:
        proposals = []
        horizon = self.cycle
        for rm in [k for k in env.inv.keys() if k.startswith("RM")]:
            # Map RM demand from FG forecast via BOM
            implied_rm = 0
            for fg, bom in env.bom.items():
                if rm in bom:
                    implied_rm += int(forecast.get(fg, 0) * bom[rm])

            mu = implied_rm * horizon
            sigma = self.cov * mu**0.5
            safety = int(self.ss_mult * self.z * sigma)
            S = mu + safety
            s = int(mu * 0.4)  # reorder point heuristic

            on_hand = env.inv.get(rm, 0)
            pipeline = sum(p.qty for p in env.inbound if p.item_id == rm)
            pos = on_hand + pipeline
            if pos < s:
                order_qty = max(0, int(S - pos))
                if order_qty > 0:
                    proposals.append((rm, order_qty))
        return proposals

class ProductionSchedulerAgent:
    """Greedy EDF (earliest-due-first) production under capacity."""
    def __init__(self, wc_id: str):
        self.wc_id = wc_id

    def propose(self, t: int, env: Env, capacity_units: int) -> List[Tuple[str, int]]:
        # Demand today to cover orders due in <= X days (simple: today’s created)
        todays = [o for o in env.orders if o.day_created == t]
        by_item_need = {}
        for o in todays:
            by_item_need[o.item_id] = by_item_need.get(o.item_id, 0) + o.qty

        # Greedy: produce items with highest backlog first
        plan = []
        cap_left = capacity_units
        for fg, need in sorted(by_item_need.items(), key=lambda x: -x[1]):
            # Check RM availability
            max_by_rm = 10**9
            for rm, q in env.bom[fg].items():
                have = env.inv.get(rm, 0)
                max_by_rm = min(max_by_rm, int(have // q))
            qty_make = min(need, cap_left, max_by_rm)
            if qty_make > 0:
                plan.append((fg, qty_make))
                cap_left -= qty_make
        return plan

class ExpediteLogisticsAgent:
    """If backlog risk exceeds threshold, convert future inbound to Air (faster)."""
    def __init__(self, backlog_threshold: float):
        self.thr = backlog_threshold

    def propose(self, t: int, env: Env) -> List[str]:
        open_need = sum(max(0, o.qty - o.shipped_qty) for o in env.orders if (o.ship_day is None))
        fg_on_hand = sum(env.inv.get(x, 0) for x in env.inv.keys() if x.startswith("FG"))
        ratio = 0.0 if open_need == 0 else max(0.0, (open_need - fg_on_hand) / open_need)
        actions = []
        if ratio >= self.thr:
            for po in env.inbound:
                lane = env.lanes[po.lane_id]
                if (po.eta_day - t) > 8 and lane.mode == "Ocean" and not po.expedited:
                    po.eta_day = max(t + 1, po.eta_day - lane.expedite_cut)
                    po.expedited = True
                    env.logistics_cost_accum += 0.2 * po.qty  # placeholder uplift
                    actions.append(f"Expedited {po.po_id} (eta->{po.eta_day})")
        return actions

class SupplierRiskAgent:
    """Injects random supplier delays based on reliability."""
    def __init__(self, daily_check_prob: float = 0.15):
        self.p = daily_check_prob

    def propose(self, t: int, env: Env) -> List[str]:
        msgs = []
        if random.random() > self.p:
            return msgs
        for po in env.inbound:
            sup = None
            # crude: infer by lane country (demo only)
            if po.lane_id.startswith("CN"):
                sup = next((s for s in env.suppliers.values() if s.country == "CN"), None)
            else:
                sup = next((s for s in env.suppliers.values() if s.country == "IN"), None)
            if sup and random.random() > sup.on_time_prob:
                delay = random.randint(1, 5)
                po.eta_day += delay
                msgs.append(f"Supplier delay on {po.po_id}: +{delay}d")
        return msgs

# ---------------------------
# Coordinator
# ---------------------------
class Orchestrator:
    def __init__(self, env: Env, agents: dict):
        self.env = env
        self.agents = agents
        self.po_counter = 1
        self.so_counter = 1

    def _gen_po_id(self) -> str:
        pid = f"PO-{self.po_counter:05d}"; self.po_counter += 1; return pid

    def _gen_so_id(self) -> str:
        sid = f"SO-{self.so_counter:05d}"; self.so_counter += 1; return sid

    def _simulate_orders_today(self, t: int, demand_forecast: Dict[str, int], promise_days_choices=(3,5,7)) -> Dict[str,int]:
        observed = {}
        for fg, mu in demand_forecast.items():
            # Poisson-ish with noise
            qty = max(0, int(np.random.normal(mu, max(3, mu * 0.15))))
            # Split into 1-3 orders
            n_orders = np.random.choice([1,2,3], p=[0.55,0.35,0.10])
            split = np.random.multinomial(qty, [1/n_orders]*n_orders) if qty > 0 else [0]
            for j, q in enumerate(split):
                if q > 0:
                    due = t + int(np.random.choice(promise_days_choices, p=[0.4,0.4,0.2]))
                    self.env.orders.append(SalesOrder(
                        so_id=self._gen_so_id(), item_id=fg, qty=int(q),
                        day_created=t, due_day=due
                    ))
            observed[fg] = qty
        return observed

    def _receive_inbound(self, t: int):
        arrived = [po for po in self.env.inbound if po.eta_day == t]
        for po in arrived:
            self.env.inv[po.item_id] = self.env.inv.get(po.item_id, 0) + po.qty
        self.env.inbound = [po for po in self.env.inbound if po.eta_day != t]
        return [po.po_id for po in arrived]

    def _place_rm_pos(self, t: int, proposals: List[Tuple[str, int]], lane_map: Dict[str,str]):
        for (rm, qty) in proposals:
            lane_id = lane_map.get(rm, "IN-US-Ocean")
            lane = self.env.lanes[lane_id]
            eta = t + lane.transit_days
            # Simple supplier reliability (baked into risk agent later)
            po = InboundPO(po_id=self._gen_po_id(), item_id=rm, qty=int(qty), eta_day=int(eta), lane_id=lane_id)
            self.env.inbound.append(po)
            self.env.action_log.append(f"Placed {po.po_id} for {rm} {qty} via {lane.mode} (ETA {po.eta_day})")

    def _produce(self, t: int, plan: List[Tuple[str,int]]):
        for (fg, qty) in plan:
            # consume RM
            for rm, q in self.env.bom[fg].items():
                self.env.inv[rm] = self.env.inv.get(rm, 0) - int(q * qty)
            # add FG
            self.env.inv[fg] = self.env.inv.get(fg, 0) + int(qty)
            self.env.action_log.append(f"Built {qty} of {fg}")

    def _ship(self, t: int):
        # earliest-due-first shipping
        open_orders = [o for o in self.env.orders if o.ship_day is None]
        open_orders.sort(key=lambda o: o.due_day)
        for o in open_orders:
            avail = self.env.inv.get(o.item_id, 0)
            ship = min(avail, o.qty - o.shipped_qty)
            if ship > 0:
                self.env.inv[o.item_id] = avail - ship
                o.shipped_qty += ship
                o.ship_day = t
                # revenue & COGS
                self.env.revenue_accum += ship * self.env.sales_price[o.item_id]
                unit_cogs = 0.0
                for rm, rmq in self.env.bom[o.item_id].items():
                    unit_cogs += self.env.rm_cost.get(rm, 0.0) * rmq
                self.env.cogs_accum += ship * unit_cogs

    def step(self, t: int, cfg: dict):
        # 1) Demand sensing → forecast
        observed_prev = {}  # will be computed after orders
        sensed = self.agents["demand"].propose(t, self.env, observed_prev or {})

        # 2) Realize orders for today (using sensed as mean)
        observed_today = self._simulate_orders_today(t, sensed)
        # update sensing next step
        sensed = self.agents["demand"].propose(t, self.env, observed_today)

        # 3) Receive inbound
        arrived_ids = self._receive_inbound(t)

        # 4) Planner (RM POs)
        planner_props = self.agents["planner"].propose(t, self.env, sensed)
        self._place_rm_pos(t, planner_props, cfg["lane_map"])

        # 5) Supplier risk events
        risk_msgs = self.agents["supplier_risk"].propose(t, self.env)

        # 6) Expedite if backlog risk high
        exp_msgs = self.agents["expedite"].propose(t, self.env)

        # 7) Production scheduling
        cap = self.env.wc["WC-ASM"].daily_capacity_units if cfg["apply_capacity"] else 10**9
        prod_plan = self.agents["scheduler"].propose(t, self.env, cap)
        self._produce(t, prod_plan)

        # 8) Ship
        self._ship(t)

        # KPIs
        todays = [o for o in self.env.orders if o.day_created == t]
        if len(todays) > 0:
            otif = sum(1 for o in todays if (o.ship_day is not None and o.ship_day <= o.due_day and o.shipped_qty >= o.qty)) / len(todays)
        else:
            otif = np.nan
        inv_units = sum(v for k, v in self.env.inv.items() if k.startswith(("RM","FG")))
        backlog = sum(max(0, o.qty - o.shipped_qty) for o in self.env.orders if o.ship_day is None)
        self.env.day_log.append({
            "day": t,
            "OTIF": otif,
            "InventoryUnits": inv_units,
            "Revenue": self.env.revenue_accum,
            "COGS": self.env.cogs_accum,
            "BacklogUnits": backlog,
            "Arrivals": ",".join(arrived_ids) if arrived_ids else "",
            "PlacedPOs": sum(q for _, q in planner_props) if planner_props else 0,
            "Expedites": len(exp_msgs),
        })
        self.env.action_log.extend(risk_msgs)
        self.env.action_log.extend(exp_msgs)

# ---------------------------
# Default Dataset Generator
# ---------------------------
def build_default_env(seed: int = 42) -> Env:
    set_seed(seed)
    suppliers = {
        "SUP-IN-01": Supplier("SUP-IN-01", lead_time_days=14, on_time_prob=0.92, country="IN"),
        "SUP-IN-02": Supplier("SUP-IN-02", lead_time_days=18, on_time_prob=0.90, country="IN"),
        "SUP-CN-01": Supplier("SUP-CN-01", lead_time_days=20, on_time_prob=0.88, country="CN"),
    }
    lanes = {
        "IN-US-Ocean": TransportLane("IN-US-Ocean", mode="Ocean", transit_days=30, expedite_cut=22),
        "CN-US-Ocean": TransportLane("CN-US-Ocean", mode="Ocean", transit_days=28, expedite_cut=20),
        "IN-US-Air":   TransportLane("IN-US-Air",   mode="Air",   transit_days=6,  expedite_cut=0),
        "CN-US-Air":   TransportLane("CN-US-Air",   mode="Air",   transit_days=5,  expedite_cut=0),
    }
    env = Env(
        inv={"FG-100": 300, "FG-200": 300, "RM-10": 4000, "RM-20": 3000, "RM-30": 2500},
        bom={"FG-100": {"RM-10": 2.0, "RM-20": 1.0, "RM-30": 1.0},
             "FG-200": {"RM-10": 1.0, "RM-20": 2.0, "RM-30": 1.0}},
        wc={"WC-ASM": WorkCenter("WC-ASM", daily_capacity_units=800)},
        suppliers=suppliers,
        lanes=lanes,
        base_demand={"FG-100": 180, "FG-200": 160},
        demand_sigma=15,
        seasonality_amp=0.15,
    )
    return env

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Agentic SCM Simulator (Multi‑Agent, Streamlit)")

with st.sidebar:
    st.header("Simulation Settings")
    days = st.slider("Horizon (days)", 30, 150, 60, 5)
    seed = st.number_input("Random Seed", value=42, step=1)
    apply_capacity = st.checkbox("Apply Assembly Capacity", value=True)
    asm_cap = st.number_input("ASM Daily Capacity (units)", value=800, step=50)

    st.header("Demand Sensing")
    ewma_alpha = st.slider("EWMA α", 0.05, 0.9, 0.3, 0.05)
    news_shock_pct = st.slider("News Shock (%)", -50, 200, 0, 5)

    st.header("Inventory Planner")
    cycle_days = st.slider("Review Cycle (days)", 5, 30, 14, 1)
    z_service = st.selectbox("Service Level (z)", [1.28, 1.44, 1.645, 1.96], index=2, help="~90%, 92.5%, 95%, 97.5%")
    cov = st.slider("CoV (demand uncertainty)", 0.05, 0.6, 0.25, 0.05)
    ss_mult = st.slider("Safety Stock Multiplier", 0.5, 2.0, 1.0, 0.1)

    st.header("Expedite Logic")
    backlog_thr = st.slider("Expedite if Backlog ≥", 0.0, 1.0, 0.25, 0.05)

    st.header("Lanes (for RM)")
    lane_map = {
        "RM-10": st.selectbox("RM-10 Lane", ["IN-US-Ocean", "IN-US-Air"], index=0),
        "RM-20": st.selectbox("RM-20 Lane", ["IN-US-Ocean", "IN-US-Air"], index=0),
        "RM-30": st.selectbox("RM-30 Lane", ["CN-US-Ocean", "CN-US-Air"], index=0),
    }

run = st.button("Run Simulation")

if run:
    env = build_default_env(seed)
    env.wc["WC-ASM"].daily_capacity_units = int(asm_cap)

    orchestrator = Orchestrator(
        env=env,
        agents={
            "demand": DemandSensingAgent(ewma_alpha=ewma_alpha, news_shock_pct=news_shock_pct),
            "planner": InventoryPlannerAgent(cycle_days=cycle_days, z_service=float(z_service), cov=cov, ss_multiplier=ss_mult),
            "scheduler": ProductionSchedulerAgent(wc_id="WC-ASM"),
            "expedite": ExpediteLogisticsAgent(backlog_threshold=backlog_thr),
            "supplier_risk": SupplierRiskAgent(daily_check_prob=0.20),
        }
    )
    cfg = {"apply_capacity": apply_capacity, "lane_map": lane_map}

    for t in range(1, days + 1):
        orchestrator.step(t, cfg)

    log = pd.DataFrame(env.day_log)
    st.success("Simulation complete.")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    latest = log.dropna(subset=["OTIF"]).iloc[-1] if len(log.dropna(subset=["OTIF"])) > 0 else None
    with c1:
        st.metric("OTIF (latest day)", f"{(latest['OTIF']*100):.1f}%" if latest is not None else "n/a")
    with c2:
        st.metric("Inventory Units", f"{int(log['InventoryUnits'].iloc[-1])}")
    with c3:
        st.metric("Revenue (cum)", f"${log['Revenue'].iloc[-1]:,.0f}")
    with c4:
        st.metric("COGS (cum)", f"${log['COGS'].iloc[-1]:,.0f}")

    # Charts — one per plot, no explicit colors/styles
    fig1, ax1 = plt.subplots()
    ax1.plot(log["day"], log["OTIF"])
    ax1.set_title("Daily OTIF")
    ax1.set_xlabel("Day"); ax1.set_ylabel("OTIF")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(log["day"], log["InventoryUnits"])
    ax2.set_title("Inventory Units Over Time")
    ax2.set_xlabel("Day"); ax2.set_ylabel("Units")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(log["day"], log["Revenue"], label="Revenue")
    ax3.plot(log["day"], log["COGS"], label="COGS")
    ax3.set_title("Revenue vs COGS (Cumulative)")
    ax3.set_xlabel("Day"); ax3.set_ylabel("USD")
    ax3.legend()
    st.pyplot(fig3)

    # Tables
    st.subheader("Daily Log (head)")
    st.dataframe(log.head(20))

    # Orders and inbound snapshots
    orders_df = pd.DataFrame([o.__dict__ for o in env.orders])
    inbound_df = pd.DataFrame([p.__dict__ for p in env.inbound]) if env.inbound else pd.DataFrame(columns=["po_id","item_id","qty","eta_day","lane_id","expedited"])
    st.subheader("Open Orders (sample)")
    st.dataframe(orders_df.head(50))
    st.subheader("Outstanding Inbound POs")
    st.dataframe(inbound_df)

    # Action log
    st.subheader("Agent Actions / Events")
    st.text("\n".join(env.action_log[-200:]))

    # Downloads
    t1, t2, t3 = st.tabs(["Download KPI Log", "Download Orders", "Download Action Log"])
    with t1:
        st.download_button("Download KPI Log (CSV)", data=log.to_csv(index=False).encode("utf-8"),
                           file_name="agentic_kpis.csv", mime="text/csv")
    with t2:
        st.download_button("Download Orders (CSV)", data=orders_df.to_csv(index=False).encode("utf-8"),
                           file_name="orders.csv", mime="text/csv")
    with t3:
        actions_df = pd.DataFrame({"action": env.action_log})
        st.download_button("Download Actions (CSV)", data=actions_df.to_csv(index=False).encode("utf-8"),
                           file_name="actions.csv", mime="text/csv")

else:
    st.info("Configure parameters in the sidebar, then click **Run Simulation**.")

st.markdown("---")
st.markdown(
    "**Agents:** DemandSensing → InventoryPlanner → SupplierRisk → ExpediteLogistics → ProductionScheduler  \n"
    "**SCOR:** PLAN (sensing/forecast, reorder) • SOURCE (POs, lanes, risk) • MAKE (capacity, build) • DELIVER (ship/OTIF) • RETURN (extend) • ENABLE (policies, data)"
)
