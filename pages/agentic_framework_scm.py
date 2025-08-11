# agentic_framework_scm.py
# Streamlit SCM Agentic Framework Demo: Perceive → Plan → Act → Reflect with Tools + Blackboard
# Run: streamlit run agentic_framework_scm.py

import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Agentic SCM Framework (SCOR)", layout="wide")

# ============ Core Data Models ============
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
class InboundPO:
    po_id: str
    item_id: str
    qty: int
    eta_day: int
    mode: str  # "Ocean"|"Air"
    expedited: bool = False

@dataclass
class Env:
    inv: Dict[str, int]
    bom: Dict[str, Dict[str, float]]
    prices: Dict[str, float]
    rm_cost: Dict[str, float]
    ocean_transit: Dict[str, int]    # RM -> days
    air_transit: Dict[str, int]      # RM -> days
    suppliers_reliability: Dict[str, float]  # RM -> on-time prob

    base_demand: Dict[str, int]  # FG -> mean
    season_amp: float
    demand_sigma: float

    orders: List[SalesOrder] = field(default_factory=list)
    inbound: List[InboundPO] = field(default_factory=list)

    revenue: float = 0.0
    cogs: float = 0.0
    logistics_cost: float = 0.0

# ============ Blackboard / Memory ============
@dataclass
class Blackboard:
    """Shared state visible to all agents (perception -> planning -> action -> reflection)."""
    day: int = 0
    forecast: Dict[str, int] = field(default_factory=dict)        # PERC
    backlog_ratio: float = 0.0                                    # PERC
    reorder_recs: List[Tuple[str, int]] = field(default_factory=list)  # PLAN
    prod_plan: List[Tuple[str, int]] = field(default_factory=list)     # PLAN
    actions: List[str] = field(default_factory=list)               # ACT
    reflections: List[str] = field(default_factory=list)           # REFLECT
    policy: Dict[str, float] = field(default_factory=lambda: {
        "service_z": 1.645,       # ~95%
        "cov": 0.25,              # demand uncertainty
        "ss_mult": 1.0,           # safety stock multiplier
        "expedite_thr": 0.25,     # backlog ratio trigger
        "review_cycle": 14,       # days
    })
    trace: List[Dict] = field(default_factory=list)                # KPI snapshots

# ============ Tools (callable by agents) ============
class ForecastTool:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.last = {}

    def __call__(self, t: int, env: Env, observed: Dict[str, int]) -> Dict[str, int]:
        out = {}
        for fg, mu in env.base_demand.items():
            prev = self.last.get(fg, mu)
            season = 1.0 + env.season_amp * math.sin(2*math.pi*t/7.0)
            obs = observed.get(fg, mu)
            ewma = self.alpha*obs + (1-self.alpha)*prev
            out[fg] = max(0, int(ewma*season))
        self.last = out
        return out

class InventoryPolicyTool:
    """Compute (s,S) from forecast + policy."""
    def __call__(self, bb: Blackboard, env: Env) -> List[Tuple[str, int]]:
        z, cov, ss_mult, cycle = bb.policy["service_z"], bb.policy["cov"], bb.policy["ss_mult"], int(bb.policy["review_cycle"])
        proposals = []
        for rm in [k for k in env.inv if k.startswith("RM")]:
            implied_period = 0
            for fg, bom in env.bom.items():
                if rm in bom:
                    implied_period += int(bb.forecast.get(fg, 0)*bom[rm])
            mu = implied_period * cycle
            sigma = cov * max(1, mu)**0.5
            S = int(mu + ss_mult*z*sigma)
            s = int(mu * 0.4)
            on_hand = env.inv.get(rm, 0)
            pipe = sum(p.qty for p in env.inbound if p.item_id == rm)
            pos = on_hand + pipe
            if pos < s:
                proposals.append((rm, max(0, S - pos)))
        return proposals

class POPlacementTool:
    """Create POs; default mode Ocean; return PO ids."""
    def __init__(self):
        self.po_counter = 1
    def __call__(self, t: int, env: Env, rm: str, qty: int, mode: str = "Ocean") -> str:
        po_id = f"PO-{self.po_counter:05d}"; self.po_counter += 1
        transit = env.ocean_transit[rm] if mode=="Ocean" else env.air_transit[rm]
        eta = t + transit
        env.inbound.append(InboundPO(po_id, rm, int(qty), int(eta), mode))
        return po_id

class ExpediteTool:
    """Convert future Ocean POs to Air (advance ETA, extra cost)."""
    def __call__(self, t: int, env: Env, advance_days: int = 20, cost_per_unit: float = 0.2) -> int:
        changed = 0
        for po in env.inbound:
            if po.mode == "Ocean" and not po.expedited and (po.eta_day - t) > 8:
                po.mode = "Air"
                po.eta_day = max(t+1, po.eta_day - advance_days)
                po.expedited = True
                env.logistics_cost += cost_per_unit * po.qty
                changed += 1
        return changed

class ProductionTool:
    """Greedy: make FG limited by RM availability and (optional) capacity."""
    def __call__(self, env: Env, plan: List[Tuple[str,int]], capacity: Optional[int]) -> List[str]:
        msgs = {}
        out_msgs = []
        cap_left = capacity if capacity is not None else 1_000_000
        for fg, need in plan:
            if cap_left <= 0: break
            max_rm = 10**9
            for rm, q in env.bom[fg].items():
                max_rm = min(max_rm, int(env.inv.get(rm, 0) // q))
            qty = min(need, cap_left, max_rm)
            if qty > 0:
                for rm, q in env.bom[fg].items():
                    env.inv[rm] -= int(q*qty)
                env.inv[fg] = env.inv.get(fg, 0) + qty
                cap_left -= qty
                out_msgs.append(f"Built {qty} {fg}")
        return out_msgs

class ShippingTool:
    """Ship EDF; returns messages, updates revenue/COGS."""
    def __call__(self, t: int, env: Env) -> List[str]:
        msgs = []
        open_orders = [o for o in env.orders if o.ship_day is None]
        open_orders.sort(key=lambda o: o.due_day)
        for o in open_orders:
            avail = env.inv.get(o.item_id, 0)
            ship = min(avail, o.qty - o.shipped_qty)
            if ship > 0:
                env.inv[o.item_id] = avail - ship
                o.shipped_qty += ship
                o.ship_day = t
                env.revenue += ship*env.prices[o.item_id]
                unit_cogs = sum(env.rm_cost[rm]*q for rm,q in env.bom[o.item_id].items())
                env.cogs += ship*unit_cogs
                msgs.append(f"Shipped {ship} on {o.so_id}")
        return msgs

# ============ Agents (Perceive/Plan/Act/Reflect) ============
class BaseAgent:
    def __init__(self, name: str): self.name = name
    def sense(self, t, env: Env, bb: Blackboard): return {}
    def think(self, t, env: Env, bb: Blackboard, percept): return {}
    def act(self, t, env: Env, bb: Blackboard, plan): return []
    def reflect(self, t, env: Env, bb: Blackboard, result): return ""

class PlannerAgent(BaseAgent):
    def __init__(self, forecast_tool: ForecastTool, policy_tool: InventoryPolicyTool):
        super().__init__("Planner")
        self.forecast_tool = forecast_tool; self.policy_tool = policy_tool

    def sense(self, t, env, bb):
        # observe orders created today
        obs = {}
        todays = [o for o in env.orders if o.day_created==t]
        for o in todays: obs[o.item_id] = obs.get(o.item_id, 0)+o.qty
        return obs

    def think(self, t, env, bb, percept):
        bb.forecast = self.forecast_tool(t, env, percept)
        return {"reorders": self.policy_tool(bb, env)}

    def act(self, t, env, bb, plan):
        bb.reorder_recs = plan["reorders"]
        return [f"Reorder recs: {bb.reorder_recs}"]

    def reflect(self, t, env, bb, result):
        return "Planner updated forecast and reorder proposals."

class SourcingAgent(BaseAgent):
    def __init__(self, place_tool: POPlacementTool, expedite_tool: ExpediteTool):
        super().__init__("Sourcing")
        self.place, self.expedite = place_tool, expedite_tool

    def sense(self, t, env, bb):
        # backlog ratio
        open_need = sum(max(0,o.qty-o.shipped_qty) for o in env.orders if o.ship_day is None)
        fg_onhand = sum(env.inv.get(k,0) for k in env.inv if k.startswith("FG"))
        ratio = 0 if open_need==0 else max(0,(open_need-fg_onhand)/open_need)
        bb.backlog_ratio = ratio
        return {"backlog_ratio": ratio}

    def think(self, t, env, bb, percept):
        # choose mode if backlog risk is high
        high_risk = percept["backlog_ratio"] >= bb.policy["expedite_thr"]
        return {"mode": ("Air" if high_risk else "Ocean"), "high_risk": high_risk}

    def act(self, t, env, bb, plan):
        msgs = []
        for rm, qty in bb.reorder_recs:
            pid = self.place(t, env, rm, qty, mode=plan["mode"])
            msgs.append(f"Placed {pid} {rm} {qty} via {plan['mode']}")
        if plan["high_risk"]:
            changed = self.expedite(t, env, advance_days=20, cost_per_unit=0.2)
            if changed: msgs.append(f"Expedited {changed} open POs to Air")
        return msgs

    def reflect(self, t, env, bb, result):
        return f"Sourcing used mode={('Air' if bb.backlog_ratio>=bb.policy['expedite_thr'] else 'Ocean')}."

class ProductionAgent(BaseAgent):
    def __init__(self, production_tool: ProductionTool, capacity_units: int):
        super().__init__("Production")
        self.prod = production_tool; self.cap0 = capacity_units

    def sense(self, t, env, bb):
        due_soon = [o for o in env.orders if (o.ship_day is None and o.due_day <= t+2)]
        need = {}
        for o in due_soon: need[o.item_id] = need.get(o.item_id,0) + (o.qty-o.shipped_qty)
        return {"need": need}

    def think(self, t, env, bb, percept):
        # earliest-due-first simplified: produce what’s due soon
        plan = [(fg, q) for fg,q in percept["need"].items()]
        return {"plan": plan, "cap": self.cap0}

    def act(self, t, env, bb, plan):
        bb.prod_plan = plan["plan"]
        return self.prod(env, plan["plan"], capacity=plan["cap"])

    def reflect(self, t, env, bb, result):
        made = 0
        if result:
            for x in result:
                try:
                    made += int(x.split()[1])
                except Exception:
                    pass
        return f"Production made {made} units across SKUs."

class DeliveryAgent(BaseAgent):
    def __init__(self, ship_tool: ShippingTool):
        super().__init__("Delivery"); self.ship = ship_tool
    def sense(self, t, env, bb): return {}
    def think(self, t, env, bb, percept): return {}
    def act(self, t, env, bb, plan): return self.ship(t, env)
    def reflect(self, t, env, bb, result): return f"Delivery shipped {len(result)} lines."

class CriticAgent(BaseAgent):
    """Reflects on KPI deltas; adapts policy (agentic reflection)."""
    def __init__(self, target_otif=0.95):
        super().__init__("Critic")
        self.target = target_otif
        self.history = []

    def sense(self, t, env, bb):
        # compute today OTIF
        todays = [o for o in env.orders if o.day_created==t]
        otif = (sum(1 for o in todays if (o.ship_day is not None and o.ship_day<=o.due_day and o.shipped_qty>=o.qty))/len(todays)) if todays else np.nan
        inv_units = sum(v for k,v in env.inv.items() if k.startswith(("RM","FG")))
        return {"OTIF": otif, "InvUnits": inv_units}

    def think(self, t, env, bb, percept):
        self.history.append(percept)
        # simple rules: if OTIF low -> raise ss_mult; if InvUnits high & OTIF ok -> lower ss_mult
        delta = {}
        if not math.isnan(percept["OTIF"]):
            if percept["OTIF"] < self.target:
                delta["ss_mult"] = min(2.0, bb.policy["ss_mult"] + 0.1)
                delta["expedite_thr"] = max(0.05, bb.policy["expedite_thr"] - 0.05)
            elif percept["InvUnits"] > 9000:
                delta["ss_mult"] = max(0.5, bb.policy["ss_mult"] - 0.1)
        return {"policy_delta": delta}

    def act(self, t, env, bb, plan):
        msgs = []
        for k,v in plan["policy_delta"].items():
            old = bb.policy[k]; bb.policy[k] = v
            msgs.append(f"Policy update: {k} {old:.2f} → {v:.2f}")
        return msgs

    def reflect(self, t, env, bb, result):
        return "Critic adjusted policy based on KPI trend."

# ============ Orchestrator ============
class Orchestrator:
    def __init__(self, env: Env, bb: Blackboard, agents: List[BaseAgent]):
        self.env, self.bb, self.agents = env, bb, agents
        self.so_counter = 1

    def _create_orders(self, t: int, means: Dict[str,int]):
        for fg, mu in means.items():
            qty = max(0, int(np.random.normal(mu, max(3, mu*0.15))))
            n = np.random.choice([1,2,3], p=[0.55,0.35,0.10])
            split = np.random.multinomial(qty, [1/n]*n) if qty>0 else [0]
            for q in split:
                if q>0:
                    due = t + int(np.random.choice([3,5,7], p=[0.4,0.4,0.2]))
                    so = SalesOrder(f"SO-{self.so_counter:05d}", fg, int(q), t, due)
                    self.so_counter += 1
                    self.env.orders.append(so)

    def _receive_inbound(self, t: int):
        arrived = [po for po in self.env.inbound if po.eta_day==t]
        for po in arrived:
            self.env.inv[po.item_id] = self.env.inv.get(po.item_id,0)+po.qty
        self.env.inbound = [po for po in self.env.inbound if po.eta_day!=t]
        return arrived

    def step(self, t: int):
        self.bb.day = t
        day_trace = {"day": t, "logs": []}

        # Perceive → (Planner) produce forecast for order creation itself (open loop starting point)
        sensed = {"Planner": self.agents[0].sense(t, self.env, self.bb)}
        plan   = {"Planner": self.agents[0].think(t, self.env, self.bb, sensed["Planner"])}
        self._create_orders(t, self.bb.forecast or self.env.base_demand)  # create demand
        acted  = {"Planner": self.agents[0].act(t, self.env, self.bb, plan["Planner"])}
        refl   = {"Planner": self.agents[0].reflect(t, self.env, self.bb, acted["Planner"])}
        day_trace["logs"].append(("Planner", sensed["Planner"], plan["Planner"], acted["Planner"], refl["Planner"]))

        # Stochastic supplier delays (world events)
        for po in self.env.inbound:
            if random.random() > self.env.suppliers_reliability.get(po.item_id, 0.9):
                po.eta_day += random.randint(1,3)

        # Receive materials (before sourcing decisions)
        arrivals = self._receive_inbound(t)
        if arrivals:
            self.bb.actions.append(f"Received {[po.po_id for po in arrivals]}")

        # Sourcing
        sensed_s = self.agents[1].sense(t, self.env, self.bb)
        plan_s   = self.agents[1].think(t, self.env, self.bb, sensed_s)
        acted_s  = self.agents[1].act(t, self.env, self.bb, plan_s)
        refl_s   = self.agents[1].reflect(t, self.env, self.bb, acted_s)
        day_trace["logs"].append(("Sourcing", sensed_s, plan_s, acted_s, refl_s))

        # Production
        sensed_p = self.agents[2].sense(t, self.env, self.bb)
        plan_p   = self.agents[2].think(t, self.env, self.bb, sensed_p)
        acted_p  = self.agents[2].act(t, self.env, self.bb, plan_p)
        refl_p   = self.agents[2].reflect(t, self.env, self.bb, acted_p)
        day_trace["logs"].append(("Production", sensed_p, plan_p, acted_p, refl_p))

        # Delivery
        sensed_d = self.agents[3].sense(t, self.env, self.bb)
        plan_d   = self.agents[3].think(t, self.env, self.bb, sensed_d)
        acted_d  = self.agents[3].act(t, self.env, self.bb, plan_d)
        refl_d   = self.agents[3].reflect(t, self.env, self.bb, acted_d)
        day_trace["logs"].append(("Delivery", sensed_d, plan_d, acted_d, refl_d))

        # Critic (reflection → policy learning)
        sensed_c = self.agents[4].sense(t, self.env, self.bb)
        plan_c   = self.agents[4].think(t, self.env, self.bb, sensed_c)
        acted_c  = self.agents[4].act(t, self.env, self.bb, plan_c)
        refl_c   = self.agents[4].reflect(t, self.env, self.bb, acted_c)
        day_trace["logs"].append(("Critic", sensed_c, plan_c, acted_c, refl_c))

        # KPIs for charts
        todays = [o for o in self.env.orders if o.day_created==t]
        otif = (sum(1 for o in todays if (o.ship_day is not None and o.ship_day<=o.due_day and o.shipped_qty>=o.qty))/len(todays)) if todays else np.nan
        inv_units = sum(v for k,v in self.env.inv.items() if k.startswith(("RM","FG")))
        backlog = sum(max(0,o.qty-o.shipped_qty) for o in self.env.orders if o.ship_day is None)
        self.bb.trace.append({
            "day": t, "OTIF": otif, "InvUnits": inv_units,
            "Revenue": self.env.revenue, "COGS": self.env.cogs,
            "BacklogUnits": backlog, "Arrivals": len(arrivals)
        })
        return day_trace

# ============ Default Environment ============
def build_env(seed=42) -> Env:
    random.seed(seed); np.random.seed(seed)
    return Env(
        inv={"FG-100":300,"FG-200":300,"RM-10":4000,"RM-20":3000,"RM-30":2500},
        bom={"FG-100":{"RM-10":2.0,"RM-20":1.0,"RM-30":1.0},
             "FG-200":{"RM-10":1.0,"RM-20":2.0,"RM-30":1.0}},
        prices={"FG-100":120.0,"FG-200":140.0},
        rm_cost={"RM-10":10.0,"RM-20":5.0,"RM-30":20.0},
        ocean_transit={"RM-10":30,"RM-20":30,"RM-30":28},
        air_transit={"RM-10":6,"RM-20":6,"RM-30":5},
        suppliers_reliability={"RM-10":0.92,"RM-20":0.90,"RM-30":0.88},
        base_demand={"FG-100":180,"FG-200":160},
        season_amp=0.15, demand_sigma=15
    )

# ============ UI ============
st.title("Agentic SCM Framework — SCOR: Plan • Source • Make • Deliver • (Reflect)")

with st.sidebar:
    st.header("Run Settings")
    days = st.slider("Horizon (days)", 30, 150, 60, 5)
    seed = st.number_input("Random Seed", value=42, step=1)
    asm_cap = st.number_input("Assembly Capacity (units/day)", value=800, step=50)

    st.header("Policy (initial)")
    ss_mult = st.slider("Safety Stock Multiplier", 0.5, 2.0, 1.0, 0.1)
    expedite_thr = st.slider("Expedite if Backlog ≥", 0.0, 1.0, 0.25, 0.05)
    cov = st.slider("Demand CoV", 0.05, 0.6, 0.25, 0.05)
    z = st.selectbox("Service z", [1.28,1.44,1.645,1.96], index=2)
    review_cycle = st.slider("Review Cycle (days)", 5, 30, 14, 1)

run = st.button("Run Agentic Simulation")

if run:
    env = build_env(seed)
    bb  = Blackboard()
    bb.policy.update({"ss_mult": float(ss_mult), "expedite_thr": float(expedite_thr),
                      "cov": float(cov), "service_z": float(z), "review_cycle": int(review_cycle)})

    # Agents wired with tools (tool-use == “actions”)
    agents = [
        PlannerAgent(ForecastTool(alpha=0.3), InventoryPolicyTool()),
        SourcingAgent(POPlacementTool(), ExpediteTool()),
        ProductionAgent(ProductionTool(), capacity_units=int(asm_cap)),
        DeliveryAgent(ShippingTool()),
        CriticAgent(target_otif=0.95)
    ]
    orch = Orchestrator(env, bb, agents)

    # Sim loop
    daily_traces = []
    for t in range(1, days+1):
        trace = orch.step(t)
        daily_traces.append(trace)

    # ====== Output ======
    log = pd.DataFrame(bb.trace)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    latest = log.dropna(subset=["OTIF"]).iloc[-1] if len(log.dropna(subset=["OTIF"]))>0 else None
    with c1: st.metric("OTIF (latest)", f"{(latest['OTIF']*100):.1f}%" if latest is not None else "n/a")
    with c2: st.metric("Inventory Units", f"{int(log['InvUnits'].iloc[-1])}")
    with c3: st.metric("Revenue (cum)", f"${log['Revenue'].iloc[-1]:,.0f}")
    with c4: st.metric("COGS (cum)", f"${log['COGS'].iloc[-1]:,.0f}")

    # Charts (matplotlib, 1 plot per chart, no styles)
    fig1, ax1 = plt.subplots(); ax1.plot(log["day"], log["OTIF"]); ax1.set_title("Daily OTIF"); ax1.set_xlabel("Day"); ax1.set_ylabel("OTIF"); st.pyplot(fig1)
    fig2, ax2 = plt.subplots(); ax2.plot(log["day"], log["InvUnits"]); ax2.set_title("Inventory Units"); ax2.set_xlabel("Day"); ax2.set_ylabel("Units"); st.pyplot(fig2)
    fig3, ax3 = plt.subplots(); ax3.plot(log["day"], log["Revenue"], label="Revenue"); ax3.plot(log["day"], log["COGS"], label="COGS")
    ax3.set_title("Revenue vs COGS (Cumulative)"); ax3.set_xlabel("Day"); ax3.set_ylabel("USD"); ax3.legend(); st.pyplot(fig3)

    # Agent Trace (explicit perceive→plan→act→reflect)
    st.subheader("Agent Trace (Perceive → Plan → Act → Reflect)")
    for d in daily_traces:
        with st.expander(f"Day {d['day']}"):
            for (agent, sensed, plan, acted, refl) in d["logs"]:
                st.markdown(f"**{agent}**")
                st.write("• Perceive:", sensed)
                st.write("• Plan:", plan)
                st.write("• Act:", acted)
                st.write("• Reflect:", refl)

    # Blackboard snapshot
    st.subheader("Blackboard (Shared Memory) — Final")
    st.json({"policy": bb.policy, "backlog_ratio": bb.backlog_ratio,
             "last_forecast": bb.forecast, "last_prod_plan": bb.prod_plan})

    # Tables
    st.subheader("Outstanding Inbound POs")
    inbound_df = pd.DataFrame([po.__dict__ for po in env.inbound]) if env.inbound else pd.DataFrame(columns=["po_id","item_id","qty","eta_day","mode","expedited"])
    st.dataframe(inbound_df)

    st.subheader("Orders (sample)")
    orders_df = pd.DataFrame([o.__dict__ for o in env.orders]).head(50)
    st.dataframe(orders_df)

    # Downloads
    t1, t2 = st.tabs(["Download KPI Log", "Download Orders"])
    with t1:
        st.download_button("KPI Log (CSV)", data=log.to_csv(index=False).encode("utf-8"), file_name="agentic_kpis.csv", mime="text/csv")
    with t2:
        st.download_button("Orders (CSV)", data=orders_df.to_csv(index=False).encode("utf-8"), file_name="orders_sample.csv", mime="text/csv")

else:
    st.info("Set parameters in the sidebar, then click **Run Agentic Simulation**. This demo exposes agent loops & policy adaptation (reflection) explicitly.")
