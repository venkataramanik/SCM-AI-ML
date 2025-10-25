# app.py
# ─────────────────────────────────────────────────────────────────────────────
# TMS Step-By-Step Simulator (Streamlit)
# What it does end-to-end:
# 1) Order Capture  → create shipments (lanes, weights, due times)
# 2) Rating & Mode  → compute Parcel/LTL/TL costs & pick feasible mode
# 3) Load Planning  → convert rated orders to loads (direct; TL consolidation optional)
# 4) Tendering      → send to carriers; simulate accept/decline
# 5) Events         → generate PU/DEP/ARR/EXC/DEL visibility events
# 6) Invoice & Pay  → create invoices and 3-way match (approve/short-pay)
#
# Viewer controls:
# • “Next step” advances through the pipeline and shows results for that step
# • “Run all” executes the full pipeline and shows every section
# • Download buttons export CSV snapshots at each step
#
# No external data or APIs; pure Python + Streamlit + pandas.

from __future__ import annotations
import math, random, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & State
# ─────────────────────────────────────────────────────────────────────────────

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def df(data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data)

def init_state():
    ss = st.session_state
    if "step" not in ss:
        ss.step = 0
    keys = [
        "orders", "rated", "loads", "tenders", "events",
        "invoices", "pay", "carriers", "tariff", "now"
    ]
    for k in keys:
        if k not in ss:
            ss[k] = None

# ─────────────────────────────────────────────────────────────────────────────
# Domain Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Location:
    code: str
    name: str
    lat: float
    lon: float

GEO = {
    "ATL": Location("ATL", "Atlanta, GA", 33.7490, -84.3880),
    "LAX": Location("LAX", "Los Angeles, CA", 33.9416, -118.4085),
    "DFW": Location("DFW", "Dallas, TX", 32.8998, -97.0403),
    "ORD": Location("ORD", "Chicago, IL", 41.9742, -87.9073),
    "JFK": Location("JFK", "New York, NY", 40.6413, -73.7781),
    "MIA": Location("MIA", "Miami, FL", 25.7959, -80.2870),
    "SEA": Location("SEA", "Seattle, WA", 47.4502, -122.3088),
    "DEN": Location("DEN", "Denver, CO", 39.8561, -104.6737),
    "MEM": Location("MEM", "Memphis, TN", 35.1495, -90.0490),
}

def haversine_miles(a: Location, b: Location) -> float:
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [a.lat, a.lon, b.lat, b.lon])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

@dataclass
class Order:
    order_id: str
    origin: str
    destination: str
    weight_lb: float
    ready_time: datetime
    due_time: datetime
    service: str = "STANDARD"

@dataclass
class Carrier:
    carrier_id: str
    name: str
    modes: List[str]              # ["PARCEL","LTL","TL"]
    appetite: float               # 0..1
    cost_factor: float            # internal cost vs tariff (~0.85-1.1)

@dataclass
class RateCard:
    parcel_rate_per_lb: float = 1.1
    parcel_min: float = 12.0
    ltl_rate_per_cwt: float = 32.0
    ltl_min: float = 95.0
    tl_rate_per_mile: float = 2.10
    tl_stop_charge: float = 50.0
    tl_min: float = 600.0
    fuel_surcharge_pct: float = 0.22

@dataclass
class Load:
    load_id: str
    mode: str
    stops: List[str]              # [origin, destination] (multi-stop TL optional extension)
    distance_mi: float
    linehaul_est: float
    fsc_est: float
    accessorials_est: float
    total_est: float
    tender_status: str = "NOT_SENT"  # SENT, ACCEPTED, DECLINED, EXPIRED
    carrier_id: Optional[str] = None
    carrier_name: Optional[str] = None

@dataclass
class Tender:
    tender_id: str
    load_id: str
    carrier_id: str
    status: str                   # SENT/ACCEPTED/DECLINED/EXPIRED

@dataclass
class ShipmentEvent:
    event_id: str
    load_id: str
    timestamp: datetime
    code: str                     # PU/DEP/ARR/EXC/DEL
    detail: str

@dataclass
class Invoice:
    invoice_id: str
    load_id: str
    total: float

@dataclass
class PayDecision:
    load_id: str
    expected: float
    invoiced: float
    status: str                   # APPROVED/SHORT_PAY
    note: str

# ─────────────────────────────────────────────────────────────────────────────
# Business Logic
# ─────────────────────────────────────────────────────────────────────────────

def make_orders(now: datetime, lanes: list[tuple[str,str]], seed: int) -> list[Order]:
    random.seed(seed)
    out: list[Order] = []
    for o, d in lanes:
        w = random.choice([30, 60, 1200, 4000, 9000, 35000])  # parcel→LTL→TL candidates
        ready = now + timedelta(hours=random.randint(0, 6))
        due = ready + timedelta(days=random.choice([1, 2, 3]))
        out.append(Order(gen_id("ORD"), o, d, w, ready, due))
    return out

def choose_mode(weight_lb: float) -> str:
    if weight_lb <= 70: return "PARCEL"
    if weight_lb <= 10000: return "LTL"
    return "TL"

def rate_order(o: Order, rate: RateCard) -> tuple[str, float, float, float, float, float]:
    dist = haversine_miles(GEO[o.origin], GEO[o.destination])
    mode = choose_mode(o.weight_lb)
    if mode == "PARCEL":
        base = max(rate.parcel_min, o.weight_lb * rate.parcel_rate_per_lb)
        acc = 0.0
    elif mode == "LTL":
        cwt = o.weight_lb / 100.0
        base = max(rate.ltl_min, cwt * rate.ltl_rate_per_cwt * (1 + dist/2000.0))
        acc = 25.0 if o.weight_lb > 5000 else 0.0
    else:  # TL
        base = max(rate.tl_min, dist * rate.tl_rate_per_mile) + rate.tl_stop_charge
        acc = 0.0
    fsc = base * rate.fuel_surcharge_pct
    total = base + fsc + acc
    return mode, dist, base, fsc, acc, total

def plan_loads(orders: list[Order], rate: RateCard) -> list[Load]:
    loads: list[Load] = []
    for o in orders:
        mode, dist, base, fsc, acc, total = rate_order(o, rate)
        loads.append(
            Load(
                load_id=gen_id("LOAD"),
                mode=mode,
                stops=[o.origin, o.destination],
                distance_mi=round(dist, 1),
                linehaul_est=round(base, 2),
                fsc_est=round(fsc, 2),
                accessorials_est=round(acc, 2),
                total_est=round(total, 2),
            )
        )
    return loads

def make_carriers() -> list[Carrier]:
    return [
        Carrier("C1","SwiftParcel",["PARCEL"],0.85,0.95),
        Carrier("C2","BlueFreight",["LTL"],0.70,0.90),
        Carrier("C3","RoadRunner",["TL"],0.55,0.92),
        Carrier("C4","OmniCarrier",["PARCEL","LTL","TL"],0.65,0.97),
    ]

def tender_loads(loads: list[Load], carriers: list[Carrier]) -> list[Tender]:
    tenders: list[Tender] = []
    for ld in loads:
        elig = [c for c in carriers if ld.mode in c.modes]
        elig.sort(key=lambda c: (-c.appetite, c.cost_factor))  # most likely first
        accepted = False
        for c in elig:
            internal_cost = ld.total_est * c.cost_factor
            margin = (ld.total_est - internal_cost) / max(1.0, internal_cost)
            p = max(0.05, min(0.95, c.appetite * (0.4 + 0.6*(margin + 0.5))))
            if random.random() < p:
                ld.tender_status = "ACCEPTED"
                ld.carrier_id = c.carrier_id
                ld.carrier_name = c.name
                tenders.append(Tender(gen_id("TDR"), ld.load_id, c.carrier_id, "ACCEPTED"))
                accepted = True
                break
            else:
                tenders.append(Tender(gen_id("TDR"), ld.load_id, c.carrier_id, "DECLINED"))
        if not accepted:
            ld.tender_status = "EXPIRED"
    return tenders

def simulate_events(loads: list[Load], start: datetime) -> list[ShipmentEvent]:
    evs: list[ShipmentEvent] = []
    for ld in loads:
        if ld.tender_status != "ACCEPTED":
            continue
        t = start + timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "PU", f"Picked up by {ld.carrier_name}"))
        t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "DEP", f"Depart {ld.stops[0]}"))
        t += timedelta(hours=(ld.distance_mi / 45.0) + 1.0)
        if random.random() < 0.1:
            evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "EXC", "Traffic congestion"))
            t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "ARR", f"Arrive {ld.stops[-1]}"))
        t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "DEL", "Delivered, POD signed"))
    return evs

def generate_invoices(loads: list[Load], rate: RateCard) -> list[Invoice]:
    invs: list[Invoice] = []
    for ld in loads:
        if ld.tender_status != "ACCEPTED":
            continue
        linehaul = ld.linehaul_est * random.uniform(0.98, 1.02)
        fsc = linehaul * rate.fuel_surcharge_pct
        acc = ld.accessorials_est
        total = round(linehaul + fsc + acc, 2)
        invs.append(Invoice(gen_id("INV"), ld.load_id, total))
    return invs

def match_pay(loads: list[Load], invoices: list[Invoice], tol_pct=0.02, tol_abs=10.0) -> list[PayDecision]:
    exp = {ld.load_id: ld.total_est for ld in loads}
    decisions: list[PayDecision] = []
    for inv in invoices:
        expected = round(exp.get(inv.load_id, 0.0), 2)
        diff = inv.total - expected
        within = abs(diff) <= max(tol_abs, expected * tol_pct)
        if within:
            decisions.append(PayDecision(inv.load_id, expected, inv.total, "APPROVED", "Within tolerance"))
        else:
            note = "Over tolerance; short-pay excess" if diff > 0 else "Underrun; approve"
            status = "SHORT_PAY" if diff > 0 else "APPROVED"
            decisions.append(PayDecision(inv.load_id, expected, inv.total, status, note))
    return decisions

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

init_state()
st.set_page_config(page_title="TMS Step-By-Step Simulator", page_icon="🚚", layout="wide")
st.title("🚚 TMS Step-By-Step Simulator")
st.caption("Orders → Rating & Mode → Load Planning → Tendering → Events → Invoice & 3-Way Match")

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=11, step=1)
    lanes_default = "ATL:JFK, ATL:ORD, DFW:LAX, ORD:DEN, DEN:SEA, JFK:MIA"
    lanes_str = st.text_input("Lanes (comma-separated origin:dest)", value=lanes_default)
    tol_pct = st.slider("Pay tolerance (%)", 0.0, 10.0, 2.0, 0.5)
    tol_abs = st.slider("Pay tolerance ($)", 0.0, 100.0, 10.0, 1.0)
    st.markdown("---")
    colsb = st.columns(3)
    with colsb[0]:
        if st.button("Reset", type="secondary"):
            for k in list(st.session_state.keys()):
                if k not in ("seed_persist",):  # keep nothing except special flags
                    del st.session_state[k]
            init_state()
            st.experimental_rerun()
    with colsb[1]:
        if st.button("Next step ▶️", type="primary"):
            st.session_state.step = min(st.session_state.step + 1, 6)
    with colsb[2]:
        if st.button("Run all ✅", type="primary"):
            st.session_state.step = 6

# Parse lanes
lanes = []
for token in [s.strip() for s in lanes_str.split(",") if s.strip()]:
    if ":" in token:
        o, d = token.split(":", 1)
        o, d = o.strip().upper(), d.strip().upper()
        if o in GEO and d in GEO:
            lanes.append((o, d))
if not lanes:
    st.error("No valid lanes parsed. Use format like: ATL:JFK, DFW:LAX")
    st.stop()

# Initialize common objects
if st.session_state.now is None:
    st.session_state.now = datetime.now()
if st.session_state.tariff is None:
    st.session_state.tariff = RateCard()
if st.session_state.carriers is None:
    st.session_state.carriers = make_carriers()

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Overview blurb
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🔎 What this simulation does (end-to-end) — click to expand", expanded=True):
    st.markdown("""
**In one screen you can watch freight flow digitally through a TMS:**

**1) Order Capture** — Create shipments (origin/destination, weight, due date).  
**2) Rating & Mode** — Price Parcel/LTL/TL and choose a feasible mode.  
**3) Load Planning** — Turn rated orders into executable loads.  
**4) Tendering** — Offer loads to carriers; simulate accept/decline.  
**5) Events** — Generate PU/DEP/ARR/EXC/DEL visibility milestones.  
**6) Invoice & 3-Way Match** — Compare invoice to plan; approve or short-pay.

Use **Next step** to progress, or **Run all** to execute the full pipeline.
""")

# Utility to render and download a dataframe
def show_table(title: str, data: list[dict], use_index=False):
    st.subheader(title)
    df_obj = df(data)
    st.dataframe(df_obj, use_container_width=True, hide_index=not use_index)
    st.download_button(
        label=f"⬇️ Download {title} (CSV)",
        data=df_obj.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Orders
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 1:
    with st.container(border=True):
        st.markdown("## 1) **Order Capture**")
        st.write("We create shipment orders with lanes, weights, and due dates. Good order data drives mode, cost, and on-time delivery.")
        if st.session_state.orders is None:
            st.session_state.orders = make_orders(st.session_state.now, lanes, seed)
        orders_out = [
            {
                "order_id": o.order_id,
                "lane": f"{o.origin} → {o.destination}",
                "weight_lb": o.weight_lb,
                "ready_time": o.ready_time.strftime("%Y-%m-%d %H:%M"),
                "due_time": o.due_time.strftime("%Y-%m-%d %H:%M"),
                "service": o.service,
            }
            for o in st.session_state.orders
        ]
        show_table("Orders", orders_out)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Rating & Mode
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 2:
    with st.container(border=True):
        st.markdown("## 2) **Rating & Mode Selection (Parcel / LTL / TL)**")
        st.write("We compute estimated charges and choose a feasible mode: small → Parcel, mid-weight → LTL, heavy → TL.")
        if st.session_state.rated is None:
            rate = st.session_state.tariff
            rated = []
            for o in st.session_state.orders:
                mode, dist, base, fsc, acc, total = rate_order(o, rate)
                rated.append({
                    "order_id": o.order_id,
                    "mode": mode,
                    "miles": round(dist, 1),
                    "linehaul_est": round(base, 2),
                    "fsc_est": round(fsc, 2),
                    "accessorials_est": round(acc, 2),
                    "est_total": round(total, 2),
                })
            st.session_state.rated = rated
        show_table("Rated Orders", st.session_state.rated)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load Planning
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 3:
    with st.container(border=True):
        st.markdown("## 3) **Load Planning**")
        st.write("We turn rated orders into executable loads. Parcel/LTL are direct. (TL consolidation can be added as an extension.)")
        if st.session_state.loads is None:
            st.session_state.loads = plan_loads(st.session_state.orders, st.session_state.tariff)
        loads_out = [
            {
                "load_id": ld.load_id,
                "mode": ld.mode,
                "stops": " → ".join(ld.stops),
                "miles": ld.distance_mi,
                "linehaul_est": ld.linehaul_est,
                "fsc_est": ld.fsc_est,
                "accessorials_est": ld.accessorials_est,
                "est_total": ld.total_est,
                "tender_status": ld.tender_status
            } for ld in st.session_state.loads
        ]
        show_table("Planned Loads", loads_out)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Tendering
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 4:
    with st.container(border=True):
        st.markdown("## 4) **Tendering (Send → Accept/Decline)**")
        st.write("We offer each load to eligible carriers. Acceptance reflects appetite and margin. If accepted, we assign the carrier.")
        if st.session_state.tenders is None:
            st.session_state.tenders = tender_loads(st.session_state.loads, st.session_state.carriers)
        loads_now = [
            {
                "load_id": ld.load_id,
                "mode": ld.mode,
                "tender_status": ld.tender_status,
                "carrier": ld.carrier_name or "-",
                "est_total": ld.total_est,
            } for ld in st.session_state.loads
        ]
        show_table("Tender Outcomes", loads_now)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Events
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 5:
    with st.container(border=True):
        st.markdown("## 5) **Shipment Events (Visibility)**")
        st.write("For accepted loads, we generate pickup (PU), depart/arrive (DEP/ARR), occasional exceptions (EXC), and delivery (DEL).")
        if st.session_state.events is None:
            st.session_state.events = simulate_events(st.session_state.loads, st.session_state.now)
        events_out = [
            {
                "load_id": e.load_id,
                "code": e.code,
                "detail": e.detail,
                "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M")
            } for e in st.session_state.events
        ]
        show_table("Event Stream", events_out)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Invoice & Pay
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step >= 6:
    with st.container(border=True):
        st.markdown("## 6) **Invoice & Freight Pay (3-Way Match)**")
        st.write(f"Carriers submit invoices. We compare to planned cost. If within ±{tol_pct:.1f}% or ${tol_abs:.0f} tolerance → APPROVED; else SHORT_PAY (or approve if underrun).")
        if st.session_state.invoices is None:
            st.session_state.invoices = generate_invoices(st.session_state.loads, st.session_state.tariff)
        if st.session_state.pay is None:
            st.session_state.pay = match_pay(st.session_state.loads, st.session_state.invoices, tol_pct/100.0, tol_abs)
        invoices_out = [{"invoice_id": inv.invoice_id, "load_id": inv.load_id, "total": inv.total} for inv in st.session_state.invoices]
        show_table("Carrier Invoices", invoices_out)
        pay_out = [
            {"load_id": p.load_id, "expected": p.expected, "invoiced": p.invoiced, "decision": p.status, "note": p.note}
            for p in st.session_state.pay
        ]
        show_table("Pay Decisions", pay_out)

# Footer
st.markdown("---")
st.caption("© TMS step-by-step simulator • Pure Python demo for education & showcases.")
