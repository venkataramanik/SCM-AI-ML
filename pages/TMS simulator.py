# app.py
# TMS Step-by-Step Simulator (Cloud-safe, no emojis)
# Orders → Rating → Load Planning → Tendering → Events → Invoice & Pay

import math, random, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="TMS Step-by-Step Simulator", layout="wide")

# ---------- Utility ----------
def gen_id(prefix):
    return "%s-%s" % (prefix, uuid.uuid4().hex[:8])

def df(rows):
    return pd.DataFrame(rows)

# ---------- Session state ----------
DEFAULT_KEYS = ["orders","loads","events","invoices","pay","tenders","carriers","step"]
for k in DEFAULT_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.step is None:
    st.session_state.step = 0

# ---------- Domain ----------
@dataclass
class Location:
    code: str
    name: str
    lat: float
    lon: float

GEO = {
    "ATL": Location("ATL","Atlanta, GA",33.75,-84.39),
    "JFK": Location("JFK","New York, NY",40.64,-73.78),
    "ORD": Location("ORD","Chicago, IL",41.97,-87.91),
    "DFW": Location("DFW","Dallas, TX",32.90,-97.04),
    "LAX": Location("LAX","Los Angeles, CA",33.94,-118.41),
    "DEN": Location("DEN","Denver, CO",39.86,-104.67),
    "SEA": Location("SEA","Seattle, WA",47.45,-122.31),
}

def miles(a, b):
    R = 3958.8
    A, B = GEO[a], GEO[b]
    lat1, lon1, lat2, lon2 = map(math.radians, [A.lat, A.lon, B.lat, B.lon])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2.0)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(h))

@dataclass
class Order:
    order_id: str
    origin: str
    destination: str
    weight_lb: float
    due: datetime

@dataclass
class Load:
    load_id: str
    mode: str
    origin: str
    destination: str
    miles: float
    total_est: float
    tender: str = "NOT_SENT"
    carrier: Optional[str] = None

@dataclass
class Carrier:
    name: str
    modes: List[str]
    appetite: float
    cost_factor: float

@dataclass
class Event:
    load_id: str
    code: str
    detail: str
    time: str

@dataclass
class Invoice:
    load_id: str
    total: float

@dataclass
class PayDecision:
    load_id: str
    expected: float
    invoiced: float
    decision: str
    note: str

# ---------- Business logic ----------
def make_orders():
    lanes = [("ATL","JFK"),("ATL","ORD"),("DFW","LAX"),("ORD","DEN"),("DEN","SEA")]
    now = datetime.now()
    out = []
    for o, d in lanes:
        w = random.choice([60, 2000, 8000, 35000])  # parcel → LTL → TL candidates
        out.append(Order(gen_id("ORD"), o, d, w, now + timedelta(days=random.randint(1,3))))
    return out

def choose_mode(weight_lb):
    if weight_lb <= 70: return "PARCEL"
    if weight_lb <= 10000: return "LTL"
    return "TL"

def rate_order(o):
    dist = miles(o.origin, o.destination)
    mode = choose_mode(o.weight_lb)
    if mode == "PARCEL":
        base = max(12.0, o.weight_lb * 1.1)
    elif mode == "LTL":
        base = max(95.0, (o.weight_lb/100.0) * 32.0 * (1.0 + dist/2000.0))
    else:
        base = max(600.0, dist * 2.10) + 50.0  # TL stop charge (simple)
    total = round(base * 1.22, 2)  # add fuel surcharge
    return Load(gen_id("LOAD"), mode, o.origin, o.destination, round(dist, 0), total)

def tender(loads):
    carriers = [
        Carrier("SwiftParcel", ["PARCEL"], 0.80, 0.95),
        Carrier("BlueFreight", ["LTL"],    0.70, 0.90),
        Carrier("RoadRunner",  ["TL"],     0.60, 0.92)
    ]
    for ld in loads:
        options = [c for c in carriers if ld.mode in c.modes]
        # deterministic-ish order for stability
        options.sort(key=lambda c: (-c.appetite, c.cost_factor, c.name))
        accepted = False
        for c in options:
            # simple acceptance based on appetite
            if random.random() < c.appetite:
                ld.tender = "ACCEPTED"
                ld.carrier = c.name
                accepted = True
                break
        if not accepted:
            ld.tender = "EXPIRED"
    return loads

def make_events(loads):
    out = []
    for ld in loads:
        if ld.tender != "ACCEPTED":
            continue
        t0 = datetime.now()
        out.append(Event(ld.load_id, "PU",  "Picked up by %s" % ld.carrier, t0.strftime("%Y-%m-%d %H:%M")))
        out.append(Event(ld.load_id, "DEL", "Delivered", (t0 + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")))
    return out

def make_invoices(loads):
    out = []
    for ld in loads:
        if ld.tender == "ACCEPTED":
            amt = round(ld.total_est * random.uniform(0.98, 1.03), 2)
            out.append(Invoice(ld.load_id, amt))
    return out

def paymatch(loads, invoices, tol_pct=0.02, tol_abs=10.0):
    exp = {ld.load_id: ld.total_est for ld in loads}
    out = []
    for inv in invoices:
        expected = round(exp.get(inv.load_id, 0.0), 2)
        diff = inv.total - expected
        within = abs(diff) <= max(tol_abs, expected * tol_pct)
        if within:
            out.append(PayDecision(inv.load_id, expected, inv.total, "APPROVED", "Within tolerance"))
        else:
            if diff > 0:
                out.append(PayDecision(inv.load_id, expected, inv.total, "SHORT_PAY", "Over tolerance; short-pay excess"))
            else:
                out.append(PayDecision(inv.load_id, expected, inv.total, "APPROVED", "Underrun; approve"))
    return out

# ---------- UI ----------
st.title("TMS Step-by-Step Simulator (No Icons)")
st.write(
    "Watch freight flow digitally through a TMS:\n"
    "1) Order Capture → 2) Rating & Mode → 3) Load Planning → 4) Tendering → 5) Events → 6) Invoice & 3-Way Match.\n"
    "Use the Next Step button to proceed."
)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("Next Step →"):
        st.session_state.step = min(st.session_state.step + 1, 6)
with c2:
    if st.button("Reset"):
        for k in DEFAULT_KEYS:
            st.session_state[k] = None
        st.session_state.step = 0
        st.experimental_rerun()

step = st.session_state.step

# Step 1: Orders
if step >= 1:
    st.header("1) Order Capture")
    st.info("Orders represent customer shipment requests with lanes, weights, and due dates.")
    if st.session_state.orders is None:
        random.seed(11)
        st.session_state.orders = make_orders()
    st.dataframe(df([asdict(o) for o in st.session_state.orders]), use_container_width=True)

# Step 2: Rating
if step >= 2:
    st.header("2) Rating & Mode Selection")
    st.info("Each order is priced by a feasible mode (Parcel/LTL/TL) using distance and weight.")
    if st.session_state.loads is None:
        st.session_state.loads = [rate_order(o) for o in st.session_state.orders]
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 3: Load Planning
if step >= 3:
    st.header("3) Load Planning")
    st.info("Orders become loads with an estimated cost and distance (direct moves for Parcel/LTL).")
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 4: Tendering
if step >= 4:
    st.header("4) Tendering")
    st.info("Loads are offered to eligible carriers; each may accept or decline.")
    if st.session_state.tenders is None:
        st.session_state.loads = tender(st.session_state.loads)
        st.session_state.tenders = True
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 5: Events
if step >= 5:
    st.header("5) Shipment Events")
    st.info("We simulate pickup (PU) and delivery (DEL) for accepted loads.")
    if st.session_state.events is None:
        st.session_state.events = make_events(st.session_state.loads)
    st.dataframe(df([asdict(e) for e in st.session_state.events]), use_container_width=True)

# Step 6: Invoice & Pay
if step >= 6:
    st.header("6) Invoice & Freight Pay")
    st.info("Carriers send invoices; we match to plan and approve or short-pay using tolerance rules.")
    if st.session_state.invoices is None:
        st.session_state.invoices = make_invoices(st.session_state.loads)
        st.session_state.pay = paymatch(st.session_state.loads, st.session_state.invoices)
    st.subheader("Invoices vs Expected (Decision)")
    st.dataframe(df([asdict(p) for p in st.session_state.pay]), use_container_width=True)

st.markdown("---")
st.caption("TMS Simulation • Streamlit app • Plain Python • No icons/emojis")

