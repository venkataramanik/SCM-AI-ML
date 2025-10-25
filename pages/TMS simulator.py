# app.py (or pages/TMS simulator.py)
# TMS Step-by-Step Simulator — Next button fixed (always visible on page)
import math, random, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="TMS Step-by-Step Simulator", layout="wide")

# ---------------- Utilities ------------------
def gen_id(prefix):
    return "%s-%s" % (prefix, uuid.uuid4().hex[:8])

def df(rows):
    return pd.DataFrame(rows)

def normalize_code(s):
    return (s or "").strip().upper()

# ---------------- Domain ---------------------
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
    "MIA": Location("MIA","Miami, FL",25.80,-80.29),
    "MEM": Location("MEM","Memphis, TN",35.15,-90.05),
}

def miles(a, b):
    a = normalize_code(a); b = normalize_code(b)
    if a not in GEO or b not in GEO:
        raise ValueError("Unknown code(s): origin=%r valid=%r, dest=%r valid=%r" %
                         (a, a in GEO, b, b in GEO))
    A, B = GEO[a], GEO[b]
    R = 3958.8
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

# ---------------- Session State --------------
DEFAULT_KEYS = [
    "orders","loads","events","invoices","pay","tenders",
    "step","lanes_valid","lanes_rejected","seed"
]
for k in DEFAULT_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.step is None:
    st.session_state.step = 0
if st.session_state.seed is None:
    st.session_state.seed = 11

# ---------------- Sidebar: lanes & options ---
st.sidebar.header("Inputs")
lanes_default = "ATL:JFK, ATL:ORD, DFW:LAX, ORD:DEN, DEN:SEA"
lanes_str = st.sidebar.text_input("Lanes (origin:dest, comma separated)", value=lanes_default)
st.sidebar.caption("Known codes: " + ", ".join(sorted(GEO.keys())))

def parse_lanes(text):
    valid, rejected = [], []
    tokens = [t.strip() for t in (text or "").split(",") if t.strip()]
    for tok in tokens:
        if ":" not in tok: rejected.append(tok); continue
        o, d = tok.split(":", 1)
        o, d = normalize_code(o), normalize_code(d)
        if o in GEO and d in GEO: valid.append((o, d))
        else: rejected.append(tok)
    return valid, rejected

lanes_valid, lanes_rejected = parse_lanes(lanes_str)
st.session_state.lanes_valid = lanes_valid
st.session_state.lanes_rejected = lanes_rejected

if lanes_rejected:
    st.sidebar.warning("Ignored invalid lanes: %s" % ", ".join(lanes_rejected))

# ---------------- Business Logic ------------
def choose_mode(weight_lb):
    if weight_lb <= 70: return "PARCEL"
    if weight_lb <= 10000: return "LTL"
    return "TL"

def make_orders(lanes, seed):
    random.seed(seed)
    now = datetime.now()
    out = []
    for o, d in lanes:
        w = random.choice([60, 2000, 8000, 35000])  # parcel→LTL→TL candidates
        out.append(Order(gen_id("ORD"), o, d, w, now + timedelta(days=random.randint(1,3))))
    return out

def rate_order(o):
    try:
        dist = miles(o.origin, o.destination)
    except ValueError:
        return Load(gen_id("LOAD"), "INVALID", o.origin, o.destination, 0.0, 0.0, "INVALID", None)
    mode = choose_mode(o.weight_lb)
    if mode == "PARCEL":
        base = max(12.0, o.weight_lb * 1.1)
    elif mode == "LTL":
        base = max(95.0, (o.weight_lb/100.0) * 32.0 * (1.0 + dist/2000.0))
    else:
        base = max(600.0, dist * 2.10) + 50.0
    total = round(base * 1.22, 2)  # fuel surcharge
    return Load(gen_id("LOAD"), mode, o.origin, o.destination, round(dist, 0), total)

def tender(loads):
    carriers = [
        Carrier("SwiftParcel", ["PARCEL"], 0.80, 0.95),
        Carrier("BlueFreight", ["LTL"],    0.70, 0.90),
        Carrier("RoadRunner",  ["TL"],     0.60, 0.92)
    ]
    for ld in loads:
        if ld.mode not in ("PARCEL","LTL","TL"):
            ld.tender = "INVALID"; continue
        options = [c for c in carriers if ld.mode in c.modes]
        options.sort(key=lambda c: (-c.appetite, c.cost_factor, c.name))
        accepted = False
        for c in options:
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
        if ld.tender != "ACCEPTED": continue
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

# ---------------- Header & Top Controls -----
st.title("TMS Step-by-Step Simulator")

st.write(
    "Watch freight flow: "
    "1) Order Capture → 2) Rating & Mode → 3) Load Planning → 4) Tendering → 5) Events → 6) Invoice & 3-Way Match."
)

# Always-visible main-page controls (NOT in sidebar)
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Next Step →", key="next_main"):
        st.session_state.step = min((st.session_state.step or 0) + 1, 6)
with c2:
    if st.button("Run All", key="runall_main"):
        st.session_state.step = 6
with c3:
    if st.button("Reset", key="reset_main"):
        for k in DEFAULT_KEYS:
            st.session_state[k] = None
        st.session_state.step = 0
        st.session_state.seed = 11
        st.experimental_rerun()

# ---------------- Guard rails --------------
if not st.session_state.lanes_valid:
    st.error("No valid lanes. Add at least one valid origin:dest pair in the sidebar.")
    st.stop()

# ---------------- Steps ---------------------
step = st.session_state.step or 0

# Step 1
if step >= 1:
    st.header("1) Order Capture")
    st.info("We create orders from your valid lanes input. Invalid lanes are ignored and won't break the app.")
    if st.session_state.orders is None:
        st.session_state.orders = make_orders(st.session_state.lanes_valid, st.session_state.seed)
    st.dataframe(df([asdict(o) for o in st.session_state.orders]), use_container_width=True)

# Step 2
if step >= 2:
    st.header("2) Rating & Mode Selection")
    st.info("Each order is priced by a feasible mode (Parcel/LTL/TL) using distance and weight. INVALID loads indicate bad lanes.")
    if st.session_state.loads is None:
        st.session_state.loads = [rate_order(o) for o in st.session_state.orders]
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 3
if step >= 3:
    st.header("3) Load Planning")
    st.info("Orders are now loads with estimated cost and distance (direct moves for this demo).")
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 4
if step >= 4:
    st.header("4) Tendering")
    st.info("Loads are offered to eligible carriers; acceptance depends on appetite. INVALID loads are skipped.")
    if st.session_state.tenders is None:
        st.session_state.loads = tender(st.session_state.loads)
        st.session_state.tenders = True
    st.dataframe(df([asdict(l) for l in st.session_state.loads]), use_container_width=True)

# Step 5
if step >= 5:
    st.header("5) Shipment Events")
    st.info("For accepted loads, we simulate pickup (PU) and delivery (DEL).")
    if st.session_state.events is None:
        st.session_state.events = make_events(st.session_state.loads)
    st.dataframe(df([asdict(e) for e in st.session_state.events]), use_container_width=True)

# Step 6
if step >= 6:
    st.header("6) Invoice & Freight Pay")
    st.info("We compare invoices to planned cost and approve or short-pay within tolerances.")
    if st.session_state.invoices is None:
        st.session_state.invoices = make_invoices(st.session_state.loads)
        st.session_state.pay = paymatch(st.session_state.loads, st.session_state.invoices)
    st.subheader("Pay Decisions")
    st.dataframe(df([asdict(p) for p in st.session_state.pay]), use_container_width=True)

st.markdown("---")
st.caption("TMS Simulation • Streamlit app • Always-visible Next button")
