# app.py
# TMS Step-by-Step Simulator — TL consolidation (multi-stop), TL sequential tendering,
# Parcel/LTL auto-accept, robust reset, no emojis/icons.

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
    service: str = "STANDARD"

@dataclass
class Load:
    load_id: str
    mode: str                       # "PARCEL","LTL","TL"
    origin: str
    stops: List[str]                # sequence of stops (first is origin)
    miles: float
    linehaul_est: float
    fsc_est: float
    accessorials_est: float
    total_est: float
    tender: str = "NOT_SENT"        # for TL: NOT_SENT/SENT/ACCEPTED/EXPIRED ; for Parcel/LTL: ACCEPTED
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
    "orders","rated","loads","events","invoices","pay",
    "tenders","step","lanes_valid","lanes_rejected","seed"
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
lanes_default = "ATL:JFK, ATL:ORD, DFW:LAX, ORD:DEN, DEN:SEA, ATL:MIA, ATL:SEA"
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
    # create multiple orders per lane to enable TL consolidation
    for o, d in lanes:
        # 2–3 orders per lane
        for _ in range(random.choice([1,2,3])):
            w = random.choice([30, 60, 1200, 4000, 9000, 12000, 16000, 22000, 28000, 35000])
            out.append(Order(gen_id("ORD"), o, d, w, now + timedelta(days=random.randint(1,3))))
    return out

def rate_parcel(weight_lb, dist):
    base = max(12.0, weight_lb * 1.10)
    fsc = base * 0.22
    return base, fsc, 0.0, base + fsc

def rate_ltl(weight_lb, dist):
    base = max(95.0, (weight_lb/100.0) * 32.0 * (1.0 + dist/2000.0))
    fsc = base * 0.22
    acc = 25.0 if weight_lb > 5000 else 0.0
    return base, fsc, acc, base + fsc + acc

def rate_tl(dist, num_stops):
    base = max(600.0, dist * 2.10) + max(0, num_stops - 2) * 50.0
    fsc = base * 0.22
    acc = max(0, num_stops - 2) * 15.0
    return base, fsc, acc, base + fsc + acc

def nn_sequence(start, destinations):
    # nearest-neighbor sequence of destinations from the start
    seq = []
    remaining = destinations[:]
    curr = start
    while remaining:
        nxt = min(remaining, key=lambda x: miles(curr, x))
        seq.append(nxt)
        remaining.remove(nxt)
        curr = nxt
    return seq

def consolidate_tl(orders):
    """
    Consolidate TL-eligible orders by origin with simple capacity (44,000 lb).
    One pickup at origin, multiple drops sequenced via nearest neighbor.
    Returns list of (origin, [destinations], [order_ids], total_weight)
    """
    trailer_cap = 44000.0
    # group TL candidates by origin
    by_origin = {}
    for o in orders:
        by_origin.setdefault(o.origin, []).append(o)

    bundles = []
    for origin, group in by_origin.items():
        # simple bin-pack by weight into batches <= capacity
        remaining = sorted(group, key=lambda x: -x.weight_lb)
        while remaining:
            batch = []
            w_sum = 0.0
            i = 0
            while i < len(remaining):
                if w_sum + remaining[i].weight_lb <= trailer_cap:
                    batch.append(remaining.pop(i))
                    w_sum += batch[-1].weight_lb
                else:
                    i += 1
            # if nothing fit (single overweight), force add first to avoid infinite loop
            if not batch:
                batch.append(remaining.pop(0))
                w_sum = batch[0].weight_lb
            dests = [o.destination for o in batch]
            dest_seq = nn_sequence(origin, list(dict.fromkeys(dests)))  # unique destinations, ordered
            bundles.append((
                origin,
                dest_seq,
                [o.order_id for o in batch],
                w_sum
            ))
    return bundles

def build_loads(orders):
    """
    Builds loads:
      - Parcel/LTL: direct loads per order (ACCEPTED automatically later)
      - TL: consolidate multi-stop per origin with capacity; then rate TL loads
    """
    parcel_ltl = []
    tl_candidates = []

    # classify orders
    for o in orders:
        mode = choose_mode(o.weight_lb)
        if mode in ("PARCEL", "LTL"):
            parcel_ltl.append(o)
        else:
            tl_candidates.append(o)

    # Direct loads for Parcel/LTL
    direct_loads = []
    for o in parcel_ltl:
        try:
            dist = miles(o.origin, o.destination)
        except ValueError:
            dist = 0.0
        if choose_mode(o.weight_lb) == "PARCEL":
            base, fsc, acc, total = rate_parcel(o.weight_lb, dist)
            mode = "PARCEL"
        else:
            base, fsc, acc, total = rate_ltl(o.weight_lb, dist)
            mode = "LTL"
        direct_loads.append(
            Load(
                load_id=gen_id("LOAD"),
                mode=mode,
                origin=o.origin,
                stops=[o.origin, o.destination],
                miles=round(dist, 1),
                linehaul_est=round(base,2),
                fsc_est=round(fsc,2),
                accessorials_est=round(acc,2),
                total_est=round(total,2),
                tender="ACCEPTED",              # business rule: Parcel/LTL auto-accept
                carrier=("SwiftParcel" if mode=="PARCEL" else "BlueFreight")
            )
        )

    # Consolidate TL into multi-stop
    tl_loads = []
    if tl_candidates:
        bundles = consolidate_tl(tl_candidates)
        for origin, dest_seq, order_ids, w_sum in bundles:
            # compute distance over [origin] + dest_seq
            stops = [origin] + dest_seq
            dist = 0.0
            for a, b in zip(stops, stops[1:]):
                dist += miles(a, b)
            base, fsc, acc, total = rate_tl(dist, len(stops))
            tl_loads.append(
                Load(
                    load_id=gen_id("LOAD"),
                    mode="TL",
                    origin=origin,
                    stops=stops,
                    miles=round(dist,1),
                    linehaul_est=round(base,2),
                    fsc_est=round(fsc,2),
                    accessorials_est=round(acc,2),
                    total_est=round(total,2),
                    tender="NOT_SENT"
                )
            )

    return direct_loads + tl_loads

def sequential_tender_tl(loads):
    """
    Tender only TL loads, sequentially, until a carrier accepts or it expires.
    Parcel/LTL loads are already ACCEPTED and skipped here.
    """
    carriers = [
        Carrier("RoadRunner",  ["TL"],     0.60, 0.92),
        Carrier("OmniCarrier", ["PARCEL","LTL","TL"], 0.55, 0.97),
        Carrier("BigRig",      ["TL"],     0.50, 1.00),
    ]
    for ld in loads:
        if ld.mode != "TL":
            continue
        ld.tender = "SENT"
        accepted = False
        # order by a simple preference (more appetite first, then lower cost)
        choices = sorted([c for c in carriers if "TL" in c.modes],
                         key=lambda c: (-c.appetite, c.cost_factor, c.name))
        for c in choices:
            internal_cost = ld.total_est * c.cost_factor
            margin = (ld.total_est - internal_cost) / max(1.0, internal_cost)
            p = max(0.05, min(0.95, c.appetite * (0.4 + 0.6*(margin + 0.5))))
            if random.random() < p:
                ld.tender = "ACCEPTED"
                ld.carrier = c.name
                accepted = True
                break
        if not accepted:
            ld.tender = "EXPIRED"
    return loads

def simulate_events(loads):
    out = []
    for ld in loads:
        if ld.tender != "ACCEPTED":
            continue
        t0 = datetime.now()
        out.append(Event(ld.load_id, "PU",  "Picked up by %s" % ld.carrier, t0.strftime("%Y-%m-%d %H:%M")))
        # walk through each leg for multi-stop TL; for Parcel/LTL just direct
        for a, b in zip(ld.stops, ld.stops[1:]):
            t0 = t0 + timedelta(hours=1 + miles(a,b)/45.0)
            out.append(Event(ld.load_id, "ARR", "Arrived %s" % b, t0.strftime("%Y-%m-%d %H:%M")))
        out.append(Event(ld.load_id, "DEL", "Delivered, POD captured", (t0 + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")))
    return out

def make_invoices(loads):
    out = []
    for ld in loads:
        if ld.tender == "ACCEPTED":
            # keep invoice near planned
            linehaul = ld.linehaul_est * random.uniform(0.98, 1.02)
            fsc = ld.fsc_est * (linehaul / max(1.0, ld.linehaul_est))  # tie FSC to linehaul drift
            acc = ld.accessorials_est
            total = round(linehaul + fsc + acc, 2)
            out.append(Invoice(ld.load_id, total))
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
    "Flow: 1) Order Capture → 2) Rating & Mode → 3) Load Planning (TL consolidation & multi-stop) "
    "→ 4) Tendering (TL only, sequential) → 5) Events → 6) Invoice & 3-Way Match."
)

# Always-visible main-page controls
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Next Step →", key="next_main"):
        st.session_state.step = min((st.session_state.step or 0) + 1, 6)
with c2:
    if st.button("Run All", key="runall_main"):
        st.session_state.step = 6
with c3:
    if st.button("Reset", key="reset_main"):
        # robust reset: clear all known keys and rerun
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Guard rails
if not st.session_state.lanes_valid:
    st.error("No valid lanes. Add at least one valid origin:dest pair in the sidebar.")
    st.stop()

# ---------------- Steps ---------------------
step = st.session_state.step or 0

# Step 1 — Orders
if step >= 1:
    st.header("1) Order Capture")
    st.info("We create multiple orders per lane (varying weights) so TL consolidation is meaningful.")
    if st.session_state.orders is None:
        st.session_state.orders = make_orders(st.session_state.lanes_valid, st.session_state.seed)
    st.dataframe(df([asdict(o) for o in st.session_state.orders]), use_container_width=True)

# Step 2 — Rating & Mode
if step >= 2:
    st.header("2) Rating & Mode Selection")
    st.info("Parcel ≤70 lb, LTL ≤10,000 lb, else TL. We compute mode-driven costs per order.")
    # Show per-order view with chosen mode and a simple direct-distance estimate
    rated_rows = []
    for o in st.session_state.orders:
        try:
            dist = miles(o.origin, o.destination)
        except ValueError:
            dist = 0.0
        m = choose_mode(o.weight_lb)
        if m == "PARCEL":
            base, fsc, acc, tot = rate_parcel(o.weight_lb, dist)
        elif m == "LTL":
            base, fsc, acc, tot = rate_ltl(o.weight_lb, dist)
        else:
            # provisional TL direct (final TL rating happens after consolidation)
            base, fsc, acc, tot = 0.0, 0.0, 0.0, 0.0
        rated_rows.append({
            "order_id": o.order_id, "lane": "%s→%s" % (o.origin, o.destination),
            "weight_lb": o.weight_lb, "mode": m, "miles_est": round(dist,1),
            "est_total_provisional": round(tot,2)
        })
    st.session_state.rated = rated_rows
    st.dataframe(df(rated_rows), use_container_width=True)

# Step 3 — Load Planning (with TL consolidation & multi-stop)
if step >= 3:
    st.header("3) Load Planning (TL consolidation & multi-stop)")
    st.info("Parcel/LTL become direct loads. TL orders are consolidated by origin into multi-stop loads with capacity ~44,000 lb and nearest-neighbor drop sequencing.")
    if st.session_state.loads is None:
        st.session_state.loads = build_loads(st.session_state.orders)
    loads_view = []
    for ld in st.session_state.loads:
        loads_view.append({
            "load_id": ld.load_id, "mode": ld.mode, "stops": " → ".join(ld.stops),
            "miles": ld.miles, "linehaul_est": ld.linehaul_est, "fsc_est": ld.fsc_est,
            "accessorials_est": ld.accessorials_est, "est_total": ld.total_est,
            "tender": ld.tender, "carrier": ld.carrier or "-"
        })
    st.dataframe(df(loads_view), use_container_width=True)

# Step 4 — Tendering (TL only, sequential)
if step >= 4:
    st.header("4) Tendering (TL only, sequential)")
    st.info("Per business rule: Parcel & LTL are auto-accepted. Only TL loads follow a sequential tender (one carrier at a time) until accepted or expired.")
    if st.session_state.tenders is None:
        st.session_state.loads = sequential_tender_tl(st.session_state.loads)
        st.session_state.tenders = True
    tender_view = []
    for ld in st.session_state.loads:
        tender_view.append({
            "load_id": ld.load_id, "mode": ld.mode, "stops": " → ".join(ld.stops),
            "tender": ld.tender, "carrier": ld.carrier or "-"
        })
    st.dataframe(df(tender_view), use_container_width=True)

# Step 5 — Events
if step >= 5:
    st.header("5) Shipment Events")
    st.info("We simulate pickup, arrivals at each stop for multi-stop TL, and delivery. Parcel/LTL have direct PU→DEL.")
    if st.session_state.events is None:
        st.session_state.events = simulate_events(st.session_state.loads)
    st.dataframe(df([asdict(e) for e in st.session_state.events]), use_container_width=True)

# Step 6 — Invoice & Pay
if step >= 6:
    st.header("6) Invoice & Freight Pay (3-Way Match)")
    st.info("We compare carrier invoices to planned cost. Within ±2% or $10 → APPROVED; otherwise SHORT_PAY if over, approve if underrun.")
    if st.session_state.invoices is None:
        st.session_state.invoices = make_invoices(st.session_state.loads)
        st.session_state.pay = paymatch(st.session_state.loads, st.session_state.invoices)
    st.subheader("Pay Decisions")
    st.dataframe(df([asdict(p) for p in st.session_state.pay]), use_container_width=True)

st.markdown("---")
st.caption("TMS Simulation • Streamlit app • TL consolidation + sequential tendering; Parcel/LTL auto-accept • Reset uses st.rerun()")
