# app.py
# TMS Step-by-Step Simulator (full version)
# - Tabs (minimize scrolling)
# - Big bold "Next Step →" button (CSS)
# - Parcel <=150, LTL <=19,000, else TL/IMDL
# - Hub & Spoke (optional) and Intermodal (IMDL) for long-haul
# - Parcel/LTL auto-accept; TL/IMDL sequential tendering only
# - TL consolidation (44,000 lb) with multi-stop sequencing
# - Analytics dashboard (tender metrics, cost per lane/mode, cost per mile, hub usage)
# - Robust reset; safe lane parsing; session migration + safe getters to prevent AttributeError

import math, random, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────
# Page config + CSS (big bold buttons)
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="TMS Step-by-Step", layout="wide")
st.markdown("""
<style>
.stButton > button {
  font-weight: 800 !important;
  font-size: 1.05rem !important;
  padding: 0.75rem 1.25rem !important;
  border: 2px solid #2c67ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)

def normalize_code(s: str) -> str:
    return (s or "").strip().upper()

def ensure_load_defaults(loads):
    """Backfill attributes so older session objects (or dicts) don't break rendering."""
    fixed = []
    for ld in loads or []:
        if isinstance(ld, dict):
            ld.setdefault("tender_attempts", 0)
            ld.setdefault("carrier", None)
            ld.setdefault("tender", "NOT_SENT")
            ld.setdefault("stops", ld.get("stops", []))
            fixed.append(ld)
        else:
            if not hasattr(ld, "tender_attempts"): ld.tender_attempts = 0
            if not hasattr(ld, "carrier"): ld.carrier = None
            if not hasattr(ld, "tender"): ld.tender = "NOT_SENT"
            if not hasattr(ld, "stops"): ld.stops = []
            fixed.append(ld)
    return fixed

def getf(obj, name, default=None):
    """Safe getter for dataclass objects or dicts."""
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

# ─────────────────────────────────────────────────────────
# Domain
# ─────────────────────────────────────────────────────────
@dataclass
class Location:
    code: str
    name: str
    lat: float
    lon: float

GEO: Dict[str, Location] = {
    "ATL": Location("ATL","Atlanta, GA",33.75,-84.39),
    "JFK": Location("JFK","New York, NY",40.64,-73.78),
    "ORD": Location("ORD","Chicago, IL",41.97,-87.91),
    "DFW": Location("DFW","Dallas, TX",32.90,-97.04),
    "LAX": Location("LAX","Los Angeles, CA",33.94,-118.41),
    "DEN": Location("DEN","Denver, CO",39.86,-104.67),
    "SEA": Location("SEA","Seattle, WA",47.45,-122.31),
    "MIA": Location("MIA","Miami, FL",25.80,-80.29),
    "MEM": Location("MEM","Memphis, TN",35.15,-90.05),  # hub option
}

def miles(a: str, b: str) -> float:
    a = normalize_code(a); b = normalize_code(b)
    if a not in GEO or b not in GEO:
        raise ValueError(f"Unknown code(s): {a}, {b}")
    A, B = GEO[a], GEO[b]
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians,[A.lat,A.lon,B.lat,B.lon])
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

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
    mode: str                   # "PARCEL","LTL","TL","IMDL"
    stops: List[str]            # includes hub or multi-stop sequence
    miles: float
    linehaul_est: float
    fsc_est: float
    accessorials_est: float
    total_est: float
    tender: str = "NOT_SENT"    # Parcel/LTL set to ACCEPTED immediately
    carrier: Optional[str] = None
    tender_attempts: int = 0

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

# ─────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────
STATE_KEYS = [
    "step","orders","loads","events","invoices","pay","tenders_log",
    "lanes_valid","lanes_rejected","seed"
]
for k in STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.step is None:
    st.session_state.step = 0
if st.session_state.seed is None:
    st.session_state.seed = 11

# ─────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────
st.sidebar.header("Inputs")

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=st.session_state.seed or 11, step=1)
st.session_state.seed = seed

lanes_default = "ATL:JFK, ATL:ORD, DFW:LAX, ORD:DEN, DEN:SEA, ATL:MIA, ATL:SEA"
lanes_str = st.sidebar.text_input("Lanes (origin:dest, comma-separated)", value=lanes_default)

enable_hub = st.sidebar.checkbox("Enable Hub & Spoke routing", value=True)
hub_code = st.sidebar.selectbox("Hub", options=sorted(GEO.keys()), index=sorted(GEO.keys()).index("MEM"))

enable_intermodal = st.sidebar.checkbox("Enable Intermodal (IMDL) for long-haul", value=True)
imdl_mile_threshold = st.sidebar.slider("IMDL threshold (miles)", 500, 2500, 1500, step=100)

st.sidebar.markdown("---")
col_sb1, col_sb2, col_sb3 = st.sidebar.columns(3)
with col_sb1:
    if st.button("Next Step →", key="next_side"):
        st.session_state.step = min((st.session_state.step or 0) + 1, 6)
with col_sb2:
    if st.button("Run All", key="runall_side"):
        st.session_state.step = 6
with col_sb3:
    if st.button("Reset", key="reset_side"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

def parse_lanes(text: str) -> Tuple[List[Tuple[str,str]], List[str]]:
    valid, rejected = [], []
    tokens = [t.strip() for t in (text or "").split(",") if t.strip()]
    for tok in tokens:
        if ":" not in tok:
            rejected.append(tok); continue
        o, d = [normalize_code(x) for x in tok.split(":",1)]
        if o in GEO and d in GEO: valid.append((o,d))
        else: rejected.append(tok)
    return valid, rejected

lanes_valid, lanes_rejected = parse_lanes(lanes_str)
st.session_state.lanes_valid = lanes_valid
st.session_state.lanes_rejected = lanes_rejected
if lanes_rejected:
    st.sidebar.warning("Ignored invalid lanes: " + ", ".join(lanes_rejected))
if not lanes_valid:
    st.error("No valid lanes. Fix the lanes field in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────
# Business Logic
# ─────────────────────────────────────────────────────────
# Mode thresholds per request:
# Parcel <= 150 lb; LTL <= 19,000 lb; else TL/IMDL
def choose_mode(weight_lb: float, dist_mi: float) -> str:
    if weight_lb <= 150: return "PARCEL"
    if weight_lb <= 19000: return "LTL"
    if enable_intermodal and dist_mi >= imdl_mile_threshold:
        return "IMDL"
    return "TL"

def make_orders(lanes: List[Tuple[str,str]], seed: int) -> List[Order]:
    random.seed(seed)
    now = datetime.now()
    out: List[Order] = []
    for o, d in lanes:
        # multiple orders per lane for consolidation and hubbing
        for _ in range(random.choice([1,2,3])):
            w = random.choice([40, 120, 800, 5000, 12000, 18000, 24000, 30000, 38000])
            out.append(Order(gen_id("ORD"), o, d, w, now + timedelta(days=random.randint(1,3))))
    return out

# Simple tariffs
def rate_parcel(weight_lb, dist):
    base = max(14.0, weight_lb * 1.05)
    fsc = base * 0.18
    return base, fsc, 0.0, base + fsc

def rate_ltl(weight_lb, dist):
    base = max(110.0, (weight_lb/100.0) * 30.0 * (1.0 + dist/2200.0))
    fsc = base * 0.20
    acc = 25.0 if weight_lb > 5000 else 0.0
    return base, fsc, acc, base + fsc + acc

def rate_tl(dist, num_stops):
    base = max(700.0, dist * 2.10) + max(0, num_stops-2) * 65.0
    fsc = base * 0.22
    acc = max(0, num_stops-2) * 20.0
    return base, fsc, acc, base + fsc + acc

def rate_imdl(dist, num_stops):
    base = max(650.0, dist * 1.60) + max(0, num_stops-2) * 50.0
    fsc = base * 0.18
    acc = max(0, num_stops-2) * 15.0
    return base, fsc, acc, base + fsc + acc

def nn_sequence(start: str, destinations: List[str]) -> List[str]:
    seq, remaining, curr = [], destinations[:], start
    while remaining:
        nxt = min(remaining, key=lambda x: miles(curr, x))
        seq.append(nxt); remaining.remove(nxt); curr = nxt
    return seq

def maybe_hub_route(o: Order, hub: str) -> Optional[List[str]]:
    if not enable_hub: return None
    try:
        direct = miles(o.origin, o.destination)
        via = miles(o.origin, hub) + miles(hub, o.destination)
    except ValueError:
        return None
    if via + 1e-6 < 0.90 * direct:   # ≥ ~10% shorter
        return [o.origin, hub, o.destination]
    return None

def build_loads(orders: List[Order]) -> List[Load]:
    loads: List[Load] = []
    # Parcel/LTL (direct or hubbed) — auto-accept
    for o in orders:
        try:
            direct_dist = miles(o.origin, o.destination)
        except ValueError:
            direct_dist = 0.0
        mode = choose_mode(o.weight_lb, direct_dist)
        if mode in ("PARCEL","LTL"):
            hubbed = maybe_hub_route(o, hub_code)
            if hubbed:
                total_dist = 0.0; base=fsc=acc=tot=0.0
                for a,b in zip(hubbed, hubbed[1:]):
                    d = miles(a,b); total_dist += d
                    if mode == "PARCEL": b0,f0,a0,t0 = rate_parcel(o.weight_lb, d)
                    else:               b0,f0,a0,t0 = rate_ltl(o.weight_lb, d)
                    base+=b0; fsc+=f0; acc+=a0; tot+=t0
                loads.append(Load(gen_id("LOAD"), mode, hubbed, round(total_dist,1),
                                  round(base,2), round(fsc,2), round(acc,2), round(tot,2),
                                  tender="ACCEPTED", carrier=("SwiftParcel" if mode=="PARCEL" else "BlueFreight")))
            else:
                if mode == "PARCEL": base,fsc,acc,tot = rate_parcel(o.weight_lb, direct_dist)
                else:                 base,fsc,acc,tot = rate_ltl(o.weight_lb, direct_dist)
                loads.append(Load(gen_id("LOAD"), mode, [o.origin,o.destination], round(direct_dist,1),
                                  round(base,2), round(fsc,2), round(acc,2), round(tot,2),
                                  tender="ACCEPTED", carrier=("SwiftParcel" if mode=="PARCEL" else "BlueFreight")))
    # TL/IMDL consolidation by origin (44,000 lb)
    tl_imdl_orders = []
    for o in orders:
        try: d_est = miles(o.origin,o.destination)
        except ValueError: d_est = 0.0
        m = choose_mode(o.weight_lb, d_est)
        if m in ("TL","IMDL"): tl_imdl_orders.append(o)

    by_origin: Dict[str, List[Order]] = {}
    for o in tl_imdl_orders:
        by_origin.setdefault(o.origin, []).append(o)

    trailer_cap = 44000.0
    for origin, group in by_origin.items():
        remaining = sorted(group, key=lambda x: -x.weight_lb)
        while remaining:
            batch, w_sum, i = [], 0.0, 0
            while i < len(remaining):
                if w_sum + remaining[i].weight_lb <= trailer_cap:
                    batch.append(remaining.pop(i)); w_sum += batch[-1].weight_lb
                else:
                    i += 1
            if not batch:
                batch.append(remaining.pop(0)); w_sum = batch[0].weight_lb
            dests = [o.destination for o in batch]
            seq_drops = nn_sequence(origin, list(dict.fromkeys(dests)))
            stops = [origin] + seq_drops
            total_dist = 0.0
            for a,b in zip(stops, stops[1:]):
                total_dist += miles(a,b)
            # choose TL vs IMDL by cost if long-haul and IMDL enabled
            mode_choice = "TL"
            if enable_intermodal and total_dist >= imdl_mile_threshold:
                bT,fT,aT,tT = rate_tl(total_dist, len(stops))
                bI,fI,aI,tI = rate_imdl(total_dist, len(stops))
                if tI < tT: base,fsc,acc,tot = bI,fI,aI,tI; mode_choice = "IMDL"
                else:       base,fsc,acc,tot = bT,fT,aT,tT
            else:
                base,fsc,acc,tot = rate_tl(total_dist, len(stops))
            loads.append(Load(gen_id("LOAD"), mode_choice, stops, round(total_dist,1),
                              round(base,2), round(fsc,2), round(acc,2), round(tot,2)))
    return loads

def sequential_tender(loads: List[Load]) -> Tuple[List[Load], List[dict]]:
    carriers = [
        Carrier("RoadRunner",  ["TL","IMDL"], 0.62, 0.92),
        Carrier("OmniCarrier", ["PARCEL","LTL","TL","IMDL"], 0.58, 0.97),
        Carrier("BigRig",      ["TL"], 0.50, 1.00),
    ]
    tenders_log = []
    for ld in loads:
        if ld.mode in ("PARCEL","LTL"):
            continue
        ld.tender = "SENT"
        accepted = False
        for c in sorted([c for c in carriers if ld.mode in c.modes], key=lambda x: (-x.appetite, x.cost_factor, x.name)):
            ld.tender_attempts += 1
            internal_cost = ld.total_est * c.cost_factor
            margin = (ld.total_est - internal_cost) / max(1.0, internal_cost)
            p = max(0.05, min(0.95, c.appetite * (0.4 + 0.6*(margin + 0.5))))
            hit = random.random() < p
            tenders_log.append({"load_id": ld.load_id, "mode": ld.mode, "carrier": c.name, "accepted": hit})
            if hit:
                ld.tender = "ACCEPTED"; ld.carrier = c.name; accepted = True
                break
        if not accepted:
            ld.tender = "EXPIRED"
    return loads, tenders_log

def simulate_events(loads: List[Load]) -> List[Event]:
    out: List[Event] = []
    for ld in loads:
        if ld.tender != "ACCEPTED": continue
        t0 = datetime.now()
        out.append(Event(ld.load_id, "PU", f"Picked up by {ld.carrier or '-'}", t0.strftime("%Y-%m-%d %H:%M")))
        for a,b in zip(ld.stops, ld.stops[1:]):
            t0 = t0 + timedelta(hours=1 + miles(a,b)/45.0)
            out.append(Event(ld.load_id, "ARR", f"Arrived {b}", t0.strftime("%Y-%m-%d %H:%M")))
        out.append(Event(ld.load_id, "DEL", "Delivered, POD captured", (t0+timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")))
    return out

def generate_invoices(loads: List[Load]) -> List[Invoice]:
    invs: List[Invoice] = []
    for ld in loads:
        if ld.tender != "ACCEPTED": continue
        linehaul = ld.linehaul_est * random.uniform(0.98, 1.02)
        fsc      = ld.fsc_est      * random.uniform(0.95, 1.05)
        acc      = ld.accessorials_est
        total = round(linehaul + fsc + acc, 2)
        invs.append(Invoice(ld.load_id, total))
    return invs

def paymatch(loads: List[Load], invoices: List[Invoice], tol_pct=0.02, tol_abs=10.0) -> List[PayDecision]:
    exp = {ld.load_id: ld.total_est for ld in loads}
    out: List[PayDecision] = []
    for inv in invoices:
        expected = round(exp.get(inv.load_id, 0.0), 2)
        diff = inv.total - expected
        within = abs(diff) <= max(tol_abs, expected * tol_pct)
        if within:
            out.append(PayDecision(inv.load_id, expected, inv.total, "APPROVED", "Within tolerance"))
        else:
            out.append(PayDecision(inv.load_id, expected, inv.total,
                                   "SHORT_PAY" if diff>0 else "APPROVED",
                                   "Over tolerance; short-pay excess" if diff>0 else "Underrun; approve"))
    return out

# ─────────────────────────────────────────────────────────
# Header + top controls (always visible)
# ─────────────────────────────────────────────────────────
st.title("TMS Step-by-Step Simulator")

top_c1, top_c2, top_c3 = st.columns([1,1,1])
with top_c1:
    if st.button("Next Step →", key="next_top"):
        st.session_state.step = min((st.session_state.step or 0) + 1, 6)
with top_c2:
    if st.button("Run All", key="runall_top"):
        st.session_state.step = 6
with top_c3:
    if st.button("Reset", key="reset_top"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────────────────
# Build data per step (with migrations)
# ─────────────────────────────────────────────────────────
if st.session_state.step >= 1 and st.session_state.orders is None:
    st.session_state.orders = make_orders(st.session_state.lanes_valid, st.session_state.seed)

if st.session_state.step >= 3 and st.session_state.loads is None:
    st.session_state.loads = build_loads(st.session_state.orders)
st.session_state.loads = ensure_load_defaults(st.session_state.loads)

if st.session_state.step >= 4 and st.session_state.tenders_log is None:
    st.session_state.loads, st.session_state.tenders_log = sequential_tender(st.session_state.loads)
st.session_state.loads = ensure_load_defaults(st.session_state.loads)

if st.session_state.step >= 5 and st.session_state.events is None:
    st.session_state.events = simulate_events(st.session_state.loads)

if st.session_state.step >= 6 and (st.session_state.invoices is None or st.session_state.pay is None):
    st.session_state.invoices = generate_invoices(st.session_state.loads)
    st.session_state.pay = paymatch(st.session_state.loads, st.session_state.invoices)

# ─────────────────────────────────────────────────────────
# Tabs (minimize scrolling)
# ─────────────────────────────────────────────────────────
tabs = st.tabs(["Overview","Orders","Rating","Planning","Tendering","Events","Pay","Dashboard"])

with tabs[0]:
    st.subheader("What this simulation shows")
    st.write("""
1) **Order Capture** → multiple orders per lane.  
2) **Rating & Mode** → **Parcel ≤150 lb**, **LTL ≤19,000 lb**, else **TL/IMDL** (IMDL for long-haul when enabled).  
3) **Planning** → Parcel/LTL direct **or** **Hub & Spoke**; **TL multi-stop** consolidation by origin (≈44,000 lb).  
4) **Tendering** → Parcel/LTL **auto-accept**; **TL/IMDL sequential tendering** only.  
5) **Events** → PU, ARR per leg, DEL.  
6) **Freight Pay** → Invoices vs plan with tolerances.  
**Dashboard** → tender acceptance, attempts, freight cost per lane/mode, cost per mile, hub usage.
""")

with tabs[1]:
    st.subheader("Orders")
    if st.session_state.orders is not None:
        st.dataframe(df([{
            "order_id": o.order_id, "origin": o.origin, "destination": o.destination,
            "weight_lb": o.weight_lb, "due": o.due.strftime("%Y-%m-%d")
        } for o in st.session_state.orders]), use_container_width=True)
    else:
        st.info("Click **Next Step →** to create orders.")

with tabs[2]:
    st.subheader("Rating & Mode (thresholds + options)")
    if st.session_state.orders is None:
        st.info("Create orders first.")
    else:
        rows=[]
        for o in st.session_state.orders:
            try: d = miles(o.origin,o.destination)
            except ValueError: d = 0.0
            m = choose_mode(o.weight_lb, d)
            rows.append({"order_id":o.order_id,"lane":f"{o.origin}→{o.destination}",
                         "weight_lb":o.weight_lb,"miles":round(d,1),"mode":m})
        st.dataframe(df(rows), use_container_width=True)

with tabs[3]:
    st.subheader("Load Planning (Hub & Spoke + TL/IMDL consolidation)")
    if st.session_state.loads is not None:
        loads_view = []
        for ld in st.session_state.loads:
            loads_view.append({
                "load_id":   getf(ld,"load_id","-"),
                "mode":      getf(ld,"mode","-"),
                "stops":     " → ".join(getf(ld,"stops",[])),
                "miles":     getf(ld,"miles",0.0),
                "linehaul":  getf(ld,"linehaul_est",0.0),
                "fsc":       getf(ld,"fsc_est",0.0),
                "access":    getf(ld,"accessorials_est",0.0),
                "est_total": getf(ld,"total_est",0.0),
                "tender":    getf(ld,"tender","-"),
                "carrier":   (getf(ld,"carrier",None) or "-"),
                "attempts":  getf(ld,"tender_attempts",0)
            })
        st.dataframe(df(loads_view), use_container_width=True)
    else:
        st.info("Advance to Planning.")

with tabs[4]:
    st.subheader("Tendering (TL/IMDL only; sequential)")
    if st.session_state.tenders_log is not None:
        st.write("Parcel/LTL were auto-accepted earlier; only TL/IMDL are tendered here.")
        st.dataframe(df(st.session_state.tenders_log), use_container_width=True)
        st.write("Current load statuses:")
        st.dataframe(df([{
            "load_id": getf(ld,"load_id","-"),
            "mode": getf(ld,"mode","-"),
            "tender": getf(ld,"tender","-"),
            "carrier": (getf(ld,"carrier",None) or "-"),
            "attempts": getf(ld,"tender_attempts",0)
        } for ld in st.session_state.loads]), use_container_width=True)
    else:
        st.info("Advance to tendering.")

with tabs[5]:
    st.subheader("Shipment Events")
    if st.session_state.events is not None:
        st.dataframe(df([asdict(e) for e in st.session_state.events]), use_container_width=True)
    else:
        st.info("Advance to events.")

with tabs[6]:
    st.subheader("Invoice & Freight Pay (3-Way Match)")
    if st.session_state.pay is not None:
        st.dataframe(df([asdict(p) for p in st.session_state.pay]), use_container_width=True)
    else:
        st.info("Advance to pay.")

with tabs[7]:
    st.subheader("Analytics Dashboard")
    if st.session_state.loads is None:
        st.info("Run through planning/tender to populate analytics.")
    else:
        loads_df = df([{
            "load_id": getf(ld,"load_id","-"),
            "mode": getf(ld,"mode","-"),
            "lane": ("%s→%s" % (getf(ld,"stops",['-'])[0], getf(ld,"stops",['-'])[-1]))
                    if getf(ld,"stops",[]) else "-",
            "stops": " → ".join(getf(ld,"stops",[])),
            "miles": getf(ld,"miles",0.0),
            "total": getf(ld,"total_est",0.0),
            "tender": getf(ld,"tender","-"),
            "carrier": (getf(ld,"carrier",None) or "-"),
            "attempts": getf(ld,"tender_attempts",0)
        } for ld in st.session_state.loads])

        # Tender performance (TL/IMDL)
        tl_df = loads_df[loads_df["mode"].isin(["TL","IMDL"])]
        accept_rate = 0.0 if tl_df.empty else (tl_df["tender"].eq("ACCEPTED").mean()*100.0)
        avg_attempts = 0.0 if tl_df.empty else tl_df["attempts"].mean()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("TL/IMDL Acceptance Rate", f"{accept_rate:.1f}%")
        with c2:
            st.metric("Avg Tender Attempts (TL/IMDL)", f"{avg_attempts:.2f}")
        with c3:
            st.metric("Loads Planned", f"{len(loads_df)}")

        # Freight cost per lane (planned)
        cost_lane = loads_df.groupby("lane", as_index=False)["total"].sum().sort_values("total", ascending=False)
        st.write("Freight Cost by Lane (planned)")
        st.dataframe(cost_lane, use_container_width=True)
        if not cost_lane.empty:
            st.bar_chart(cost_lane.set_index("lane")["total"])

        # Cost per mode & cost/mile
        by_mode = loads_df.groupby("mode", as_index=False).agg(total=("total","sum"), miles=("miles","sum"))
        by_mode["cost_per_mile"] = by_mode.apply(lambda r: (r["total"]/r["miles"]) if r["miles"] else 0.0, axis=1)
        st.write("Cost & Cost/Mile by Mode")
        st.dataframe(by_mode, use_container_width=True)
        if not by_mode.empty:
            st.bar_chart(by_mode.set_index("mode")["total"])

        # Hub usage (if hub in route)
        hub_usage = loads_df[loads_df["stops"].str.contains(hub_code, na=False)]
        st.write(f"Hub Usage ({hub_code}) — loads through hub")
        st.dataframe(hub_usage[["load_id","mode","stops","total"]], use_container_width=True)
