# app.py
# TMS Step-by-Step Simulator (demo-guaranteed behaviors)
# - Live step panel + tabs
# - Parcel <=150, LTL <=19,000, else TL/IMDL
# - Hub & Spoke: demo-forced for some Parcel/LTL when enabled (clearly labeled)
# - Intermodal (IMDL): for long-haul lanes; guaranteed at least one
# - TL consolidation: packs orders by origin (≤44k), builds multi-stop, shows source orders
# - Sequential tendering (TL/IMDL): deterministic declines→accept→some expire
# - Invoices: deterministic overcharge on a subset → clear rejections/short-pay
# - Analytics dashboard

import math, random, uuid, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import pandas as pd
import streamlit as st

# ── Page + CSS ────────────────────────────────────────────────────
st.set_page_config(page_title="TMS Step-by-Step", layout="wide")
st.markdown("""
<style>
.stButton > button { font-weight:800; font-size:1.05rem; padding:0.6rem 1.1rem; border:2px solid #2c67ff; }
.smallcaps { font-variant: small-caps; color:#444; }
.badge { display:inline-block; padding:.2rem .45rem; border-radius:.4rem; background:#eef; border:1px solid #cfe; margin-right:.35rem;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────
def gen_id(prefix: str) -> str: return f"{prefix}-{uuid.uuid4().hex[:8]}"
def normalize_code(s: str) -> str: return (s or "").strip().upper()
def hflag(s: str, mod: int) -> int:
    # stable hash bucket (for deterministic demo outcomes)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

def ensure_load_defaults(loads):
    fixed = []
    for ld in loads or []:
        if isinstance(ld, dict):
            ld.setdefault("tender_attempts", 0)
            ld.setdefault("carrier", None)
            ld.setdefault("tender", "NOT_SENT")
            ld.setdefault("stops", ld.get("stops", []))
            ld.setdefault("consolidated_from", [])
            ld.setdefault("via_hub", False)
            fixed.append(ld)
        else:
            if not hasattr(ld, "tender_attempts"): ld.tender_attempts = 0
            if not hasattr(ld, "carrier"): ld.carrier = None
            if not hasattr(ld, "tender"): ld.tender = "NOT_SENT"
            if not hasattr(ld, "stops"): ld.stops = []
            if not hasattr(ld, "consolidated_from"): ld.consolidated_from = []
            if not hasattr(ld, "via_hub"): ld.via_hub = False
            fixed.append(ld)
    return fixed

def getf(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

def df_with_schema(rows: List[dict], columns: List[str]) -> pd.DataFrame:
    if not rows: return pd.DataFrame({c: [] for c in columns})
    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns: df[c] = None
    return df[columns]

# ── Domain ───────────────────────────────────────────────────────
@dataclass
class Location: code:str; name:str; lat:float; lon:float
GEO: Dict[str, Location] = {
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
def miles(a: str, b: str) -> float:
    a=normalize_code(a); b=normalize_code(b)
    if a not in GEO or b not in GEO: raise ValueError(f"Unknown code(s): {a}, {b}")
    A,B=GEO[a],GEO[b]; R=3958.8
    lat1,lon1,lat2,lon2=map(math.radians,[A.lat,A.lon,B.lat,B.lon])
    dlat,dlon=lat2-lat1, lon2-lon1
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

@dataclass
class Order:
    order_id:str; origin:str; destination:str; weight_lb:float; due:datetime; service:str="STANDARD"

@dataclass
class Load:
    load_id:str; mode:str; stops:List[str]; miles:float
    linehaul_est:float; fsc_est:float; accessorials_est:float; total_est:float
    tender:str="NOT_SENT"; carrier:Optional[str]=None; tender_attempts:int=0
    consolidated_from:List[str]=None; via_hub:bool=False

@dataclass
class Carrier: name:str; modes:List[str]; appetite:float; cost_factor:float
@dataclass
class Event: load_id:str; code:str; detail:str; time:str
@dataclass
class Invoice: load_id:str; total:float
@dataclass
class PayDecision: load_id:str; expected:float; invoiced:float; decision:str; note:str

# ── Session ──────────────────────────────────────────────────────
STATE_KEYS=["step","orders","loads","events","invoices","pay","tenders_log","lanes_valid","lanes_rejected","seed"]
for k in STATE_KEYS:
    if k not in st.session_state: st.session_state[k]=None
if st.session_state.step is None: st.session_state.step=0
if st.session_state.seed is None: st.session_state.seed=42  # default demo seed

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("Inputs")
seed = st.sidebar.number_input("Random seed", 0, 999999, int(st.session_state.seed or 42), 1)
st.session_state.seed = int(seed)

lanes_default="ATL:JFK, ATL:ORD, DFW:LAX, ORD:DEN, DEN:SEA, ATL:MIA, ATL:SEA"
lanes_str = st.sidebar.text_input("Lanes (origin:dest, comma-separated)", value=lanes_default)

enable_hub = st.sidebar.checkbox("Enable Hub & Spoke routing (demo may force some)", value=True)
hub_code   = st.sidebar.selectbox("Hub", options=sorted(GEO.keys()), index=sorted(GEO.keys()).index("MEM"))

enable_intermodal = st.sidebar.checkbox("Enable Intermodal (IMDL) for long-haul", value=True)
imdl_mile_threshold = st.sidebar.slider("IMDL threshold (miles)", 500, 2500, 1500, step=100)

st.sidebar.caption('<span class="smallcaps">Demo forcing: ~50% hubbing on parcel/ltl; '
                   'guaranteed TL consolidation; predictable tender declines/accepts; invoice rejects.</span>',
                   unsafe_allow_html=True)

st.sidebar.markdown("---")
sb1,sb2,sb3=st.sidebar.columns(3)
with sb1:
    if st.button("Next Step →", key="next_side"):
        st.session_state.step = min(int(st.session_state.step or 0)+1, 6)
with sb2:
    if st.button("Run All", key="runall_side"):
        st.session_state.step = 6
with sb3:
    if st.button("Reset", key="reset_side"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

def parse_lanes(text:str)->Tuple[List[Tuple[str,str]],List[str]]:
    valid,rejected=[],[]
    tokens=[t.strip() for t in (text or "").split(",") if t.strip()]
    for tok in tokens:
        if ":" not in tok: rejected.append(tok); continue
        o,d=[normalize_code(x) for x in tok.split(":",1)]
        if o in GEO and d in GEO: valid.append((o,d))
        else: rejected.append(tok)
    return valid, rejected

lanes_valid, lanes_rejected = parse_lanes(lanes_str)
st.session_state.lanes_valid, st.session_state.lanes_rejected = lanes_valid, lanes_rejected
if lanes_rejected: st.sidebar.warning("Ignored invalid lanes: " + ", ".join(lanes_rejected))
if not lanes_valid:
    st.error("No valid lanes. Fix the lanes in the sidebar."); st.stop()

# ── Business rules ───────────────────────────────────────────────
def choose_mode(weight_lb:float, dist_mi:float)->str:
    if weight_lb <= 150: return "PARCEL"
    if weight_lb <= 19000: return "LTL"
    if enable_intermodal and dist_mi >= imdl_mile_threshold: return "IMDL"
    return "TL"

def make_orders(lanes:List[Tuple[str,str]], seed:int)->List[Order]:
    # Generate multiple orders per lane; ensure some heavy ones for TL consolidation
    random.seed(seed); now=datetime.now(); out=[]
    for o,d in lanes:
        # 3-5 orders per lane for rich consolidation
        for _ in range(random.randint(3,5)):
            # mix of weights, include heavy to trigger TL; some medium to keep multi-stop
            w=random.choice([60,120,800,5000,12000,18000,20000,24000,28000,38000])
            out.append(Order(gen_id("ORD"),o,d,w,now+timedelta(days=random.randint(1,3))))
    return out

def rate_parcel(w,dist): base=max(14.0,w*1.05); fsc=base*0.18; return base,fsc,0.0,base+fsc
def rate_ltl(w,dist):    base=max(110.0,(w/100.0)*30.0*(1.0+dist/2200.0)); fsc=base*0.20; acc=25.0 if w>5000 else 0.0; return base,fsc,acc,base+fsc+acc
def rate_tl(dist,nstops): base=max(700.0,dist*2.10)+max(0,nstops-2)*65.0; fsc=base*0.22; acc=max(0,nstops-2)*20.0; return base,fsc,acc,base+fsc+acc
def rate_imdl(dist,nstops): base=max(650.0,dist*1.60)+max(0,nstops-2)*50.0; fsc=base*0.18; acc=max(0,nstops-2)*15.0; return base,fsc,acc,base+fsc+acc

def nn_sequence(start:str,dests:List[str])->List[str]:
    seq=[]; rem=list(dict.fromkeys(dests)); curr=start
    while rem:
        nxt=min(rem, key=lambda x: miles(curr,x))
        seq.append(nxt); rem.remove(nxt); curr=nxt
    return seq

def maybe_hub_route(o:Order, hub:str) -> Optional[List[str]]:
    """Demo: 50% chance to hub Parcel/LTL when hub is enabled OR if via is 10% shorter."""
    try:
        direct = miles(o.origin, o.destination)
        via = miles(o.origin, hub) + miles(hub, o.destination)
    except ValueError:
        return None
    demo_force = enable_hub and (hflag(o.order_id, 2) == 0)  # ~50%
    shorter = via + 1e-6 < 0.90 * direct
    if enable_hub and (demo_force or shorter) and hub not in [o.origin, o.destination]:
        return [o.origin, hub, o.destination]
    return None

def build_loads(orders:List[Order])->List[Load]:
    loads=[]
    # Parcel/LTL (direct or hubbed) — demo: auto-accept, some hubbed
    for o in orders:
        d = miles(o.origin, o.destination)
        m = choose_mode(o.weight_lb, d)
        if m in ("PARCEL","LTL"):
            hubbed = maybe_hub_route(o, hub_code)
            if hubbed:
                total=0.0; b=f=a=t=0.0
                for a1,b1 in zip(hubbed, hubbed[1:]):
                    seg=miles(a1,b1); total+=seg
                    (b0,f0,a0,t0)=rate_parcel(o.weight_lb,seg) if m=="PARCEL" else rate_ltl(o.weight_lb,seg)
                    b+=b0; f+=f0; a+=a0; t+=t0
                loads.append(Load(gen_id("LOAD"),m,hubbed,round(total,1),round(b,2),round(f,2),round(a,2),round(t,2),
                                  tender="ACCEPTED", carrier=("SwiftParcel" if m=="PARCEL" else "BlueFreight"),
                                  consolidated_from=[o.order_id], via_hub=True))
            else:
                (b,f,a,t)=rate_parcel(o.weight_lb,d) if m=="PARCEL" else rate_ltl(o.weight_lb,d)
                loads.append(Load(gen_id("LOAD"),m,[o.origin,o.destination],round(d,1),round(b,2),round(f,2),round(a,2),round(t,2),
                                  tender="ACCEPTED", carrier=("SwiftParcel" if m=="PARCEL" else "BlueFreight"),
                                  consolidated_from=[o.order_id], via_hub=False))

    # TL/IMDL consolidation by origin (<=44k)
    cap=44000.0
    tl_imdl=[o for o in orders if choose_mode(o.weight_lb, miles(o.origin,o.destination)) in ("TL","IMDL")]
    by_origin:Dict[str,List[Order]]={}
    for o in tl_imdl: by_origin.setdefault(o.origin, []).append(o)

    # Deterministic packing: largest-first; guarantee at least one multi-stop if ≥2 dests
    for origin, group in by_origin.items():
        rem=sorted(group, key=lambda x:-x.weight_lb)
        while rem:
            batch=[]; wsum=0.0; i=0
            while i < len(rem):
                if wsum + rem[i].weight_lb <= cap:
                    batch.append(rem.pop(i)); wsum += batch[-1].weight_lb
                else:
                    i += 1
            if not batch:
                batch.append(rem.pop(0)); wsum=batch[0].weight_lb

            dests=[o.destination for o in batch]
            seq=nn_sequence(origin, dests)
            stops=[origin] + seq
            dist=sum(miles(a,b) for a,b in zip(stops, stops[1:]))

            # Force some IMDL for long-haul: compare cost; if long & cheaper → IMDL
            if enable_intermodal and dist >= imdl_mile_threshold:
                bT,fT,aT,tT=rate_tl(dist, len(stops))
                bI,fI,aI,tI=rate_imdl(dist, len(stops))
                if tI <= tT: mode="IMDL"; base,fsc,acc,tot=bI,fI,aI,tI
                else: mode="TL"; base,fsc,acc,tot=bT,fT,aT,tT
            else:
                mode="TL"; base,fsc,acc,tot=rate_tl(dist, len(stops))

            loads.append(Load(
                gen_id("LOAD"), mode, stops, round(dist,1),
                round(base,2), round(fsc,2), round(acc,2), round(tot,2),
                tender="NOT_SENT", carrier=None, tender_attempts=0,
                consolidated_from=[o.order_id for o in batch], via_hub=False
            ))
    return loads

def sequential_tender(loads:List[Load])->Tuple[List[Load],List[dict]]:
    """Deterministic pattern:
       - First eligible carrier declines for ~50% of loads.
       - Second carrier accepts many of those; a few still expire.
    """
    carriers=[Carrier("RoadRunner",["TL","IMDL"],0.62,0.92),
              Carrier("OmniCarrier",["PARCEL","LTL","TL","IMDL"],0.58,0.97),
              Carrier("BigRig",["TL"],0.50,1.00)]
    log=[]
    for ld in loads:
        if ld.mode in ("PARCEL","LTL"):  # already auto-accepted
            continue
        ld.tender="SENT"
        accepted=False

        # Carrier attempt order: most appetite first
        eligible = [c for c in carriers if ld.mode in c.modes]
        eligible.sort(key=lambda x:(-x.appetite, x.cost_factor, x.name))

        for idx, c in enumerate(eligible):
            ld.tender_attempts += 1
            # Deterministic decline pattern: hash bucket decides the first attempt outcome
            first_decline = (idx == 0 and hflag(ld.load_id, 2) == 0)
            # Cost pressure also influences acceptance: cheaper loads easier
            internal_cost = ld.total_est * c.cost_factor
            margin = (ld.total_est - internal_cost) / max(1.0, internal_cost)
            may_accept = (margin >= -0.05)  # don't accept if deep negative margin

            hit = (not first_decline) and may_accept
            log.append({"load_id": ld.load_id, "mode": ld.mode, "carrier": c.name, "accepted": hit})
            if hit:
                ld.tender="ACCEPTED"; ld.carrier=c.name; accepted=True; break

        if not accepted:
            # A few expire intentionally for demonstration
            if hflag(ld.load_id, 3) == 0:
                ld.tender="EXPIRED"
            else:
                # Last resort: accept by the last carrier to avoid zero-move demos
                last = eligible[-1]
                ld.tender_attempts += 1
                ld.tender="ACCEPTED"; ld.carrier=last.name
                log.append({"load_id": ld.load_id, "mode": ld.mode, "carrier": last.name, "accepted": True})
    return loads, log

def simulate_events(loads:List[Load])->List[Event]:
    out=[]
    for ld in loads:
        if ld.tender!="ACCEPTED": continue
        t0=datetime.now()
        out.append(Event(ld.load_id,"PU",f"Picked up by {ld.carrier or '-'}",t0.strftime("%Y-%m-%d %H:%M")))
        for a,b in zip(ld.stops, ld.stops[1:]):
            t0=t0+timedelta(hours=1+miles(a,b)/45.0)
            out.append(Event(ld.load_id,"ARR",f"Arrived {b}",t0.strftime("%Y-%m-%d %H:%M")))
        out.append(Event(ld.load_id,"DEL","Delivered, POD captured",(t0+timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")))
    return out

def generate_invoices(loads:List[Load])->List[Invoice]:
    """Deterministic invoice behavior:
       - Base noise
       - A subset overcharges by +8–15% to exceed tolerances → rejections
    """
    inv=[]
    for ld in loads:
        if ld.tender!="ACCEPTED": continue
        linehaul=ld.linehaul_est*1.00
        fsc=ld.fsc_est*1.00
        total = round(linehaul+fsc+ld.accessorials_est, 2)

        bucket = hflag(ld.load_id, 5)
        if bucket in (0,1):  # 40% loads: modest noise within tolerance
            bump = total * 0.01 * (bucket+1)  # +1%..+2%
            total = round(total + bump, 2)
        elif bucket == 2:    # ~20%: negative noise (underrun)
            total = round(total - total*0.03, 2)
        else:                # ~40%: deliberate overcharge beyond tolerance
            total = round(total + total*0.12, 2)  # +12%

        inv.append(Invoice(ld.load_id, total))
    return inv

def paymatch(loads:List[Load], invoices:List[Invoice], tol_pct=0.02, tol_abs=10.0)->List[PayDecision]:
    exp={ld.load_id:ld.total_est for ld in loads}; out=[]
    for inv in invoices:
        expected=round(exp.get(inv.load_id,0.0),2)
        diff=inv.total-expected
        within=abs(diff)<=max(tol_abs, expected*tol_pct)
        if within:
            out.append(PayDecision(inv.load_id, expected, inv.total, "APPROVED", "Within tolerance"))
        else:
            out.append(PayDecision(inv.load_id, expected, inv.total,
                                   "SHORT_PAY" if diff>0 else "APPROVED",
                                   "Over tolerance; short-pay excess" if diff>0 else "Underrun; approve"))
    return out

# ── Header ───────────────────────────────────────────────────────
st.title("TMS Step-by-Step Simulator")

tc1,tc2,tc3=st.columns([1,1,1])
with tc1:
    if st.button("Next Step →", key="next_top"):
        st.session_state.step = min(int(st.session_state.step or 0)+1, 6)
with tc2:
    if st.button("Run All", key="runall_top"):
        st.session_state.step = 6
with tc3:
    if st.button("Reset", key="reset_top"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ── Build per step (robust) ──────────────────────────────────────
if st.session_state.step >= 1 and (not st.session_state.orders):
    st.session_state.orders = make_orders(st.session_state.lanes_valid, int(st.session_state.seed))

if st.session_state.step >= 3 and (not st.session_state.loads):
    st.session_state.loads = build_loads(st.session_state.orders or [])
st.session_state.loads = ensure_load_defaults(st.session_state.loads)

if st.session_state.step >= 4 and st.session_state.tenders_log is None:
    st.session_state.loads, st.session_state.tenders_log = sequential_tender(st.session_state.loads or [])
st.session_state.loads = ensure_load_defaults(st.session_state.loads)

if st.session_state.step >= 5 and st.session_state.events is None:
    st.session_state.events = simulate_events(st.session_state.loads or [])

if st.session_state.step >= 6 and (st.session_state.invoices is None or st.session_state.pay is None):
    st.session_state.invoices = generate_invoices(st.session_state.loads or [])
    st.session_state.pay = paymatch(st.session_state.loads or [], st.session_state.invoices or [])

# ── Live Step Panel ──────────────────────────────────────────────
step = int(st.session_state.step or 0)
titles = {0:"Ready",1:"Orders Created",2:"Rating & Mode",3:"Load Planning",4:"Tendering",5:"Shipment Events",6:"Invoice & Freight Pay"}
st.subheader(f"Live Step Panel — {titles.get(step,'Ready')}")

if step == 0:
    st.info("Click **Next Step →** to create Orders.")

elif step == 1:
    st.success("Orders have been created.")
    rows=[{"order_id":o.order_id,"origin":o.origin,"destination":o.destination,"weight_lb":o.weight_lb,"due":o.due.strftime("%Y-%m-%d")}
          for o in (st.session_state.orders or [])]
    st.dataframe(df_with_schema(rows, ["order_id","origin","destination","weight_lb","due"]), use_container_width=True)

elif step == 2:
    rows=[]
    for o in (st.session_state.orders or []):
        d=miles(o.origin,o.destination)
        m=choose_mode(o.weight_lb,d)
        rows.append({"order_id":o.order_id,"lane":f"{o.origin}→{o.destination}","weight_lb":o.weight_lb,"miles":round(d,1),"mode":m})
    st.success("Rating & mode suggestions")
    st.dataframe(df_with_schema(rows, ["order_id","lane","weight_lb","miles","mode"]), use_container_width=True)

elif step == 3:
    if not (st.session_state.loads):
        st.session_state.loads = build_loads(st.session_state.orders or [])
        st.session_state.loads = ensure_load_defaults(st.session_state.loads)
    st.success("Loads planned (multi-stop, hub & spoke visible, IMDL where applicable).")
    rows=[{
        "load_id": getf(ld,"load_id","-"),
        "mode": getf(ld,"mode","-"),
        "stops": " → ".join(getf(ld,"stops",[])),
        "num_stops": len(getf(ld,"stops",[])),
        "via_hub": "YES" if getf(ld,"via_hub", False) else "",
        "consolidated_from": len(getf(ld,"consolidated_from", [])),
        "miles": getf(ld,"miles",0.0),
        "est_total": getf(ld,"total_est",0.0),
        "tender": getf(ld,"tender","-"),
        "carrier": (getf(ld,"carrier",None) or "-")
    } for ld in (st.session_state.loads or [])]
    st.dataframe(df_with_schema(rows, ["load_id","mode","stops","num_stops","via_hub","consolidated_from","miles","est_total","tender","carrier"]),
                 use_container_width=True)

elif step == 4:
    st.success("Tendering (TL/IMDL only): first carrier may decline, next accepts; a few expire.")
    rows=[{
        "load_id":getf(ld,"load_id","-"),"mode":getf(ld,"mode","-"),
        "tender":getf(ld,"tender","-"),"carrier":(getf(ld,"carrier",None) or "-"),
        "attempts":getf(ld,"tender_attempts",0)
    } for ld in (st.session_state.loads or [])]
    st.dataframe(df_with_schema(rows, ["load_id","mode","tender","carrier","attempts"]), use_container_width=True)
    if st.session_state.tenders_log:
        st.write("Tender attempts log:")
        st.dataframe(df_with_schema(st.session_state.tenders_log, ["load_id","mode","carrier","accepted"]), use_container_width=True)

elif step == 5:
    st.success("Shipment events simulated.")
    ev=[asdict(e) for e in (st.session_state.events or [])]
    st.dataframe(df_with_schema(ev, ["load_id","code","detail","time"]), use_container_width=True)

elif step == 6:
    st.success("Invoices & Pay: some invoices intentionally over tolerance → SHORT_PAY.")
    pay_rows=[asdict(p) for p in (st.session_state.pay or [])]
    st.dataframe(df_with_schema(pay_rows, ["load_id","expected","invoiced","decision","note"]), use_container_width=True)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tabs = st.tabs(["Overview","Orders","Rating","Planning","Tendering","Events","Pay","Dashboard"])

with tabs[0]:
    st.subheader("What this simulation shows (end-to-end)")
    st.write("""
1) **Order Capture** → multiple orders per lane.  
2) **Rating & Mode** → **Parcel ≤150 lb**, **LTL ≤19,000 lb**, else **TL/IMDL** for long-haul.  
3) **Planning** → **TL consolidation** into **multi-stop** runs (≤44k), **Hub & Spoke** for some Parcel/LTL (demo-forced & when beneficial), **IMDL** chosen when cheaper on long lanes.  
4) **Tendering** → Parcel/LTL **auto-accept**; **TL/IMDL sequential** tendering with **declines → accept** and some **expire**.  
5) **Events** → PU → ARR (per leg) → DEL.  
6) **Freight Pay** → invoices vs plan; some **over tolerance → SHORT_PAY**.
""")
    st.caption("Look for badges in Planning: **via_hub=YES**, **consolidated_from>1** for multi-stop, and **mode=IMDL** on long lanes.")

with tabs[1]:
    st.subheader("Orders")
    if st.session_state.orders:
        rows=[{"order_id":o.order_id,"origin":o.origin,"destination":o.destination,"weight_lb":o.weight_lb,"due":o.due.strftime("%Y-%m-%d")}
              for o in st.session_state.orders]
        st.dataframe(df_with_schema(rows, ["order_id","origin","destination","weight_lb","due"]), use_container_width=True)
    else: st.info("Click **Next Step →** to create orders.")

with tabs[2]:
    st.subheader("Rating & Mode")
    if not st.session_state.orders: st.info("Create orders first.")
    else:
        rows=[]
        for o in st.session_state.orders:
            d=miles(o.origin,o.destination); m=choose_mode(o.weight_lb,d)
            rows.append({"order_id":o.order_id,"lane":f"{o.origin}→{o.destination}","weight_lb":o.weight_lb,"miles":round(d,1),"mode":m})
        st.dataframe(df_with_schema(rows, ["order_id","lane","weight_lb","miles","mode"]), use_container_width=True)

with tabs[3]:
    st.subheader("Load Planning (consolidation, hub & spoke, intermodal)")
    if not st.session_state.loads:
        st.info("No loads yet. Click **Next Step →**.")
    else:
        rows=[{
            "load_id": getf(ld,"load_id","-"),
            "mode": getf(ld,"mode","-"),
            "stops": " → ".join(getf(ld,"stops",[])),
            "num_stops": len(getf(ld,"stops",[])),
            "via_hub": "YES" if getf(ld,"via_hub", False) else "",
            "consolidated_from": len(getf(ld,"consolidated_from", [])),
            "miles": getf(ld,"miles",0.0),
            "linehaul": getf(ld,"linehaul_est",0.0),
            "fsc": getf(ld,"fsc_est",0.0),
            "access": getf(ld,"accessorials_est",0.0),
            "est_total": getf(ld,"total_est",0.0),
        } for ld in st.session_state.loads]
        st.dataframe(df_with_schema(rows, ["load_id","mode","stops","num_stops","via_hub","consolidated_from","miles","linehaul","fsc","access","est_total"]),
                     use_container_width=True)

with tabs[4]:
    st.subheader("Tendering (TL/IMDL only)")
    if st.session_state.tenders_log is not None:
        st.write("Load statuses:")
        rows=[{
            "load_id":getf(ld,"load_id","-"),"mode":getf(ld,"mode","-"),
            "tender":getf(ld,"tender","-"),"carrier":(getf(ld,"carrier",None) or "-"),
            "attempts":getf(ld,"tender_attempts",0)
        } for ld in (st.session_state.loads or [])]
        st.dataframe(df_with_schema(rows, ["load_id","mode","tender","carrier","attempts"]), use_container_width=True)
        st.write("Tender attempts log:")
        st.dataframe(df_with_schema(st.session_state.tenders_log, ["load_id","mode","carrier","accepted"]), use_container_width=True)
    else: st.info("Advance to tendering.")

with tabs[5]:
    st.subheader("Shipment Events")
    if st.session_state.events is not None:
        st.dataframe(df_with_schema([asdict(e) for e in st.session_state.events], ["load_id","code","detail","time"]), use_container_width=True)
    else: st.info("Advance to events.")

with tabs[6]:
    st.subheader("Invoice & Freight Pay")
    if st.session_state.pay is not None:
        st.dataframe(df_with_schema([asdict(p) for p in st.session_state.pay], ["load_id","expected","invoiced","decision","note"]), use_container_width=True)
    else: st.info("Advance to pay.")

with tabs[7]:
    st.subheader("Analytics Dashboard")
    if st.session_state.loads is None:
        st.info("Run through planning/tender to populate analytics.")
    else:
        load_rows = [{
            "load_id": getf(ld,"load_id","-"),
            "mode": getf(ld,"mode","-"),
            "lane": (f"{getf(ld,'stops',['-'])[0]}→{getf(ld,'stops',['-'])[-1]}" if getf(ld,"stops",[]) else "-"),
            "stops": " → ".join(getf(ld,"stops",[])),
            "miles": getf(ld,"miles",0.0),
            "total": getf(ld,"total_est",0.0),
            "tender": getf(ld,"tender","-"),
            "carrier": (getf(ld,"carrier",None) or "-"),
            "attempts": getf(ld,"tender_attempts",0),
            "via_hub": "YES" if getf(ld,"via_hub", False) else "",
            "consolidated_from": len(getf(ld,"consolidated_from", [])),
        } for ld in (st.session_state.loads or [])]
        loads_df = df_with_schema(load_rows, ["load_id","mode","lane","stops","miles","total","tender","carrier","attempts","via_hub","consolidated_from"])

        # Tender performance (TL/IMDL)
        tl_df = loads_df[loads_df["mode"].isin(["TL","IMDL"])] if not loads_df.empty else pd.DataFrame({c: [] for c in loads_df.columns})
        accept_rate = 0.0 if tl_df.empty else (tl_df["tender"].eq("ACCEPTED").mean()*100.0)
        avg_attempts = 0.0 if tl_df.empty else float(tl_df["attempts"].mean())

        c1,c2,c3=st.columns(3)
        c1.metric("TL/IMDL Acceptance Rate", f"{accept_rate:.1f}%")
        c2.metric("Avg Tender Attempts", f"{avg_attempts:.2f}")
        c3.metric("Loads Planned", f"{len(loads_df)}")

        # Freight cost by lane
        cost_lane = loads_df.groupby("lane", as_index=False)["total"].sum().sort_values("total", ascending=False) if not loads_df.empty \
                    else pd.DataFrame({"lane":[], "total":[]})
        st.write("Freight Cost by Lane (planned)")
        st.dataframe(df_with_schema(cost_lane.to_dict("records"), ["lane","total"]), use_container_width=True)
        if not cost_lane.empty: st.bar_chart(cost_lane.set_index("lane")["total"])

        # Cost per mode & cost/mile
        by_mode = loads_df.groupby("mode", as_index=False).agg(total=("total","sum"), miles=("miles","sum")) if not loads_df.empty \
                  else pd.DataFrame({"mode":[], "total":[], "miles":[]})
        if not by_mode.empty:
            by_mode["cost_per_mile"] = by_mode.apply(lambda r: (r["total"]/r["miles"]) if r["miles"] else 0.0, axis=1)
        else:
            by_mode["cost_per_mile"] = []
        st.write("Cost & Cost/Mile by Mode")
        st.dataframe(df_with_schema(by_mode.to_dict("records"), ["mode","total","miles","cost_per_mile"]), use_container_width=True)
        if not by_mode.empty: st.bar_chart(by_mode.set_index("mode")["total"])

        # Hub usage + consolidation visibility
        hub_usage = loads_df[loads_df["via_hub"] == "YES"] if not loads_df.empty else pd.DataFrame({"load_id":[],"mode":[],"stops":[],"total":[]})
        st.write(f"Hub Usage ({hub_code}) — loads through hub")
        st.dataframe(df_with_schema(hub_usage[["load_id","mode","stops","total"]].to_dict("records"), ["load_id","mode","stops","total"]), use_container_width=True)

        st.write("Multi-stop TL/IMDL (consolidated_from > 1)")
        multi = loads_df[(loads_df["mode"].isin(["TL","IMDL"])) & (loads_df["consolidated_from"] > 1)]
        st.dataframe(df_with_schema(multi[["load_id","mode","stops","consolidated_from","miles","total"]].to_dict("records"),
                                    ["load_id","mode","stops","consolidated_from","miles","total"]), use_container_width=True)
