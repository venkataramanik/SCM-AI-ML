# app.py
# TMS Step-by-Step Simulator (Business Walkthrough)
# ---------------------------------------------------
# Shows each phase of a Transportation Management System in sequence:
# Orders → Rating → Load Planning → Tendering → Events → Invoice & Pay

from __future__ import annotations
import math, random, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────
# CORE SETUP
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TMS Step-by-Step Simulator", layout="wide")

def gen_id(prefix: str): return f"{prefix}-{uuid.uuid4().hex[:8]}"
def df(data: list[dict]): return pd.DataFrame(data)

if "step" not in st.session_state:
    st.session_state.step = 0
for key in ["orders","loads","events","invoices","pay","tenders","carriers"]:
    st.session_state.setdefault(key, None)

# ────────────────────────────────────────────────────────────────
# BASIC DATA STRUCTURES
# ────────────────────────────────────────────────────────────────
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
def miles(a,b):
    R=3958.8
    A,B=GEO[a],GEO[b]
    lat1,lon1,lat2,lon2=map(math.radians,[A.lat,A.lon,B.lat,B.lon])
    dlat,dlon=lat2-lat1,lon2-lon1
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

@dataclass
class Order:
    order_id:str; origin:str; destination:str; weight_lb:float; due:datetime
@dataclass
class Load:
    load_id:str; mode:str; origin:str; destination:str; miles:float; total_est:float; tender:str="NOT_SENT"; carrier:Optional[str]=None
@dataclass
class Carrier:
    name:str; modes:list; appetite:float; cost_factor:float
@dataclass
class Event:
    load_id:str; code:str; detail:str; time:str
@dataclass
class Invoice:
    load_id:str; total:float
@dataclass
class PayDecision:
    load_id:str; expected:float; invoiced:float; decision:str

# ────────────────────────────────────────────────────────────────
# LOGIC FUNCTIONS
# ────────────────────────────────────────────────────────────────
def make_orders():
    lanes=[("ATL","JFK"),("ATL","ORD"),("DFW","LAX"),("ORD","DEN"),("DEN","SEA")]
    now=datetime.now()
    orders=[]
    for o,d in lanes:
        w=random.choice([60,2000,8000,35000])
        orders.append(Order(gen_id("ORD"),o,d,w,now+timedelta(days=random.randint(1,3))))
    return orders

def choose_mode(w): 
    if w<=70:return"PARCEL"
    if w<=10000:return"LTL"
    return"TL"

def rate_order(o):
    dist=miles(o.origin,o.destination)
    mode=choose_mode(o.weight_lb)
    base=12 if mode=="PARCEL" else (o.weight_lb/100*32 if mode=="LTL" else dist*2.1)
    total=round(base*1.22,2)
    return Load(gen_id("LOAD"),mode,o.origin,o.destination,round(dist,0),total)

def tender(loads):
    carriers=[
        Carrier("SwiftParcel",["PARCEL"],0.8,0.95),
        Carrier("BlueFreight",["LTL"],0.7,0.9),
        Carrier("RoadRunner",["TL"],0.6,0.92)
    ]
    for ld in loads:
        opts=[c for c in carriers if ld.mode in c.modes]
        for c in opts:
            if random.random()<c.appetite:
                ld.tender="ACCEPTED";ld.carrier=c.name;break
        if not ld.carrier:ld.tender="EXPIRED"
    return loads

def make_events(loads):
    ev=[]
    for ld in loads:
        if ld.tender!="ACCEPTED":continue
        t=datetime.now()
        ev.append(Event(ld.load_id,"PU",f"Picked up by {ld.carrier}",t.strftime("%Y-%m-%d %H:%M")))
        ev.append(Event(ld.load_id,"DEL","Delivered", (t+timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")))
    return ev

def make_invoices(loads):
    inv=[]
    for ld in loads:
        if ld.tender=="ACCEPTED":
            amt=round(ld.total_est*random.uniform(0.98,1.03),2)
            inv.append(Invoice(ld.load_id,amt))
    return inv

def paymatch(loads,invoices):
    out=[]
    exp={ld.load_id:ld.total_est for ld in loads}
    for i in invoices:
        diff=i.total-exp[i.load_id]
        dec="APPROVED" if abs(diff)<=10 else("SHORT_PAY" if diff>0 else"APPROVED")
        out.append(PayDecision(i.load_id,exp[i.load_id],i.total,dec))
    return out

# ────────────────────────────────────────────────────────────────
# USER INTERFACE
# ────────────────────────────────────────────────────────────────
st.title("TMS Step-by-Step Simulator")
st.write("""
This demo shows how a Transportation Management System flows from **Orders → Rating → Planning → Tendering → Events → Pay**.
Use the **Next Step →** button below to advance.
""")

col1,col2=st.columns([1,1])
with col1:
    if st.button("Next Step →"):
        st.session_state.step=min(st.session_state.step+1,6)
with col2:
    if st.button("Reset ↺"):
        for k in st.session_state.keys(): st.session_state[k]=None
        st.session_state.step=0
        st.experimental_rerun()

# ────────────────────────────────────────────────────────────────
# SECTIONS
# ────────────────────────────────────────────────────────────────
step=st.session_state.step

if step>=1:
    st.header("1️⃣ Order Capture")
    st.info("Orders represent customer shipment requests with lanes, weights, and due dates.")
    if st.session_state.orders is None:
        st.session_state.orders=make_orders()
    st.dataframe(df([asdict(o) for o in st.session_state.orders]),use_container_width=True)

if step>=2:
    st.header("2️⃣ Rating & Mode Selection")
    st.info("Each order is priced by mode (Parcel/LTL/TL) based on distance and weight.")
    if st.session_state.loads is None:
        st.session_state.loads=[rate_order(o) for o in st.session_state.orders]
    st.dataframe(df([asdict(l) for l in st.session_state.loads]),use_container_width=True)

if step>=3:
    st.header("3️⃣ Load Planning")
    st.info("Orders are now formalized as loads with estimated cost and distance.")
    st.dataframe(df([asdict(l) for l in st.session_state.loads]),use_container_width=True)

if step>=4:
    st.header("4️⃣ Tendering")
    st.info("Loads are offered to carriers; each may accept or decline.")
    if st.session_state.tenders is None:
        st.session_state.loads=tender(st.session_state.loads)
        st.session_state.tenders=True
    st.dataframe(df([asdict(l) for l in st.session_stat_]()
