
"""
TMS STEP-BY-STEP SIMULATOR
==========================
What this does (end to end):
1) Order Capture  -> Create customer shipment orders (origins, destinations, weights, due dates).
2) Rating & Mode  -> Compute Parcel/LTL/TL costs and choose the best-feasible mode per order.
3) Load Planning  -> Turn rated orders into loads (direct Parcel/LTL; TL can consolidate multi-stops).
4) Tendering      -> Offer each load to eligible carriers; simulate accept/decline.
5) Events         -> Generate shipment visibility events (PU, DEP/ARR, EXC, DEL).
6) Invoice & Pay  -> Create carrier invoices and perform 3-way match (approve/short-pay).
Each stage prints a plain-English explanation, then shows a compact result preview.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math, random, uuid

# ---------- Pretty printing helpers ----------

def line(char="─", n=72): return char * n
def h1(text): print(f"\n{line('=')}\n{text}\n{line('=')}")
def h2(text): print(f"\n{text}\n{line()}")
def blurb(text): print(text.strip() + "\n")

def show_rows(title: str, rows: List[Tuple], headers: Tuple[str,...], limit: int = 5):
    print(f"{title}  (showing up to {limit})")
    cols = list(headers)
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows[:limit])) for i, h in enumerate(cols)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*cols))
    print("-+-".join("-" * w for w in widths))
    for r in rows[:limit]:
        print(fmt.format(*[str(x) for x in r]))
    print()

# ---------- Core data & utilities ----------

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

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

def miles(a: str, b: str) -> float:
    A, B = GEO[a], GEO[b]
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [A.lat, A.lon, B.lat, B.lon])
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
    modes: List[str]
    appetite: float  # 0..1
    cost_factor: float  # internal cost vs tariff (~0.85-1.1)

@dataclass
class RateCard:
    parcel_rate_per_lb: float = 1.1
    parcel_min: float = 12.0
    ltl_rate_per_cwt: float = 32.0
    ltl_min: float = 95.0
    tl_rate_per_mile: float = 2.10
    tl_stop_charge: float = 50.0
    tl_min: float = 600.0
    fsc: float = 0.22  # fuel surcharge %

@dataclass
class Load:
    load_id: str
    mode: str
    stops: List[str]              # simple: [orig, dest] (TL could be multi-stop in extensions)
    distance_mi: float
    linehaul_est: float
    fsc_est: float
    accessorials_est: float
    total_est: float
    tender_status: str = "NOT_SENT"
    carrier_id: Optional[str] = None
    carrier_name: Optional[str] = None

@dataclass
class Tender:
    tender_id: str
    load_id: str
    carrier_id: str
    status: str  # SENT / ACCEPTED / DECLINED

@dataclass
class ShipmentEvent:
    event_id: str
    load_id: str
    timestamp: datetime
    code: str     # PU / DEP / ARR / EXC / DEL
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
    status: str    # APPROVED / SHORT_PAY
    note: str

# ---------- Stage 1: Orders ----------

def build_orders(now: datetime) -> List[Order]:
    random.seed(7)  # stable demo
    lanes = [("ATL","JFK"),("ATL","ORD"),("DFW","LAX"),("ORD","DEN"),("DEN","SEA"),("JFK","MIA")]
    orders: List[Order] = []
    for (o,d) in lanes:
        w = random.choice([30, 60, 1200, 4000, 9000, 35000])  # parcel → LTL → TL
        ready = now + timedelta(hours=random.randint(0,6))
        due = ready + timedelta(days=random.choice([1,2,3]))
        orders.append(Order(gen_id("ORD"), o, d, w, ready, due))
    return orders

# ---------- Stage 2: Rating ----------

def choose_mode(weight_lb: float) -> str:
    if weight_lb <= 70: return "PARCEL"
    if weight_lb <= 10000: return "LTL"
    return "TL"

def rate_order(o: Order, tariff: RateCard) -> Tuple[str, float, float, float, float]:
    dist = miles(o.origin, o.destination)
    mode = choose_mode(o.weight_lb)
    if mode == "PARCEL":
        base = max(tariff.parcel_min, o.weight_lb * tariff.parcel_rate_per_lb)
        acc = 0.0
    elif mode == "LTL":
        cwt = o.weight_lb / 100.0
        base = max(tariff.ltl_min, cwt * tariff.ltl_rate_per_cwt * (1 + dist/2000))
        acc = 25.0 if o.weight_lb > 5000 else 0.0
    else:
        base = max(tariff.tl_min, dist * tariff.tl_rate_per_mile) + tariff.tl_stop_charge
        acc = 0.0
    fsc = base * tariff.fsc
    total = base + fsc + acc
    return mode, dist, base, fsc, total

# ---------- Stage 3: Planning ----------

def plan_loads(orders: List[Order], tariff: RateCard) -> List[Load]:
    loads: List[Load] = []
    for o in orders:
        mode, dist, base, fsc, total = rate_order(o, tariff)
        loads.append(Load(gen_id("LOAD"), mode, [o.origin, o.destination], dist, round(base,2),
                          round(fsc,2), 0.0, round(total,2)))
    return loads

# ---------- Stage 4: Tender ----------

def tender(loads: List[Load], carriers: List[Carrier]) -> List[Tender]:
    tenders: List[Tender] = []
    for ld in loads:
        elig = [c for c in carriers if ld.mode in c.modes]
        # rank by likely acceptance: higher appetite, lower cost_factor
        elig.sort(key=lambda c: (-c.appetite, c.cost_factor))
        accepted = False
        for c in elig:
            # Simple acceptance model: appetite adjusted by margin proxy
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

# ---------- Stage 5: Events ----------

def simulate_events(loads: List[Load], start: datetime) -> List[ShipmentEvent]:
    evs: List[ShipmentEvent] = []
    for ld in loads:
        if ld.tender_status != "ACCEPTED":
            continue
        t = start + timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "PU", f"Picked up by {ld.carrier_name}"))
        # depart origin
        t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "DEP", f"Depart {ld.stops[0]}"))
        # travel time proportional to distance (45 mph) + buffer
        t += timedelta(hours=(ld.distance_mi / 45.0) + 1.0)
        # possible small exception (10%)
        if random.random() < 0.1:
            evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "EXC", "Traffic congestion"))
            t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "ARR", f"Arrive {ld.stops[-1]}"))
        t += timedelta(hours=1)
        evs.append(ShipmentEvent(gen_id("EVT"), ld.load_id, t, "DEL", "Delivered, POD signed"))
    return evs

# ---------- Stage 6: Invoices & Pay ----------

def generate_invoices(loads: List[Load], tariff: RateCard) -> List[Invoice]:
    invs: List[Invoice] = []
    for ld in loads:
        if ld.tender_status != "ACCEPTED": continue
        linehaul = ld.linehaul_est * random.uniform(0.98, 1.02)
        fsc = linehaul * tariff.fsc
        acc = 0.0
        total = round(linehaul + fsc + acc, 2)
        invs.append(Invoice(gen_id("INV"), ld.load_id, total))
    return invs

def match_pay(loads: List[Load], invoices: List[Invoice], tol_pct=0.02, tol_abs=10.0) -> List[PayDecision]:
    exp = {ld.load_id: ld.total_est for ld in loads}
    decisions: List[PayDecision] = []
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

# ---------- Demo runner with section-by-section printing ----------

def run_demo():
    random.seed(11)  # stable outputs
    tariff = RateCard()
    now = datetime.now()

    h1("END-TO-END TMS SIMULATION")
    blurb("""
    We will execute each process in sequence. For every section below, you'll see:
    1) A short business explanation of what's happening
    2) A compact printout of the key results for that stage
    """)

    # 1) Orders
    h2("1) ORDER CAPTURE")
    blurb("We create shipment orders with lane, weight, and due times. Good order data drives mode choice, cost, and on-time delivery.")
    orders = build_orders(now)
    rows = [(o.order_id, f"{o.origin}->{o.destination}", o.weight_lb, o.due_time.date()) for o in orders]
    show_rows("Orders", rows, headers=("OrderID","Lane","Weight(lb)","DueDate"))

    # 2) Rating & Mode
    h2("2) RATING & MODE SELECTION (Parcel / LTL / TL)")
    blurb("We compute an estimated charge per order and choose a feasible mode: small → Parcel, mid-weight → LTL, heavy → TL.")
    rated = [(*rate_order(o, tariff), o.order_id) for o in orders]  # (mode, dist, base, fsc, total, order_id)
    rows = [(oid, m, f"{d:.0f}", f"${t:.2f}") for (m,d,base,fsc,t,oid) in rated]
    show_rows("Rated Orders", rows, headers=("OrderID","Mode","Miles","Est Total ($)"))

    # 3) Load Planning
    h2("3) LOAD PLANNING")
    blurb("We turn rated orders into executable loads. Parcel/LTL are direct. (TL consolidation/multi-stop can be added similarly.)")
    loads = plan_loads(orders, tariff)
    rows = [(ld.load_id, ld.mode, f"{ld.distance_mi:.0f}", f"${ld.total_est:.2f}") for ld in loads]
    show_rows("Planned Loads", rows, headers=("LoadID","Mode","Miles","Est Total ($)"))

    # 4) Tendering
    h2("4) TENDERING (Send → Accept/Decline)")
    blurb("We offer each load to eligible carriers. Acceptance reflects appetite and margin. If accepted, we assign the carrier.")
    carriers = [
        Carrier("C1","SwiftParcel",["PARCEL"],0.85,0.95),
        Carrier("C2","BlueFreight",["LTL"],0.70,0.90),
        Carrier("C3","RoadRunner",["TL"],0.55,0.92),
        Carrier("C4","OmniCarrier",["PARCEL","LTL","TL"],0.65,0.97),
    ]
    tdrs = tender(loads, carriers)
    rows = [(ld.load_id, ld.mode, ld.tender_status, (ld.carrier_name or "-")) for ld in loads]
    show_rows("Tender Outcomes", rows, headers=("LoadID","Mode","Tender","Carrier"))

    # 5) Shipment Events
    h2("5) SHIPMENT EVENTS (Visibility)")
    blurb("For accepted loads, we generate pickup (PU), depart/arrive (DEP/ARR), occasional exceptions (EXC), and delivery (DEL).")
    events = simulate_events(loads, now)
    rows = [(e.load_id, e.code, e.detail, e.timestamp.strftime("%Y-%m-%d %H:%M")) for e in events]
    show_rows("Event Stream", rows, headers=("LoadID","Code","Detail","Time"))

    # 6) Invoice & Pay
    h2("6) INVOICE & FREIGHT PAY (3-Way Match)")
    blurb("Carriers submit invoices. We compare to planned cost. If within ±2% or $10 tolerance → APPROVED; else SHORT_PAY or approve if underrun.")
    invoices = generate_invoices(loads, tariff)
    pays = match_pay(loads, invoices)
    rows = [(p.load_id, f"${p.expected:.2f}", f"${p.invoiced:.2f}", p.status, p.note) for p in pays]
    show_rows("Pay Decisions", rows, headers=("LoadID","Expected","Invoiced","Decision","Note"))

    h1("SIMULATION COMPLETE")
    blurb("You now saw each step: Orders → Rating → Planning → Tender → Events → Pay. Adjust tariffs, carriers, or lanes to explore different outcomes.")

if __name__ == "__main__":
    run_demo()
