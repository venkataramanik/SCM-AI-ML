"""
TRANSPORTATION MANAGEMENT SYSTEM (TMS) – END-TO-END SIMULATION
---------------------------------------------------------------

This Python simulation demonstrates how a Transportation Management System (TMS)
works across the full shipment lifecycle:

1. ORDER CREATION – Capture customer demand (origin, destination, weight, due date)
2. RATING & ROUTING – Select cost-effective mode (Parcel, LTL, TL) and route
3. LOAD PLANNING – Group orders into efficient loads (direct or multi-stop)
4. TENDERING – Offer loads to carriers and simulate accept/decline
5. SHIPMENT EVENTS – Generate pickup, transit, and delivery events
6. INVOICING & FREIGHT PAY – Generate carrier invoices and perform 3-way match

Each section of code below is clearly labeled, with business explanations so viewers
understand what’s happening and why. This is ideal for demo, teaching, or LinkedIn use.

Author: Venkat Krishnan
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import math, random, uuid

# ===========================================================
# SECTION 1: CORE DATA ENTITIES
# ===========================================================
"""
This section defines the key business objects in a TMS:
- Location: Where shipments start/end
- Order: A customer shipment request
- Carrier: A transportation provider
- RateCard: How each mode is priced
- Load: A planned, rated shipment ready to move
- Tender, Event, Invoice, PayDecision: Represent operations, visibility, and billing
"""

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

def haversine_miles(a: Location, b: Location) -> float:
    """Approximate road miles between two points (for cost & ETA estimation)."""
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
    modes: List[str]
    appetite: float
    cost_factor: float

@dataclass
class RateCard:
    parcel_rate_per_lb: float = 1.1
    ltl_rate_per_cwt: float = 32.0
    tl_rate_per_mile: float = 2.1
    fuel_surcharge_pct: float = 0.22
    tl_stop_charge: float = 50.0

@dataclass
class Load:
    load_id: str
    mode: str
    stops: List[str]
    distance_mi: float
    total_est: float
    tender_status: str = "NOT_SENT"
    carrier_id: Optional[str] = None

@dataclass
class Tender:
    tender_id: str
    load_id: str
    carrier_id: str
    status: str

@dataclass
class ShipmentEvent:
    event_id: str
    load_id: str
    timestamp: datetime
    code: str
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
    status: str

# ===========================================================
# SECTION 2: SAMPLE DATA GENERATION
# ===========================================================
"""
Here we simulate a small shipping network: multiple customer orders,
a few carriers, and realistic freight lanes (e.g., ATL → JFK).
"""

def make_orders(now):
    lanes = [("ATL", "JFK"), ("ATL", "ORD"), ("DFW", "LAX"), ("ORD", "DEN"), ("DEN", "SEA")]
    orders = []
    for o, d in lanes:
        weight = random.choice([30, 2000, 8000, 35000])
        ready = now
        due = now + timedelta(days=random.choice([1,2,3]))
        orders.append(Order(gen_id("ORD"), o, d, weight, ready, due))
    return orders

def make_carriers():
    return [
        Carrier("C1", "SwiftParcel", ["PARCEL"], 0.8, 0.95),
        Carrier("C2", "BlueFreight", ["LTL"], 0.7, 0.9),
        Carrier("C3", "RoadRunner", ["TL"], 0.6, 0.92),
        Carrier("C4", "OmniCarrier", ["PARCEL", "LTL", "TL"], 0.65, 0.97)
    ]

# ===========================================================
# SECTION 3: RATING & MODE SELECTION
# ===========================================================
"""
Rates are estimated by mode:
- Parcel: per-lb rate
- LTL: per-100lb (CWT) rate
- TL: per-mile with stop charges

We choose mode based on weight thresholds to simulate optimization.
"""

def choose_mode(weight):
    if weight <= 70: return "PARCEL"
    if weight <= 10000: return "LTL"
    return "TL"

def rate_order(order, rate):
    dist = haversine_miles(GEO[order.origin], GEO[order.destination])
    mode = choose_mode(order.weight_lb)
    if mode == "PARCEL":
        cost = max(12, order.weight_lb * rate.parcel_rate_per_lb)
    elif mode == "LTL":
        cwt = order.weight_lb / 100
        cost = cwt * rate.ltl_rate_per_cwt * (1 + dist/2000)
    else:
        cost = dist * rate.tl_rate_per_mile + rate.tl_stop_charge
    total = cost * (1 + rate.fuel_surcharge_pct)
    return mode, dist, round(total, 2)

# ===========================================================
# SECTION 4: LOAD PLANNING & TENDERING
# ===========================================================
"""
This step transforms rated orders into planned loads,
then tenders them to carriers who may accept or decline.
"""

def plan_loads(orders, rate):
    loads = []
    for o in orders:
        mode, dist, total = rate_order(o, rate)
        loads.append(Load(gen_id("LOAD"), mode, [o.origin, o.destination], dist, total))
    return loads

def tender_loads(loads, carriers):
    tenders = []
    for ld in loads:
        options = [c for c in carriers if ld.mode in c.modes]
        random.shuffle(options)
        for c in options:
            chance = c.appetite - (c.cost_factor - 0.9)
            if random.random() < chance:
                ld.tender_status = "ACCEPTED"
                ld.carrier_id = c.name
                tenders.append(Tender(gen_id("TDR"), ld.load_id, c.carrier_id, "ACCEPTED"))
                break
        if ld.tender_status != "ACCEPTED":
            ld.tender_status = "EXPIRED"
    return tenders

# ===========================================================
# SECTION 5: SHIPMENT EVENTS
# ===========================================================
"""
Simulate real-world tracking events like pickup, transit, and delivery.
These would typically feed visibility dashboards or customer notifications.
"""

def simulate_events(loads):
    events = []
    now = datetime.now()
    for ld in loads:
        if ld.tender_status != "ACCEPTED": continue
        events.append(ShipmentEvent(gen_id("EVT"), ld.load_id, now, "PU", "Picked up"))
        events.append(ShipmentEvent(gen_id("EVT"), ld.load_id, now + timedelta(hours=5), "ARR", "Arrived destination"))
        events.append(ShipmentEvent(gen_id("EVT"), ld.load_id, now + timedelta(hours=6), "DEL", "Delivered"))
    return events

# ===========================================================
# SECTION 6: INVOICE & PAYMENT MATCH
# ===========================================================
"""
Carriers submit invoices. We perform a 3-way match to ensure
the billed amount is within acceptable tolerance of the planned cost.
"""

def create_invoices(loads):
    invoices = []
    for ld in loads:
        if ld.tender_status != "ACCEPTED": continue
        billed = ld.total_est * random.uniform(0.97, 1.05)
        invoices.append(Invoice(gen_id("INV"), ld.load_id, round(billed, 2)))
    return invoices

def match_payments(loads, invoices):
    results = []
    exp = {ld.load_id: ld.total_est for ld in loads}
    for inv in invoices:
        diff = inv.total - exp[inv.load_id]
        status = "APPROVED" if abs(diff) < 10 else "SHORT_PAY"
        results.append(PayDecision(inv.load_id, exp[inv.load_id], inv.total, status))
    return results

# ===========================================================
# SECTION 7: END-TO-END DRIVER
# ===========================================================
"""
This orchestrates the whole business process end-to-end:
orders → rating → loads → tender → events → invoice → pay
and prints summaries at each step.
"""

def run_tms_demo():
    print("\n🚚 TMS SIMULATION START")
    now = datetime.now()
    rate = RateCard()
    orders = make_orders(now)
    carriers = make_carriers()

    print(f"✅ {len(orders)} orders created.")
    loads = plan_loads(orders, rate)
    print(f"📦 {len(loads)} loads planned (Parcel/LTL/TL).")

    tenders = tender_loads(loads, carriers)
    print(f"📤 Tenders sent → {len([t for t in tenders if t.status=='ACCEPTED'])} accepted.")

    events = simulate_events(loads)
    print(f"🛰️ {len(events)} tracking events generated.")

    invoices = create_invoices(loads)
    payments = match_payments(loads, invoices)
    approved = len([p for p in payments if p.status=='APPROVED'])
    print(f"💰 {approved} of {len(payments)} invoices auto-approved.\n")

    print("--- SAMPLE LOADS ---")
    for ld in loads[:3]:
        print(f"{ld.load_id} | {ld.mode} | {ld.tender_status} | ${ld.total_est:.2f}")

if __name__ == "__main__":
    run_tms_demo()
