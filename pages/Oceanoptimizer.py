"""
Ocean Consolidation Optimizer  —  executive decision-support demo
=================================================================

Single-file Streamlit app. Deploy to Streamlit Community Cloud or run locally:

    pip install streamlit
    streamlit run app.py

Single file, no other dependencies: pandas and numpy ship with Streamlit, and
every chart is rendered in HTML/CSS — so this deploys to Streamlit Community Cloud
with NO requirements.txt needed. All data is synthetic and generated in-file.

WHAT THIS IS
------------
A demonstration of the decision logic for consolidating palletized ocean freight
at a 3PL consolidation center (e.g. CEVA / Dekalb IL) bound for overseas plants
(e.g. Juchen DE via Rotterdam). It answers, per destination + co-load class, the
operational question: SHIP NOW or WAIT FOR MORE PALLETS — anchored not to the
vessel departure but to the *commit-by* time (ETD minus the paperwork/customs/
stuffing cascade), which differs per class.

WHAT THIS IS NOT
----------------
Real money. It runs on 45 synthetic pallets and illustrative rate cards. It proves
the LOGIC. The dollar figure becomes real only after backtesting on actual 3PL
receipts + your freight-settlement rate history. Every screen says so.

ARCHITECTURE
------------
  Partition (group-by, not optimization) -> co-load classes on hard keys
  Pack      (FFD vector bin-pack)         -> pallets -> containers, min boxes
  Time      (stopping rule)               -> wait vs. ship vs. deadline
  Cost      (commitment-first + rate card)-> committed TEU first, spot overflow
  Compare   (horizon sim vs. baseline)    -> savings + timing/packing attribution

The solver is OR, not AI. FFD is a ~30-line heuristic; the timing rule is
arithmetic (expected-days-to-fill vs. days-to-commit-by). Deterministic given seed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# PAGE + DESIGN
# =============================================================================
st.set_page_config(
    page_title="Ocean Consolidation Optimizer",
    page_icon="anchor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design direction: marine executive. Deep navy ink, slate neutrals, one calm
# teal for value/on-track and one disciplined amber for act/deadline, a muted red
# for roll-risk. Structure encodes meaning via a colored left rule (urgency), not
# decoration. No all-caps eyebrows, no template card-shadow spam.
INK      = "#0B1F2A"
NAVY     = "#123047"
TEAL     = "#0E7C86"   # savings / on-track
AMBER    = "#C7811A"   # act now / deadline
RED      = "#B4432F"   # roll risk / hold
SLATE    = "#5A6B75"
PANEL    = "#F5F7F8"
LINE     = "#E3E8EB"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
      html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
      .block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
      h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
      .masthead {{ border-bottom: 3px solid {NAVY}; padding-bottom: .5rem; margin-bottom: .3rem; }}
      .masthead .title {{ font-size: 1.9rem; font-weight: 700; color: {INK}; }}
      .masthead .sub {{ color: {SLATE}; font-size: .95rem; margin-top: .1rem; }}
      .synthbar {{ background: #FBF3E4; border-left: 4px solid {AMBER}; color: #6b4c14;
                   padding: .55rem .8rem; border-radius: 4px; font-size: .85rem; margin: .6rem 0 1.1rem 0; }}
      .kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 8px;
              padding: 1rem 1.1rem; height: 100%; }}
      .kpi .lab {{ color: {SLATE}; font-size: .8rem; font-weight: 500; }}
      .kpi .val {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2.05rem;
                   font-weight: 600; color: {INK}; line-height: 1.1; margin-top: .15rem; }}
      .kpi .val.pos {{ color: {TEAL}; }}
      .kpi .note {{ color: {SLATE}; font-size: .78rem; margin-top: .25rem; }}
      .rec {{ background: #fff; border: 1px solid {LINE}; border-left-width: 5px;
              border-radius: 8px; padding: .85rem 1rem; margin-bottom: .55rem; }}
      .badge {{ display:inline-block; padding:.16rem .55rem; border-radius: 999px;
                font-size:.74rem; font-weight:600; color:#fff; }}
      .chip {{ display:inline-block; padding:.1rem .45rem; border-radius:4px; background:{PANEL};
               color:{SLATE}; font-size:.72rem; margin-right:.3rem; border:1px solid {LINE}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
      hr {{ border-color: {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

ACTION_COLOR = {
    "BOOK NOW":        AMBER,
    "SHIP FULL BOXES": TEAL,
    "WAIT":            NAVY,
    "SHIP AT DEADLINE": RED,
    "HOLD":            SLATE,
}


def html_waterfall(bb_total, d_timing, d_packing, oo_total):
    """A cost bridge rendered in pure CSS — baseline down to optimized."""
    mx = max(bb_total, 1)
    steps = [
        ("Baseline total", 0.0, bb_total, NAVY, f"${bb_total:,.0f}"),
        ("Release timing", bb_total - d_timing, bb_total, TEAL, f"\u2212${d_timing:,.0f}"),
        ("Right-sizing + LCL", oo_total, bb_total - d_timing, TEAL, f"\u2212${d_packing:,.0f}"),
        ("Optimized total", 0.0, oo_total, INK, f"${oo_total:,.0f}"),
    ]
    rows = ""
    for label, bottom, top, color, txt in steps:
        left = 100 * bottom / mx
        width = max(100 * (top - bottom) / mx, 0.6)
        rows += (
            f'<div style="display:flex;align-items:center;margin:.35rem 0;">'
            f'<div style="width:135px;color:{SLATE};font-size:.82rem;text-align:right;'
            f'padding-right:.6rem;">{label}</div>'
            f'<div style="flex:1;position:relative;height:26px;background:{PANEL};'
            f'border-radius:4px;">'
            f'<div style="position:absolute;left:{left}%;width:{width}%;top:0;bottom:0;'
            f'background:{color};border-radius:4px;"></div></div>'
            f'<div style="width:96px;color:{INK};font-size:.82rem;font-weight:600;'
            f'padding-left:.6rem;">{txt}</div></div>'
        )
    return f'<div style="padding:.4rem 0;">{rows}</div>'


def html_cascade(classes, etd, ref_today, buffer_days):
    """Segmented bars per co-load class, right-aligned at a shared ETD, extending
    left by the length of each cascade. Longer cascade = earlier commit-by."""
    seg_colors = ["#0E7C86", "#2E97A0", "#57B0B7", "#83C7CC", "#AEDBDE", "#C7811A"]
    leads = {c: class_lead_days(c, buffer_days) for c in classes}
    max_lead = max(leads.values())
    days_to_etd = (etd - ref_today).days
    bars = ""
    for c in classes:
        legs = cascade_legs(c, buffer_days)
        cb = commit_by(etd, c, buffer_days)
        total = leads[c]
        bar_w = 100 * total / max_lead
        seg_html = ""
        for i, (name, d) in enumerate(legs):
            w = 100 * d / total
            col = seg_colors[i % len(seg_colors)]
            seg_html += (
                f'<div title="{name}: {d}d" style="width:{w}%;background:{col};'
                f'height:100%;display:flex;align-items:center;justify-content:center;'
                f'color:#fff;font-size:.66rem;border-right:1px solid #fff;">'
                f'{d if w>6 else ""}</div>')
        bars += (
            f'<div style="display:flex;align-items:center;margin:.4rem 0;">'
            f'<div style="width:70px;font-weight:600;color:{INK};font-size:.85rem;">{c}</div>'
            f'<div style="width:82px;color:{SLATE};font-size:.74rem;text-align:right;'
            f'padding-right:.5rem;">commit-by<br>{cb:%b %d}</div>'
            f'<div style="flex:1;"><div style="margin-left:{100-bar_w}%;width:{bar_w}%;'
            f'height:24px;display:flex;border-radius:4px;overflow:hidden;">{seg_html}</div></div>'
            f'<div style="width:70px;color:{INK};font-size:.74rem;text-align:left;'
            f'padding-left:.5rem;">ETD<br>{etd:%b %d}</div></div>')
    legend = " &nbsp; ".join(
        f'<span style="font-size:.72rem;color:{SLATE};">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{seg_colors[i % len(seg_colors)]};'
        f'border-radius:2px;margin-right:3px;"></span>{name}</span>'
        for i, (name, _) in enumerate(cascade_legs("BONDED", buffer_days)))
    note = (f'<div style="color:{AMBER};font-size:.78rem;margin-top:.5rem;">'
            f'Reference day is {days_to_etd} days before this ETD.</div>')
    return f'<div style="padding:.3rem 0;">{bars}</div><div style="margin-top:.5rem;">{legend}</div>{note}'

# =============================================================================
# STATIC CONFIG
# =============================================================================
CARRIERS = ["Maersk", "MSC", "Hapag-Lloyd"]
PODS = {                       # port of discharge -> final plant destination
    "Rotterdam": "Juchen, DE",
    "Antwerp":   "Genk, BE",
    "Hamburg":   "Neuss, DE",
}
CLASSES = ["PLANT", "BONDED", "DG"]

# Container physical model. Capacity is PALLET SLOTS (floor positions x stack),
# not raw CBM — a stackable pallet consumes 1 slot, a non-stackable pallet
# consumes 2 (it takes a floor slot and forbids stacking above it). CBM and
# weight are secondary caps.
@dataclass(frozen=True)
class Equip:
    name: str
    slots: int
    cbm: float
    max_kg: int
    teu: int

EQUIP = {
    "20GP":  Equip("20GP",  20, 28.0, 20000, 1),
    "40GP":  Equip("40GP",  40, 58.0, 20000, 2),
    "40HC":  Equip("40HC",  40, 68.0, 20000, 2),
}
EQUIP_ORDER = ["20GP", "40GP", "40HC"]   # small -> large, for downsizing search

# Commit-by cascade legs (days before ETD). Published legs come free with the
# sailing schedule; estimated legs are lead-time parameters; class extras and the
# safety buffer stack on top. PLANT=8, DG=10, BONDED=11 (before buffer).
CASCADE_BASE = {          # applies to every class
    "CY / gate-in (published)":     2,
    "Drayage consol->port (est.)":  2,
    "Stuffing at consol (est.)":    1,
    "Export clearance (stochastic)":3,
}
CASCADE_EXTRA = {
    "DG":     ("DG declaration", 2),
    "BONDED": ("Bonded doc prep", 3),
    "PLANT":  (None, 0),
}

def class_lead_days(coload_class: str, buffer_days: int) -> int:
    base = sum(CASCADE_BASE.values())
    base += CASCADE_EXTRA[coload_class][1]
    return base + buffer_days

def cascade_legs(coload_class: str, buffer_days: int) -> list[tuple[str, int]]:
    legs = list(CASCADE_BASE.items())
    lbl, d = CASCADE_EXTRA[coload_class]
    if lbl:
        legs.append((lbl, d))
    if buffer_days:
        legs.append(("Safety buffer", buffer_days))
    return legs

# =============================================================================
# SYNTHETIC DATA (cached, deterministic)
# =============================================================================
HORIZON_START = date(2026, 9, 1)

@st.cache_data(show_spinner=False)
def gen_pallets(seed: int, horizon_days: int, per_day: int = 14) -> pd.DataFrame:
    # Realistic consol flow: ~per_day pallets/day, skewed to a dominant lane so
    # the main destinations actually fill containers and consolidation matters.
    rng = np.random.default_rng(seed)
    n = horizon_days * per_day
    rows = []
    for i in range(n):
        cls = rng.choice(CLASSES, p=[0.70, 0.20, 0.10])
        pod = rng.choice(list(PODS.keys()), p=[0.55, 0.28, 0.17])
        arrival = HORIZON_START + timedelta(days=int(rng.integers(0, max(horizon_days - 2, 1))))
        stackable = bool(rng.random() < (0.55 if cls == "DG" else 0.78))
        weight = int(rng.integers(220, 900))
        cbm = round(float(rng.uniform(0.8, 2.2)), 2)
        dg_class = None
        bonded = False
        if cls == "DG":
            dg_class = rng.choice(["3", "8", "9"])
        if cls == "BONDED":
            bonded = True
        need_by = arrival + timedelta(days=int(rng.integers(34, 52)))
        rows.append({
            "pallet_id": f"P{i+1:03d}",
            "sto_order": f"STO{rng.integers(4000000, 4999999)}",
            "arrival_date": arrival,
            "pod": pod,
            "destination": PODS[pod],
            "coload_class": cls,
            "dg_class": dg_class,
            "bonded": bonded,
            "weight_kg": weight,
            "cbm": cbm,
            "stackable": stackable,
            "slots": 1 if stackable else 2,
            "need_by": need_by,
        })
    df = pd.DataFrame(rows).sort_values("arrival_date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def gen_sailings(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    etd = HORIZON_START + timedelta(days=4)
    vid = 0
    while etd <= HORIZON_START + timedelta(days=70):
        for carrier in CARRIERS:
            # not every carrier sails every week to every port
            for pod in PODS:
                if rng.random() < 0.82:
                    transit = int(rng.integers(16, 22))
                    rows.append({
                        "sailing_id": f"V{vid:03d}",
                        "carrier": carrier,
                        "vessel": f"{carrier.split('-')[0][:3].upper()}-{rng.integers(100,999)}",
                        "voyage": f"{rng.integers(10,99)}{rng.choice(['E','W'])}",
                        "pod": pod,
                        "etd": etd,
                        "transit_days": transit,
                        "eta": etd + timedelta(days=transit),
                        "cy_cutoff": etd - timedelta(days=3),
                        "si_vgm_cutoff": etd - timedelta(days=4),
                    })
                    vid += 1
        etd += timedelta(days=7)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def gen_commitments(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    rows = []
    for c in CARRIERS:
        annual = int(rng.integers(180, 320))
        used = int(annual * rng.uniform(0.45, 0.7))
        rows.append({
            "carrier": c,
            "annual_teu": annual,
            "used_teu_ytd": used,
            "remaining_teu": annual - used,
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def gen_rates(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    base = {"20GP": 1850, "40GP": 2650, "40HC": 2850}
    rows = []
    for c in CARRIERS:
        f = rng.uniform(0.94, 1.08)
        for eq in EQUIP_ORDER:
            committed = int(base[eq] * f)
            spot = int(committed * rng.uniform(1.18, 1.30))
            rows.append({"carrier": c, "equipment": eq,
                         "committed_usd": committed, "spot_usd": spot})
    return pd.DataFrame(rows)

LCL_RATE_PER_WM = 110.0   # USD per W/M unit (max of CBM or tonne)

# =============================================================================
# ENGINE
# =============================================================================
def commit_by(etd: date, coload_class: str, buffer_days: int) -> date:
    return etd - timedelta(days=class_lead_days(coload_class, buffer_days))

def rate_lookup(rates: pd.DataFrame, carrier: str, equipment: str) -> tuple[int, int]:
    r = rates[(rates.carrier == carrier) & (rates.equipment == equipment)].iloc[0]
    return int(r.committed_usd), int(r.spot_usd)

@dataclass
class Container:
    equipment: str
    pallets: list = field(default_factory=list)
    slots: int = 0
    cbm: float = 0.0
    kg: int = 0

    def fits(self, p, eq: Equip) -> bool:
        return (self.slots + p["slots"] <= eq.slots
                and self.cbm + p["cbm"] <= eq.cbm
                and self.kg + p["weight_kg"] <= eq.max_kg)

    def add(self, p):
        self.pallets.append(p)
        self.slots += p["slots"]
        self.cbm += p["cbm"]
        self.kg += p["weight_kg"]

    def fill_pct(self) -> float:
        eq = EQUIP.get(self.equipment)
        if eq is None:  # LCL: report cube against a 40HC reference
            return round(100 * self.cbm / EQUIP["40HC"].cbm, 1)
        return round(100 * max(self.slots / eq.slots, self.cbm / eq.cbm), 1)


def ffd_pack(pallets: list, allow_downsize: bool, allow_lcl: bool,
             carrier: str, rates: pd.DataFrame) -> list[Container]:
    """First-Fit-Decreasing vector bin-pack. Opens 40HC by default, then (opt
    mode) downsizes each box to the cheapest equipment that still holds its
    contents, and swaps to LCL when that is cheaper for a light box."""
    if not pallets:
        return []
    order = sorted(pallets, key=lambda p: (p["slots"], p["cbm"], p["weight_kg"]),
                   reverse=True)
    hc = EQUIP["40HC"]
    boxes: list[Container] = []
    for p in order:
        placed = False
        for b in boxes:
            if b.fits(p, EQUIP[b.equipment]):
                b.add(p); placed = True; break
        if not placed:
            nb = Container("40HC"); nb.add(p); boxes.append(nb)

    if not allow_downsize:
        return boxes

    for b in boxes:
        # cheapest equipment that fits contents
        best_eq, best_cost = b.equipment, None
        for eq_name in EQUIP_ORDER:
            eq = EQUIP[eq_name]
            if b.slots <= eq.slots and b.cbm <= eq.cbm and b.kg <= eq.max_kg:
                committed, _ = rate_lookup(rates, carrier, eq_name)
                if best_cost is None or committed < best_cost:
                    best_cost, best_eq = committed, eq_name
        b.equipment = best_eq
        if allow_lcl:
            lcl_cost = LCL_RATE_PER_WM * max(b.cbm, b.kg / 1000.0)
            committed, _ = rate_lookup(rates, carrier, b.equipment)
            # LCL only for genuinely small trailing loads that also cost clearly less
            if b.cbm <= 8 and b.fill_pct() < 22 and lcl_cost < 0.75 * committed:
                b.equipment = "LCL"
    return boxes


def cost_boxes(boxes: list[Container], carrier: str, rates: pd.DataFrame,
               remaining_teu: dict) -> float:
    """Commitment-first costing: consume contracted TEU at the committed rate,
    overflow to spot. LCL priced W/M and consumes no commitment."""
    total = 0.0
    for b in boxes:
        if b.equipment == "LCL":
            total += LCL_RATE_PER_WM * max(b.cbm, b.kg / 1000.0)
            continue
        committed, spot = rate_lookup(rates, carrier, b.equipment)
        teu = EQUIP[b.equipment].teu
        if remaining_teu.get(carrier, 0) >= teu:
            total += committed
            remaining_teu[carrier] -= teu
        else:
            total += spot
    return total


def feasible_sailings(sailings: pd.DataFrame, pod: str, coload_class: str,
                      today: date, need_by_min: date, buffer_days: int) -> pd.DataFrame:
    s = sailings[sailings.pod == pod].copy()
    s["commit_by"] = s["etd"].apply(lambda e: commit_by(e, coload_class, buffer_days))
    ok = s[(s["eta"] <= need_by_min) & (s["commit_by"] >= today)]
    return ok.sort_values("etd")


def choose_sailing(feas: pd.DataFrame, batch: list, mode: str,
                   rates: pd.DataFrame, remaining_teu: dict):
    """base: earliest feasible sailing. opt: feasible sailing minimizing batch
    cost (best carrier by rate), tie-break earliest ETD."""
    if feas.empty:
        return None
    if mode == "base":
        return feas.iloc[0]
    best_row, best_cost = None, None
    for _, row in feas.iterrows():
        boxes = ffd_pack(batch, True, True, row.carrier, rates)
        c = cost_boxes(boxes, row.carrier, rates, dict(remaining_teu))  # eval copy
        if best_cost is None or c < best_cost - 1e-6 or (
                abs(c - best_cost) < 1e-6 and row.etd < best_row.etd):
            best_cost, best_row = c, row
    return best_row


def simulate(pallets: pd.DataFrame, sailings: pd.DataFrame, rates: pd.DataFrame,
             commitments: pd.DataFrame, pack_mode: str, time_mode: str,
             baseline_dwell: int, buffer_days: int, hold_rate: float,
             horizon_days: int, capture_snap_on: date | None = None):
    """Simulate one policy = (pack_mode, time_mode) over the horizon on the
    identical pallet stream. Returns totals and (optionally) a per-group snapshot
    of state as of `capture_snap_on` for the live recommendations screen."""
    # Incumbent (base_pack) right-sizes to standard FCL but ships on a fixed
    # cadence with the first available carrier. Optimizer (opt_pack) additionally
    # accumulates toward the deadline, uses LCL for small trailing loads, and
    # picks the cheapest carrier. Timing shows as fuller/fewer boxes; packing as
    # right-sized/LCL tails.
    allow_downsize = True
    allow_lcl = (pack_mode == "opt")
    remaining_teu = {r.carrier: int(r.remaining_teu) for _, r in commitments.iterrows()}

    horizon_end = HORIZON_START + timedelta(days=horizon_days)
    freight = holding = roll_pen = 0.0
    n_containers = 0
    n_lcl = 0
    shipments = []
    snapshots = []

    groups = pallets.groupby(["pod", "coload_class"])
    for (pod, cls), g in groups:
        arrival_rate = max(len(g) / horizon_days, 0.35)  # empirical pallets/day
        waiting: list[dict] = []
        pending = g.sort_values("arrival_date").to_dict("records")
        pi = 0
        day = HORIZON_START
        last_release = HORIZON_START
        while day <= horizon_end:
            while pi < len(pending) and pending[pi]["arrival_date"] <= day:
                waiting.append(pending[pi]); pi += 1

            if waiting:
                need_by_min = min(p["need_by"] for p in waiting)
                feas = feasible_sailings(sailings, pod, cls, day, need_by_min, buffer_days)
                acc_slots = sum(p["slots"] for p in waiting)
                full_box = EQUIP["40HC"].slots

                # deadline = the LATEST sailing that still delivers the earliest-
                # needed pallet on time. That is the true runway to accumulate.
                cb = feas.iloc[-1]["commit_by"] if not feas.empty else None
                days_to_cb = (cb - day).days if cb is not None else -1

                # ---- decide action ----
                action, ship_now, ship_pallets = "WAIT", False, []
                if feas.empty or days_to_cb <= 0:
                    action, ship_now, ship_pallets = "SHIP AT DEADLINE", True, list(waiting)
                elif time_mode == "base":
                    # incumbent: book whatever is on the floor on a fixed cadence
                    if (day - last_release).days >= baseline_dwell:
                        action, ship_now, ship_pallets = "BOOK NOW", True, list(waiting)
                else:  # opt timing
                    if acc_slots >= full_box:
                        # ship complete boxes now, keep the trailing partial
                        big_first = sorted(waiting, key=lambda p: p["slots"], reverse=True)
                        take, s = [], 0
                        for p in big_first:
                            if s + p["slots"] <= (acc_slots // full_box) * full_box:
                                take.append(p); s += p["slots"]
                        if take:
                            action, ship_now, ship_pallets = "SHIP FULL BOXES", True, take
                    # else WAIT (accumulate toward commit-by)

                # snapshot for the live screen (as of capture day, opt policy)
                if capture_snap_on is not None and day == capture_snap_on and waiting:
                    row = feas.iloc[0] if not feas.empty else None
                    snap_boxes = ffd_pack(list(waiting), allow_downsize, allow_lcl,
                                          row.carrier if row is not None else CARRIERS[0], rates)
                    exp_fill = max((b.fill_pct() for b in snap_boxes), default=0)
                    exp_days_fill = (full_box - acc_slots % full_box) / arrival_rate
                    disp_action = action
                    if action == "WAIT" and exp_days_fill > days_to_cb:
                        disp_action = "WAIT"   # will ship partial at deadline
                    snapshots.append({
                        "pod": pod, "destination": PODS[pod], "coload_class": cls,
                        "waiting_pallets": len(waiting), "acc_slots": acc_slots,
                        "acc_cbm": round(sum(p["cbm"] for p in waiting), 1),
                        "acc_kg": sum(p["weight_kg"] for p in waiting),
                        "action": disp_action,
                        "carrier": row.carrier if row is not None else "—",
                        "vessel": row.vessel if row is not None else "—",
                        "etd": row.etd if row is not None else None,
                        "commit_by": cb, "days_to_commit_by": days_to_cb,
                        "containers": len(snap_boxes),
                        "box_types": ", ".join(b.equipment for b in snap_boxes),
                        "fill_pct": exp_fill,
                        "arrival_rate": round(arrival_rate, 2),
                        "exp_days_to_fill": round(exp_days_fill, 1),
                        "need_by_min": need_by_min,
                    })

                # ---- execute ship ----
                if ship_now and ship_pallets:
                    row = choose_sailing(feas, ship_pallets, time_mode, rates, remaining_teu) \
                          if not feas.empty else None
                    carrier = row.carrier if row is not None else CARRIERS[0]
                    boxes = ffd_pack(ship_pallets, allow_downsize, allow_lcl, carrier, rates)
                    ship_cost = cost_boxes(boxes, carrier, rates, remaining_teu)
                    freight += ship_cost
                    n_containers += sum(1 for b in boxes if b.equipment != "LCL")
                    n_lcl += sum(1 for b in boxes if b.equipment == "LCL")
                    last_release = day
                    for p in ship_pallets:
                        dwell = (day - p["arrival_date"]).days
                        holding += hold_rate * max(dwell, 0)
                    rolled = (action == "SHIP AT DEADLINE" and feas.empty)
                    if rolled:
                        roll_pen += 1500 * len(boxes)  # expedite/roll penalty
                    shipments.append({"pod": pod, "destination": PODS[pod], "class": cls,
                                      "release": day, "carrier": carrier,
                                      "containers": len(boxes), "pallets": len(ship_pallets),
                                      "cost": ship_cost, "rolled": rolled})
                    ship_ids = {p["pallet_id"] for p in ship_pallets}
                    waiting = [p for p in waiting if p["pallet_id"] not in ship_ids]

            day += timedelta(days=1)

        # flush remainder at horizon end (identical treatment both policies)
        if waiting:
            need_by_min = min(p["need_by"] for p in waiting)
            feas = feasible_sailings(sailings, pod, cls, horizon_end, need_by_min, buffer_days)
            row = choose_sailing(feas, waiting, time_mode, rates, remaining_teu) \
                  if not feas.empty else None
            carrier = row.carrier if row is not None else CARRIERS[0]
            boxes = ffd_pack(waiting, allow_downsize, allow_lcl, carrier, rates)
            ship_cost = cost_boxes(boxes, carrier, rates, remaining_teu)
            freight += ship_cost
            n_containers += sum(1 for b in boxes if b.equipment != "LCL")
            n_lcl += sum(1 for b in boxes if b.equipment == "LCL")
            for p in waiting:
                holding += hold_rate * max((horizon_end - p["arrival_date"]).days, 0)
            rolled = feas.empty
            if rolled:
                roll_pen += 1500 * len(boxes)
            shipments.append({"pod": pod, "destination": PODS[pod], "class": cls,
                              "release": horizon_end, "carrier": carrier,
                              "containers": len(boxes), "pallets": len(waiting),
                              "cost": ship_cost, "rolled": rolled})

    total = freight + holding + roll_pen
    return {
        "total": total, "freight": freight, "holding": holding, "roll": roll_pen,
        "containers": n_containers, "lcl": n_lcl,
        "shipments": shipments, "snapshots": snapshots,
    }

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### Model controls")
    seed = st.number_input("Data seed", 1, 9999, 42, help="Regenerate the synthetic scenario.")
    horizon_days = st.slider("Horizon (days)", 21, 45, 30,
                             help="Window over which both policies are compared on the same pallets.")
    baseline_dwell = st.slider("Baseline dwell trigger (days)", 5, 18, 10,
                               help="'As-is' policy releases a group when the oldest pallet has waited this long, or at commit-by.")
    buffer_days = st.slider("Safety buffer (days)", 0, 4, 1,
                            help="Slack added to the commit-by cascade to absorb clearance variance. Higher = ship earlier, lower fill, less roll risk.")
    hold_rate = st.slider("Holding cost ($/pallet/day)", 4, 40, 12,
                          help="STO capital-in-transit + consol storage. This is the cost of waiting.")
    ref_offset = st.slider("'Today' for live screen (day of horizon)", 5, 25, 14,
                           help="Reference date the recommendations screen is computed as of.")
    st.markdown("---")
    st.caption("Solver: FFD vector bin-pack + stopping rule. OR, not AI. "
               "Deterministic given seed.")

reference_today = HORIZON_START + timedelta(days=ref_offset)

# =============================================================================
# BUILD DATA + RUN THE THREE POLICY RUNS
# =============================================================================
pallets = gen_pallets(seed, horizon_days)
sailings = gen_sailings(seed)
commitments = gen_commitments(seed)
rates = gen_rates(seed)

common = dict(pallets=pallets, sailings=sailings, rates=rates, commitments=commitments,
              baseline_dwell=baseline_dwell, buffer_days=buffer_days,
              hold_rate=hold_rate, horizon_days=horizon_days)

run_bb = simulate(pack_mode="base", time_mode="base", **common)              # as-is
run_bo = simulate(pack_mode="base", time_mode="opt", **common)               # timing only
run_oo = simulate(pack_mode="opt",  time_mode="opt",
                  capture_snap_on=reference_today, **common)                 # full optimizer

savings_total   = run_bb["total"] - run_oo["total"]
delta_timing    = run_bb["total"] - run_bo["total"]
delta_packing   = run_bo["total"] - run_oo["total"]
containers_base = run_bb["containers"]
containers_opt  = run_oo["containers"]

# =============================================================================
# MASTHEAD + SHARED HELPERS
# =============================================================================
st.markdown(
    f"""<div class="masthead">
          <div class="title">Ocean Consolidation Optimizer</div>
          <div class="sub">Ship-now-or-wait decisions for palletized STO freight &nbsp;·&nbsp;
          CEVA Dekalb, IL &rarr; EU plants &nbsp;·&nbsp; anchored to commit-by, not vessel ETD</div>
        </div>""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="synthbar"><b>Demonstration on {len(pallets)} synthetic pallets.</b>
        This proves the decision logic and its constraints — not the dollar figure. Real savings
        require backtesting on actual 3PL receipts and your freight-settlement rate history.</div>""",
    unsafe_allow_html=True,
)


def kpi(col, label, value, note="", pos=False):
    col.markdown(
        f"""<div class="kpi"><div class="lab">{label}</div>
             <div class="val {'pos' if pos else ''}">{value}</div>
             <div class="note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def action_legend():
    items = [
        ("BOOK NOW", AMBER, "release today — the cutoff is close"),
        ("SHIP FULL BOXES", TEAL, "a container is full, send it now"),
        ("WAIT", NAVY, "keep collecting pallets, there's still time"),
        ("SHIP AT DEADLINE", RED, "last call — must go now or miss the boat"),
    ]
    chips = "".join(
        f'<span style="display:inline-block;margin:.15rem .5rem .15rem 0;">'
        f'<span class="badge" style="background:{c}">{name}</span>'
        f'<span style="color:{SLATE};font-size:.78rem;"> &nbsp;{desc}</span></span>'
        for name, c, desc in items)
    st.markdown(f'<div style="margin:.3rem 0 .8rem 0;">{chips}</div>', unsafe_allow_html=True)


def explainer(text):
    st.markdown(
        f'<div style="background:{PANEL};border-left:4px solid {TEAL};padding:.6rem .85rem;'
        f'border-radius:4px;color:{INK};font-size:.9rem;margin:.2rem 0 .9rem 0;">{text}</div>',
        unsafe_allow_html=True)


def render_worklist(snaps, persona):
    """persona='consol' → operational verbs; 'company' → recommendation framing."""
    if not snaps:
        st.info("Nothing waiting at the reference date for this seed. "
                "Move the '\u2018Today\u2019 for live screen' slider in the sidebar.")
        return
    snaps = sorted(snaps, key=lambda s: (s["days_to_commit_by"], -s["waiting_pallets"]))
    for s in snaps:
        color = ACTION_COLOR.get(s["action"], SLATE)
        dg = " · DG" if s["coload_class"] == "DG" else ""
        bond = " · bonded" if s["coload_class"] == "BONDED" else ""
        if s["action"] == "SHIP FULL BOXES":
            why = "one or more containers are already full — no reason to hold a full box."
        elif s["action"] in ("SHIP AT DEADLINE", "BOOK NOW"):
            why = (f"the commit-by cutoff is {s['days_to_commit_by']} day(s) away — release now "
                   f"so the box clears customs, stuffing and gate-in before the vessel.")
        else:
            why = (f"this lane receives about {s['arrival_rate']} pallets/day, so a box fills in "
                   f"~{s['exp_days_to_fill']} days — sooner than the {s['days_to_commit_by']} days "
                   f"left before commit-by. Waiting means a fuller, cheaper box.")
        etd_str = f"{s['etd']:%b %d}" if s["etd"] is not None else "no feasible sailing"
        cb_str = f"{s['commit_by']:%b %d}" if s["commit_by"] is not None else "—"
        verb = "Action" if persona == "consol" else "Recommend"
        st.markdown(
            f"""<div class="rec" style="border-left-color:{color}">
                  <span class="badge" style="background:{color}">{s['action']}</span>
                  &nbsp;<b>{s['destination']}</b> &nbsp;<span class="chip">{s['coload_class']}{dg}{bond}</span>
                  <span class="chip">via {s['pod']}</span>
                  <div style="margin-top:.45rem; color:{INK}; font-size:.9rem;">
                    {s['waiting_pallets']} pallets on the floor · {s['acc_cbm']} CBM · {s['acc_kg']:,} kg
                    &nbsp;|&nbsp; {verb}: {s['containers']} box(es) [{s['box_types']}] at ~{s['fill_pct']}% full
                    &nbsp;|&nbsp; {s['carrier']} {s['vessel']}, ETD {etd_str}
                  </div>
                  <div style="margin-top:.3rem; color:{SLATE}; font-size:.83rem;">
                    Commit-by <b>{cb_str}</b> ({s['days_to_commit_by']} days out) ·
                    plant needs it by {s['need_by_min']:%b %d}<br>
                    Why: {why}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )


def render_rules():
    st.markdown("#### The decision rules, in plain language")
    explainer("These are the fixed rules the tool follows every time. No black box — each "
              "recommendation on the worklist can be traced back to one of these.")
    lead = {c: class_lead_days(c, buffer_days) for c in CLASSES}
    st.markdown(f"""
**Rule 1 — Never mix the wrong pallets (partition).**
Pallets are grouped by destination port, co-load class, and hazmat/bonded status, and
those groups are *never* combined into the same container. A DG pallet never rides with
a plant pallet; Rotterdam cargo never rides with Hamburg cargo. This is a hard safety and
customs rule, applied before anything else.

**Rule 2 — Fill the box by slots, not just cube (packing).**
Each container has a fixed number of pallet slots: **20GP = 20**, **40GP = 40**, **40HC = 40**
(plus cube and a 20-tonne weight cap). A normal stackable pallet takes **1 slot**; a
non-stackable pallet takes **2** (nothing can go on top). The tool packs each group into the
fewest boxes, then picks the *smallest* container that still fits — a half-empty 40-footer
becomes a 20-footer, or LCL if it's tiny.

**Rule 3 — Wait for a fuller box, but never miss the cutoff (timing).**
A full box ships immediately. A partial box waits and keeps collecting pallets — but only
until **commit-by**, the real deadline. Commit-by is *earlier* than the vessel's departure
because paperwork, customs, drayage and stuffing all take time:

- **PLANT** cargo: {lead['PLANT']} days before ETD
- **DG** cargo: {lead['DG']} days before ETD (extra day for the dangerous-goods declaration)
- **BONDED** cargo: {lead['BONDED']} days before ETD (extra days for bonded paperwork)

*(includes the {buffer_days}-day safety buffer set in the sidebar.)*

**Rule 4 — Use the freight you already paid for (cost).**
The company holds volume contracts with the carriers. The tool fills that committed space first (the
cheaper contracted rate) and only pays the expensive spot rate once the contract is used up.

**Rule 5 — Pick the cheapest carrier that still makes the date.**
Among the sailings that can still deliver on time, choose the one with the lowest all-in cost.
""")


# =============================================================================
# VIEW SELECTOR
# =============================================================================
view = st.radio(
    "Choose a view",
    ["Company view (cargo owner)", "CEVA — Consol provider view (the dock)"],
    horizontal=True, label_visibility="collapsed")
IS_COMPANY = view.startswith("Company")

# shared computed metrics
ship_df = pd.DataFrame(run_oo["shipments"])
total_pallets = len(pallets)
rolled_pallets = int(ship_df[ship_df["rolled"]]["pallets"].sum()) if not ship_df.empty and ship_df["rolled"].any() else 0
on_time_pct = 100 * (total_pallets - rolled_pallets) / max(total_pallets, 1)
contract_open = int(commitments["remaining_teu"].sum())
snaps = run_oo["snapshots"]
floor_pallets = sum(s["waiting_pallets"] for s in snaps)
action_set = ("BOOK NOW", "SHIP FULL BOXES", "SHIP AT DEADLINE")
groups_action = sum(1 for s in snaps if s["action"] in action_set)
boxes_to_stuff = sum(s["containers"] for s in snaps if s["action"] in action_set)
avg_fill = np.mean([s["fill_pct"] for s in snaps]) if snaps else 0

# =============================================================================
# COMPANY VIEW  (cargo owner)
# =============================================================================
if IS_COMPANY:
    explainer("<b>You're seeing the company's view — the cargo owner.</b> This is the leadership "
              "picture: what consolidation saves, whether plants still get their parts on "
              "time, and how well the ocean contracts are being used. The company sets the policy; "
              "CEVA executes it on the dock. "
              "<b style='color:%s'>Green</b> = savings / on track." % TEAL)

    c1, c2, c3, c4 = st.columns(4)
    pct = (savings_total / run_bb["total"] * 100) if run_bb["total"] else 0
    kpi(c1, "Freight savings vs. today", f"${savings_total:,.0f}",
        f"{pct:.0f}% lower over {horizon_days} days", pos=True)
    kpi(c2, "Parts on time to plants", f"{on_time_pct:.0f}%",
        "shipments that make the plant need-by date")
    net_moves = (containers_base + run_bb["lcl"]) - (containers_opt + run_oo["lcl"])
    kpi(c3, "Ocean containers", f"{containers_base} \u2192 {containers_opt}",
        f"{run_oo['lcl']} tiny loads moved to LCL · {net_moves} fewer moves")
    kpi(c4, "Contract space still open", f"{contract_open} TEU",
        "committed volume left to use before paying spot")
    st.markdown("")

    t_cost, t_actions, t_service, t_commit, t_rules, t_method = st.tabs(
        ["Cost & savings", "Recommended releases", "Service by plant",
         "Carrier contracts", "Decision rules", "How it works"])

    with t_cost:
        st.markdown("#### What consolidation saves, and where it comes from")
        explainer("We run the <b>same pallets</b> two ways over the same month — the way it "
                  "works today vs. the optimized way — and compare the total bill. The gap is "
                  "the saving. It splits into two levers: shipping <b>fuller boxes</b> (timing) "
                  "and using the <b>right-size box</b> (packing).")
        colL, colR = st.columns([1.1, 1])
        with colL:
            st.markdown(html_waterfall(run_bb["total"], delta_timing, delta_packing,
                                       run_oo["total"]), unsafe_allow_html=True)
            st.caption("Read top to bottom: today's cost, minus each lever, equals the "
                       "optimized cost.")
        with colR:
            comp = pd.DataFrame({
                "Policy": ["Today (as-is)", "+ better timing", "+ right-sizing"],
                "Freight": [run_bb["freight"], run_bo["freight"], run_oo["freight"]],
                "Holding": [run_bb["holding"], run_bo["holding"], run_oo["holding"]],
                "Total": [run_bb["total"], run_bo["total"], run_oo["total"]],
                "FCL": [run_bb["containers"], run_bo["containers"], run_oo["containers"]],
                "LCL": [run_bb["lcl"], run_bo["lcl"], run_oo["lcl"]],
            })
            st.dataframe(comp.style.format({"Freight": "${:,.0f}", "Holding": "${:,.0f}",
                                            "Total": "${:,.0f}"}),
                         use_container_width=True, hide_index=True)
        fcl_saved = run_bb["containers"] - run_bo["containers"]
        if delta_timing >= delta_packing:
            bigger_txt = ("the bigger lever here is <b>timing</b> — waiting for fuller boxes "
                          "before the cutoff ships fewer, cheaper containers overall")
        else:
            bigger_txt = ("the bigger lever here is <b>right-sizing</b> — thin lanes (DG, bonded, "
                          "smaller ports) book full boxes they don't fill")
        st.markdown(
            f"""<div style="font-size:.88rem;color:{SLATE};">
            <b>How to read the split.</b> Shipping fuller boxes (better <b>timing</b>) saves
            <b style="color:{INK}">${delta_timing:,.0f}</b> and cuts {fcl_saved} containers.
            Using the right-size box and LCL for tiny loads (<b>right-sizing</b>) saves
            <b style="color:{INK}">${delta_packing:,.0f}</b>. In this synthetic scenario
            {bigger_txt}. Timing also quietly protects the ocean contract and prevents rolled
            cargo — value that doesn't show up in this one number.</div>""",
            unsafe_allow_html=True)

    with t_actions:
        st.markdown(f"#### What we'd ask CEVA to do — as of {reference_today:%b %d, %Y}")
        explainer("These are the release decisions the company would hand to the consol provider. "
                  "One line per destination + cargo type. Today this call sits with CEVA — "
                  "moving it onto these rules is the operating-model change.")
        action_legend()
        render_worklist(snaps, "company")

    with t_service:
        st.markdown("#### Are plants getting their parts on time?")
        explainer("Consolidating saves money by waiting for fuller boxes — but never past the "
                  "date a plant needs the parts. This is the check that service holds up.")
        if not ship_df.empty:
            sd = ship_df.copy()
            sd["rolled_pallets"] = sd.apply(
                lambda r: r["pallets"] if r["rolled"] else 0, axis=1)
            svc = sd.groupby("destination").agg(
                Pallets=("pallets", "sum"), Containers=("containers", "sum"),
                Freight=("cost", "sum"), RolledPallets=("rolled_pallets", "sum")
            ).reset_index()
            svc["On-time %"] = (100 * (svc["Pallets"] - svc["RolledPallets"])
                                / svc["Pallets"]).round(0)
            svc = svc.drop(columns=["RolledPallets"]).rename(
                columns={"destination": "Plant (via port)"})
            st.dataframe(svc.style.format({"Freight": "${:,.0f}", "On-time %": "{:.0f}%"}),
                         use_container_width=True, hide_index=True)
        st.caption(f"Overall {on_time_pct:.0f}% of pallets make the plant need-by date under "
                   f"the optimized plan. At-risk = no sailing could meet the date, so the load "
                   f"needs expedite — the tool flags these early instead of at the port.")

    with t_commit:
        st.markdown("#### Carrier contracts — use what we've paid for")
        explainer("The company commits to a minimum volume with each carrier for a lower rate. The tool "
                  "fills that committed space first and only pays the pricey spot rate once it's "
                  "used up — so we don't leave contracted freight on the table.")
        merged = commitments.merge(
            rates[rates.equipment == "40HC"][["carrier", "committed_usd", "spot_usd"]],
            on="carrier")
        for _, r in merged.iterrows():
            st.markdown(f"**{r.carrier}** — {r.used_teu_ytd}/{r.annual_teu} TEU used · "
                        f"{r.remaining_teu} still open · 40HC ${r.committed_usd:,} contracted "
                        f"vs ${r.spot_usd:,} spot")
            st.progress(min(r.used_teu_ytd / r.annual_teu, 1.0))
        st.dataframe(merged.rename(columns={
            "annual_teu": "Annual TEU", "used_teu_ytd": "Used YTD", "remaining_teu": "Open",
            "committed_usd": "40HC contracted", "spot_usd": "40HC spot"}),
            use_container_width=True, hide_index=True)

    with t_rules:
        render_rules()

    with t_method:
        st.markdown("#### How it works, and what's real vs. synthetic")
        st.markdown(f"""
The engine is **operations research, not AI** — three simple layers: group the pallets that
can legally travel together, pack them into the fewest right-size boxes, and time the release
to fill boxes without missing the cutoff. Every number here is **synthetic** and generated
inside the app to show the logic.

**The honest next step.** To turn this into a real dollar figure, run the same engine over
6–12 months of actual consol-provider receipts and the company's settlement rates — offline, no system change.

**Governance.** Today CEVA owns the release decision. Adopting these rules is an operating-model
change, not just software. And once real receipts and rates go in, this moves off any public
host and behind the company's environment.

**Known simplifications.** DG is a simple flag here (real life is a compatibility matrix);
forward visibility uses average arrival rate rather than live ASNs; customs variance is a fixed
buffer rather than a fitted distribution. All three deepen easily once real data is in hand.
""")

# =============================================================================
# CONSOL PROVIDER VIEW  (CEVA — the dock)
# =============================================================================
else:
    explainer("<b>You're seeing CEVA's view — the consolidation dock.</b> This is the "
              "operational screen: what's on the floor right now, which groups to book or hold "
              "today, and the cutoffs you have to hit. "
              "<b style='color:%s'>Amber</b> = act today, "
              "<b style='color:%s'>red</b> = last call." % (AMBER, RED))

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Pallets on the floor", f"{floor_pallets}",
        f"waiting across {len(snaps)} destination/cargo groups")
    kpi(c2, "Groups to action today", f"{groups_action}",
        "need a book / ship decision now")
    kpi(c3, "Boxes to stuff", f"{boxes_to_stuff}",
        "containers to load from today's actions")
    kpi(c4, "Average fill", f"{avg_fill:.0f}%",
        "how full the planned boxes are")
    st.markdown("")

    t_work, t_floor, t_cut, t_sched, t_rules = st.tabs(
        ["Today's worklist", "Floor & receiving", "Cutoffs & cascade",
         "Sailing schedule", "Decision rules"])

    with t_work:
        st.markdown(f"#### Today's worklist — {reference_today:%b %d, %Y}")
        explainer("Work top to bottom: the most urgent groups (closest cutoff) are first. "
                  "Each line is one destination + cargo type on the floor, and the action to "
                  "take on it right now.")
        action_legend()
        render_worklist(snaps, "consol")

    with t_floor:
        st.markdown("#### What's on the floor")
        explainer("The left is a summary of pallets waiting per group (this is what you decide "
                  "on). The right is the raw receiving feed — every pallet booked in, the data "
                  "CEVA would upload to the tool.")
        if snaps:
            floor = pd.DataFrame([{
                "Destination": s["destination"], "Cargo": s["coload_class"],
                "Port": s["pod"], "Pallets": s["waiting_pallets"],
                "CBM": s["acc_cbm"], "Kg": s["acc_kg"],
                "Commit-by": f"{s['commit_by']:%b %d}" if s["commit_by"] else "—",
                "Action": s["action"],
            } for s in sorted(snaps, key=lambda x: x["days_to_commit_by"])])
            st.markdown("**Floor summary (grouped)**")
            st.dataframe(floor, use_container_width=True, hide_index=True)
        st.markdown("**Receiving feed (raw pallet data uploaded by the consol provider)**")
        recv = pallets[pallets["arrival_date"] <= reference_today].copy()
        recv = recv[["pallet_id", "sto_order", "arrival_date", "pod", "destination",
                     "coload_class", "dg_class", "bonded", "weight_kg", "cbm",
                     "stackable", "need_by"]]
        recv["arrival_date"] = recv["arrival_date"].astype(str)
        recv["need_by"] = recv["need_by"].astype(str)
        st.dataframe(recv, use_container_width=True, hide_index=True, height=340)
        st.caption(f"{len(recv)} pallets received on or before {reference_today:%b %d}. "
                   "In production this feed is the 3PL WMS receipt export.")

    with t_cut:
        st.markdown("#### The cutoffs behind every booking")
        explainer("The vessel leaves on its ETD — but your real deadline (<b>commit-by</b>) is "
                  "earlier, because customs, paperwork, drayage and stuffing all take time. "
                  "Dangerous-goods and bonded cargo need even longer, so they hit the wall first. "
                  "That's why a DG group may say ‘ship now’ while a plant group next to it can wait.")
        demo_pod = "Rotterdam"
        demo_sail = sailings[sailings.pod == demo_pod].sort_values("etd").iloc[3]
        etd = demo_sail["etd"]
        st.markdown(html_cascade(["PLANT", "DG", "BONDED"], etd, reference_today, buffer_days),
                    unsafe_allow_html=True)
        st.markdown("")
        cb_tbl = pd.DataFrame({
            "Cargo type": ["PLANT", "DG", "BONDED"],
            "Days needed before ETD": [class_lead_days(c, buffer_days) for c in ["PLANT", "DG", "BONDED"]],
            "Commit-by for this sailing": [commit_by(etd, c, buffer_days).strftime("%b %d")
                                           for c in ["PLANT", "DG", "BONDED"]],
        })
        st.dataframe(cb_tbl, use_container_width=True, hide_index=True)
        st.caption(f"Example sailing: {demo_sail.carrier} {demo_sail.vessel} from {demo_pod}, "
                   f"ETD {etd:%b %d}. Published cutoffs — CY gate-in {demo_sail.cy_cutoff:%b %d}, "
                   f"SI/VGM {demo_sail.si_vgm_cutoff:%b %d}.")

    with t_sched:
        st.markdown("#### Upcoming sailings you can book")
        explainer("The carrier schedule from today forward, with the cutoffs for each vessel. "
                  "The tool matches every group to the best sailing that still makes the plant's "
                  "need-by date.")
        up = sailings[sailings["etd"] >= reference_today].sort_values(["etd", "pod"]).copy()
        up = up[["carrier", "vessel", "voyage", "pod", "etd", "cy_cutoff",
                 "si_vgm_cutoff", "transit_days", "eta"]]
        for c in ["etd", "cy_cutoff", "si_vgm_cutoff", "eta"]:
            up[c] = up[c].astype(str)
        up = up.rename(columns={"carrier": "Carrier", "vessel": "Vessel", "voyage": "Voy",
                                "pod": "Port", "etd": "ETD", "cy_cutoff": "CY cutoff",
                                "si_vgm_cutoff": "SI/VGM", "transit_days": "Transit d",
                                "eta": "ETA"})
        st.dataframe(up, use_container_width=True, hide_index=True, height=420)
        st.caption(f"{len(up)} sailings from {reference_today:%b %d} onward across "
                   f"{sailings['pod'].nunique()} ports and {sailings['carrier'].nunique()} carriers.")

    with t_rules:
        render_rules()

st.markdown("---")
st.caption("Ocean Consolidation Optimizer · synthetic demonstration · logic proven, "
           "money pending backtest. Solver: FFD + stopping rule (OR, not AI).")
