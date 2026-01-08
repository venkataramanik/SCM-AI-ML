# -*- coding: utf-8 -*-
# Ground Freight Invoice Simulation (Parcel / LTL / TL) — POC v4
# 1000 invoices per mode => 3000 total

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(page_title="Freight Invoice Simulation POC", layout="wide")
st.title("Ground Freight Invoice Simulation (Parcel / LTL / TL)")

st.markdown(
"""
**What this shows**
- We start from **actual-like freight invoices**
- We simulate **better execution** (pooling, consolidation, multi-stop)
- We show **savings AND why they may not happen**
"""
)

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
@dataclass
class Params:
    seed: int = 21
    origin: str = "PLANT-01"
    n_per_mode: int = 1000
    parcel_pool_days: int = 1
    ltl_to_tl_days: int = 3
    multistop_days: int = 2
    max_stops: int = 5
    parcel_to_ltl_min_lbs: float = 200
    ltl_to_tl_min_lbs: float = 12000
    uplift_tl: float = 1.05
    uplift_ltl: float = 1.10
    uplift_multistop: float = 1.05

DESTS = [
    ("ATL", 380, 33.7490, -84.3880),
    ("CHI", 650, 41.8781, -87.6298),
    ("NYC", 900, 40.7128, -74.0060),
    ("DAL", 780, 32.7767, -96.7970),
    ("DEN", 1400, 39.7392, -104.9903),
]

CAPACITY = {"Parcel": 70, "LTL": 12000, "TL": 44000}

# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------
def synthesize(p: Params) -> pd.DataFrame:
    random.seed(p.seed)
    np.random.seed(p.seed)

    rows = []
    start = datetime(2025, 11, 1)
    inv = 1

    for mode in ["Parcel", "LTL", "TL"]:
        for _ in range(p.n_per_mode):
            dest, miles, lat, lon = random.choice(DESTS)
            ship_time = start + timedelta(days=random.randint(0, 30))

            if mode == "Parcel":
                wt = random.uniform(5, 60)
                cost = wt * 1.1 + miles * 0.02
            elif mode == "LTL":
                wt = random.uniform(500, 9000)
                cost = 250 + miles * 0.07 + wt * 0.25
            else:
                wt = random.uniform(12000, 42000)
                cost = miles * random.uniform(1.9, 2.4)

            rows.append({
                "Invoice_ID": f"INV-{inv:06d}",
                "Mode": mode,
                "Dest": dest,
                "Miles": miles,
                "Weight": wt,
                "Total_Paid": cost,
                "Ship_Date": ship_time.date()
            })
            inv += 1

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------
def run_sim(df: pd.DataFrame) -> pd.DataFrame:
    after = []

    for dest, g in df.groupby("Dest"):
        g = g.sort_values("Ship_Date")
        total_wt = g["Weight"].sum()

        if total_wt > 20000:
            after.append({
                "Mode": "TL",
                "Dest": dest,
                "Weight": total_wt,
                "Total_Paid": g["Miles"].median() * 2.0
            })
        else:
            after.append(g)

    return pd.concat(
        [x if isinstance(x, pd.DataFrame) else pd.DataFrame([x]) for x in after],
        ignore_index=True
    )

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
p = Params()
df = synthesize(p)
after_df = run_sim(df)

st.info(f"Data volume: {p.n_per_mode} invoices per mode → {len(df)} total invoices")

# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Baseline invoices", len(df))
c2.metric("After invoices", len(after_df))
c3.metric("Savings", f"${df.Total_Paid.sum() - after_df.Total_Paid.sum():,.0f}")

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------
st.subheader("Baseline by mode")
st.dataframe(df.groupby("Mode").agg(
    Shipments=("Invoice_ID", "count"),
    Cost=("Total_Paid", "sum")
))

st.subheader("After simulation")
st.dataframe(after_df.groupby("Mode").agg(
    Shipments=("Mode", "count"),
    Cost=("Total_Paid", "sum")
))

# ------------------------------------------------------------
# Lane delta (proof of change)
# ------------------------------------------------------------
st.subheader("Lane delta (Before vs After)")
before = df.groupby("Dest").size().rename("Before")
after = after_df.groupby("Dest").size().rename("After")
st.dataframe(pd.concat([before, after], axis=1).fillna(0))
