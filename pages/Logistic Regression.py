#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression — Business-Focused Explainer (with tiny demo)
Author: ChatGPT
Run:
  python logistic_regression_business_explainer.py
Options:
  --plot    Save a simple sigmoid & decision boundary plot to ./logreg_demo.png
  --quiet   Print fewer details
No external dependencies beyond numpy and matplotlib (optional for --plot).
"""

import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import numpy as np
except ImportError as e:
    print("This script requires NumPy. Please install it (pip install numpy) and try again.")
    sys.exit(1)

# Matplotlib is optional (only for --plot)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# Section 1: Plain-English Primer
# -----------------------------
PRIMER = """
🧭 LOGISTIC REGRESSION — Business Explainer

What it does:
    • Predicts the LIKELIHOOD (probability) that an outcome will happen (e.g., "on-time delivery").
    • Outputs a score from 0 to 1 so you can rank by risk and act proactively.

Why leaders use it:
    • Transparent and explainable (weights show which factors raise/lower risk).
    • Fast to train; works well with mid-size data; great baseline for risk scoring.
    • Easy to operationalize (threshold or rank-based actions).

When to use it:
    • Binary outcomes (yes/no): on-time vs late, churn vs retain, defect vs pass, fraud vs legit.
    • When relationships are smooth and mostly linear in the *log-odds* space.
    • When interpretability matters (compliance, audits, stakeholder trust).
"""

PENSKE_STORY = """
🚛 Penske-Style Example (Business Narrative)

Goal:
    Reduce late deliveries by proactively spotting at-risk loads.

Inputs a dispatcher already watches:
    • Distance (miles)
    • Weather (bad=1, clear=0)
    • Driver experience (years)
    • Load type (fragile=1, normal=0)
    • Day of week / Time window

What the model gives you:
    • A probability for each shipment being on-time (e.g., 0.86 = 86%).
How you use it:
    • >0.90 → standard handling
    • 0.60–0.90 → notify customer / watchlist
    • <0.60 → reroute, swap driver, or adjust promise time
"""


# -----------------------------
# Section 2: Tiny Hands-On Demo (from scratch, with NumPy)
# -----------------------------
@dataclass
class LogRegConfig:
    lr: float = 0.1           # learning rate
    epochs: int = 2000        # training iterations
    fit_intercept: bool = True
    random_state: int = 42


class SimpleLogisticRegression:
    """
    Minimal logistic regression implemented with NumPy.
    • Binary classification only (labels 0/1).
    • Gradient descent on cross-entropy loss.
    """
    def __init__(self, config: LogRegConfig = LogRegConfig()):
        self.cfg = config
        self.w: Optional[np.ndarray] = None  # weights (including intercept if fit_intercept=True)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack([intercept, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLogisticRegression":
        rng = np.random.default_rng(self.cfg.random_state)
        Xb = self._prepare_X(X).astype(float)
        y = y.astype(float).reshape(-1, 1)

        # Initialize weights
        self.w = rng.normal(loc=0.0, scale=0.01, size=(Xb.shape[1], 1))

        for _ in range(self.cfg.epochs):
            logits = Xb @ self.w
            probs = self._sigmoid(logits)
            # Cross-entropy loss gradient
            grad = Xb.T @ (probs - y) / Xb.shape[0]
            self.w -= self.cfg.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None, "Model not trained"
        Xb = self._prepare_X(X).astype(float)
        probs = self._sigmoid(Xb @ self.w)
        return probs.ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def coefficients(self) -> np.ndarray:
        assert self.w is not None, "Model not trained"
        return self.w.ravel()


def make_synthetic_penske(n: int = 400, seed: int = 7) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a toy dataset mimicking logistics risk:
      X[:,0] -> distance (miles) ~ N(300, 100)
      X[:,1] -> weather (0 clear, 1 bad) ~ Bernoulli(p=0.35)
      X[:,2] -> driver experience (years) ~ Uniform[0,10]
    Label y: 1=On-Time, 0=Late, probabilistically generated via a hidden logistic rule.
    """
    rng = np.random.default_rng(seed)
    distance = rng.normal(300, 100, n)
    weather = rng.binomial(1, 0.35, n)
    exp_years = rng.uniform(0, 10, n)

    # Hidden "true" weights for generating labels
    # Intercept=2.0, distance weight = -0.006, weather weight = -1.0, experience weight = +0.25
    z = 2.0 + (-0.006)*distance + (-1.0)*weather + (0.25)*exp_years
    p_on_time = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p_on_time).astype(int)

    X = np.c_[distance, weather, exp_years]
    feature_names = ["distance_miles", "weather_bad(1/0)", "driver_experience_years"]
    return X, y, feature_names


def train_and_report(quiet: bool = False, make_plot: bool = False) -> None:
    X, y, feature_names = make_synthetic_penske(n=500, seed=11)
    model = SimpleLogisticRegression(LogRegConfig(lr=0.1, epochs=3000, fit_intercept=True, random_state=1))
    model.fit(X, y)

    # Coefficients
    coefs = model.coefficients()
    if not quiet:
        print("\n🔧 TRAINED MODEL COEFFICIENTS (higher → increases odds of 'On-Time')")
        for i, name in enumerate(["(intercept)"] + feature_names):
            print(f"  {name:>28s} : {coefs[i]: .4f}")

    # Score a few example loads (distance, weather, exp)
    samples = np.array([
        [120, 0, 6.0],   # short, clear, solid experience → likely on-time
        [420, 1, 1.0],   # long, bad, inexperienced → risky
        [300, 0, 2.5],   # mid, clear, low exp → moderate
        [280, 1, 9.0],   # mid, bad, very experienced → experience offsets weather somewhat
    ])
    probs = model.predict_proba(samples)

    print("\n📊 SAMPLE SCORING (P(On-Time))")
    for row, p in zip(samples, probs):
        print(f"  distance={row[0]:.0f}  weather_bad={int(row[1])}  exp_years={row[2]:.1f}  →  p_on_time={p:0.3f}")

    # Lightweight quality metric
    preds = model.predict(X, threshold=0.5)
    accuracy = (preds == y).mean()
    if not quiet:
        print(f"\n✅ Training accuracy (toy data, not cross-validated): {accuracy:.3f}")

    # Optional plot
    if make_plot and HAS_MPL:
        try:
            fig = plt.figure(figsize=(6, 5))
            # Plot probability vs. distance for a fixed weather/experience
            distances = np.linspace(50, 600, 200)
            W = np.zeros_like(distances)        # clear weather
            EXP = np.full_like(distances, 5.0)  # 5 yrs experience
            grid = np.c_[distances, W, EXP]
            pgrid = model.predict_proba(grid)

            plt.plot(distances, pgrid, label="p(On-Time) vs Distance (clear, 5yr exp)")
            plt.axhline(0.5, linestyle="--", label="0.5 threshold")
            plt.xlabel("Distance (miles)")
            plt.ylabel("Predicted Probability of On-Time")
            plt.title("Logistic Curve in a Logistics Context")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig("logreg_demo.png", dpi=160)
            print("\n🖼  Saved plot: logreg_demo.png")
        except Exception as e:
            print(f"(Plotting skipped due to error: {e})")
    elif make_plot and not HAS_MPL:
        print("Matplotlib not available — skipping plot.")


# -----------------------------
# Section 3: Historical Tidbits & Trivia
# -----------------------------
HISTORY = """
📜 Historical Tidbits

• The "logistic" function (that S-shaped curve) was introduced by Pierre-François Verhulst in 1838
  to model population growth that saturates (carrying capacity).

• The "logit" name and practical use in analysis grew in the 20th century; Joseph Berkson popularized
  the logit usage in 1944, helping bridge statistics and real-world experiments.

• Modern logistic regression relies on maximum likelihood estimation (mid-20th century), and became
  a mainstay for medical studies, credit scoring, and marketing response modeling long before the
  "machine learning" term went mainstream.

• Business fun fact: scoring/ranking customers by purchase or churn probability predates many modern
  recommender systems — logistic regression was (and remains) a workhorse for direct marketing.
"""


# -----------------------------
# Section 4: Practical Playbook (Business-First)
# -----------------------------
PLAYBOOK = """
🧰 Practical Playbook (Business-First)

1) Define the decision:
   • What action will you take at different probability bands? (e.g., <60% → reroute)

2) Choose tight, actionable features:
   • Distance, weather, time window, driver tenure, facility congestion index.
   • Make them timely and available in production.

3) Make targets clean:
   • Binary label that's consistent (e.g., on-time = arrived within SLA).

4) Sanity checks:
   • Outliers? Missing values? Data drift by season or region?
   • Correlated inputs (e.g., distance and time)? Document and monitor.

5) Ship it with monitoring:
   • Track calibration (do 0.8 scores happen ~80% of the time?).
   • Retrain cadence (e.g., monthly/quarterly).
   • Keep a rules fallback for outages.
"""


# -----------------------------
# CLI Entrypoint
# -----------------------------
def main(argv: List[str]) -> None:
    want_plot = "--plot" in argv
    quiet = "--quiet" in argv

    print(PRIMER)
    print(PENSKE_STORY)
    print("—" * 72)
    print("Now a tiny hands-on demo (synthetic data, NumPy only):")
    train_and_report(quiet=quiet, make_plot=want_plot)
    print("—" * 72)
    print(HISTORY)
    print(PLAYBOOK)
    print("\nDone. Tip: run with --plot to save a simple visualization to logreg_demo.png\n")


if __name__ == "__main__":
    main(sys.argv[1:])
