"""Synthetic consumer-loan applications with a 12-month default flag.

Realistic messiness on purpose: missing employment length / utilisation,
income outliers, a mild macro deterioration over time (so OOT performance
is slightly worse than in-time, like real life).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "car",
            "medical", "small_business", "other"]
PURPOSE_P = [0.42, 0.2, 0.1, 0.1, 0.05, 0.05, 0.08]
PURPOSE_RISK = {"debt_consolidation": 0.1, "credit_card": 0.0, "home_improvement": -0.2,
                "car": -0.3, "medical": 0.3, "small_business": 0.7, "other": 0.2}
STATES = ["CA", "TX", "NY", "FL", "IL", "OH", "GA", "NC", "MI", "NJ"]


def generate_applications(n: int, start_date: str, n_months: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    days = rng.integers(0, int(n_months * 30.4), n)
    app_date = start + pd.to_timedelta(days, unit="D")

    age = np.clip(rng.normal(41, 12, n), 18, 85).round().astype(int)
    credit_score = np.clip(rng.normal(690, 60, n), 300, 850).round().astype(int)
    income = np.exp(rng.normal(11.0, 0.5, n) + 0.004 * (credit_score - 690)).round(-2)
    income[rng.random(n) < 0.003] *= 20                      # data-entry outliers
    emp_len = np.clip(rng.exponential(6, n), 0, 40).round(1)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.42, 0.13])
    purpose = rng.choice(PURPOSES, n, p=PURPOSE_P)
    term = rng.choice([36, 60], n, p=[0.72, 0.28])
    loan_amt = np.clip(income * rng.uniform(0.05, 0.45, n), 1000, 40000).round(-2)
    dti = np.clip(rng.gamma(4, 4.5, n), 0, 60).round(1)
    open_acc = rng.poisson(9, n)
    delinq = rng.poisson(np.where(credit_score < 640, 0.8, 0.15))
    inq = rng.poisson(np.where(credit_score < 660, 1.6, 0.7))
    util = np.clip(rng.beta(2, 3, n) + 0.002 * (700 - credit_score), 0, 1.2).round(3)
    rate = np.clip(0.30 - 0.00035 * (credit_score - 300) + rng.normal(0, 0.015, n)
                   + np.where(term == 60, 0.02, 0), 0.05, 0.36).round(4)

    months_elapsed = (app_date - start).days / 30.4
    macro = 0.012 * months_elapsed                             # deteriorating economy
    logit = (-3.45 - 0.011 * (credit_score - 690) + 0.03 * (dti - 18) + 0.45 * delinq
             + 0.12 * inq + 1.1 * np.clip(util - 0.6, 0, None) + 0.35 * (term == 60)
             + 0.6 * np.log(loan_amt / np.clip(income, 1, None) / 0.2)
             - 0.03 * np.clip(emp_len, 0, 10) + 0.25 * (home == "RENT")
             + np.vectorize(PURPOSE_RISK.get)(purpose) + macro + rng.normal(0, 0.6, n))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "application_id": [f"A{i:08d}" for i in range(n)],
        "application_date": app_date, "age": age, "state": rng.choice(STATES, n),
        "annual_income": income, "employment_length_years": emp_len,
        "home_ownership": home, "loan_purpose": purpose, "loan_amount": loan_amt,
        "term_months": term, "interest_rate": rate, "credit_score": credit_score,
        "dti": dti, "num_open_accounts": open_acc, "num_delinquencies_2y": delinq,
        "inquiries_6m": inq, "revolving_utilization": util, "default_12m": default,
    })
    # Missing-not-at-random: self-employed often leave employment length blank
    df.loc[rng.random(n) < 0.06, "employment_length_years"] = np.nan
    df.loc[rng.random(n) < 0.08, "revolving_utilization"] = np.nan
    return df.sort_values("application_date").reset_index(drop=True)
