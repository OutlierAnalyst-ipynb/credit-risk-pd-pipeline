"""Adverse-action reason codes.

US lenders must tell declined applicants the principal reasons (ECOA / Reg B).
Model-agnostic approach: for each actionable input, swap the applicant's value
for a 'good-applicant' reference value and measure how much the PD drops.
The biggest drops are the reasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REASON_TEXT = {
    "credit_score": "Credit score too low",
    "dti": "Debt-to-income ratio too high",
    "revolving_utilization": "Revolving credit utilization too high",
    "num_delinquencies_2y": "Recent delinquencies on credit report",
    "inquiries_6m": "Too many recent credit inquiries",
    "loan_amount": "Requested amount too high relative to income",
    "annual_income": "Income insufficient for amount requested",
    "employment_length_years": "Length of employment",
    "term_months": "Requested loan term",
    "loan_purpose": "Loan purpose risk",
    "home_ownership": "Housing status",
}


def build_reference(dev: pd.DataFrame, target: str) -> dict:
    """Typical values among applicants who did NOT default."""
    good = dev[dev[target] == 0]
    ref = {}
    for col in REASON_TEXT:
        s = good[col]
        if col == "term_months":
            ref[col] = int(s.mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(s):
            ref[col] = float(s.median())
        else:
            ref[col] = str(s.mode().iloc[0])
    return ref


def reason_codes(model, applicant: pd.DataFrame, reference: dict, top_k: int = 3) -> list[dict]:
    base_pd = float(model.predict_proba(applicant)[:, 1][0])
    variants, names = [], []
    for col, ref_val in reference.items():
        x = applicant.copy()
        x[col] = ref_val
        variants.append(x)
        names.append(col)
    pds = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    impact = base_pd - pds                      # positive = this feature is hurting the applicant
    order = np.argsort(-impact)
    return [{"feature": names[i], "reason": REASON_TEXT[names[i]],
             "pd_impact": round(float(impact[i]), 4)}
            for i in order[:top_k] if impact[i] > 0.001]
