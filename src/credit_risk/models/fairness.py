"""Fair-lending check: adverse impact ratio (AIR) across groups.

AIR = approval rate of group / approval rate of most-favoured group.
< 0.80 (four-fifths rule) is a red flag that compliance must review.
The protected attribute is NEVER a model input, only used here.
"""
from __future__ import annotations

import pandas as pd


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [17, 24, 39, 61, 200], labels=["18-24", "25-39", "40-61", "62+"]).astype(str)


def adverse_impact(groups: pd.Series, approved: pd.Series) -> pd.DataFrame:
    rates = approved.groupby(groups.values).agg(["size", "mean"])
    rates.columns = ["n", "approval_rate"]
    rates["air"] = rates["approval_rate"] / rates["approval_rate"].max()
    return rates.round(4)
