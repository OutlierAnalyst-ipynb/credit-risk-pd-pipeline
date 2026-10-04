"""Schema + business-rule validation for loan applications."""
from __future__ import annotations

import pandas as pd

REQUIRED = {
    "application_id": "id", "application_date": "datetime", "age": "numeric",
    "annual_income": "numeric", "employment_length_years": "numeric",
    "home_ownership": "category", "loan_purpose": "category", "loan_amount": "numeric",
    "term_months": "numeric", "interest_rate": "numeric", "credit_score": "numeric",
    "dti": "numeric", "num_open_accounts": "numeric", "num_delinquencies_2y": "numeric",
    "inquiries_6m": "numeric", "revolving_utilization": "numeric",
}
NULLABLE = {"employment_length_years", "revolving_utilization"}
RANGES = {"credit_score": (300, 850), "age": (18, 120), "dti": (0, 100),
          "interest_rate": (0, 1), "loan_amount": (1, 1e6), "annual_income": (0, 1e8)}
ALLOWED = {"home_ownership": {"RENT", "MORTGAGE", "OWN"}, "term_months": {36, 60}}
MAX_NULL_SHARE = 0.25           # alert if a nullable field suddenly goes mostly missing


class DataValidationError(ValueError):
    pass


def validate_applications(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    errs: list[str] = []
    missing = set(REQUIRED) - set(df.columns)
    if require_label and "default_12m" not in df.columns:
        missing.add("default_12m")
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")

    for col, kind in REQUIRED.items():
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            errs.append(f"{col}: expected numeric, got {df[col].dtype}")
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            errs.append(f"{col}: expected datetime")
        null_share = df[col].isna().mean()
        if col not in NULLABLE and null_share > 0:
            errs.append(f"{col}: {null_share:.1%} nulls in non-nullable field")
        if col in NULLABLE and null_share > MAX_NULL_SHARE:
            errs.append(f"{col}: null share {null_share:.1%} > {MAX_NULL_SHARE:.0%}")
    if errs:
        raise DataValidationError("\n".join(errs))

    for col, (lo, hi) in RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            errs.append(f"{col}: {bad.sum()} values outside [{lo}, {hi}]")
    for col, allowed in ALLOWED.items():
        extra = set(df[col].unique()) - allowed
        if extra:
            errs.append(f"{col}: unexpected values {extra}")
    if df["application_id"].duplicated().any():
        errs.append("application_id: duplicates")
    if require_label and not set(df["default_12m"].unique()) <= {0, 1}:
        errs.append("default_12m must be 0/1")
    if errs:
        raise DataValidationError("\n".join(errs))
    return df


def load_applications(path: str, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["application_date"])
    return validate_applications(df, require_label)
