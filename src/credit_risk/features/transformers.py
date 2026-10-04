"""Custom scikit-learn transformers + the full preprocessing graph.

Everything lives inside one sklearn Pipeline so the exact same transformations
run in training, batch scoring and the real-time API (no train/serve skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERIC = ["annual_income", "employment_length_years", "loan_amount", "term_months",
               "interest_rate", "credit_score", "dti", "num_open_accounts",
               "num_delinquencies_2y", "inquiries_6m", "revolving_utilization"]
RAW_CATEGORICAL = ["home_ownership", "loan_purpose"]
MODEL_INPUTS = RAW_NUMERIC + RAW_CATEGORICAL     # NOTE: no age, no state (fair lending)
DERIVED = ["log_income", "loan_to_income", "monthly_payment", "payment_to_income",
           "total_dti", "has_delinquency"]
TARGET = "default_12m"


class FinancialRatios(BaseEstimator, TransformerMixin):
    """Adds underwriting ratios. Stateless, so fit() learns nothing."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        income = X["annual_income"].clip(lower=1)
        r = X["interest_rate"] / 12
        n = X["term_months"]
        X["monthly_payment"] = X["loan_amount"] * r / (1 - (1 + r) ** (-n))   # amortisation
        X["log_income"] = np.log1p(income)
        X["loan_to_income"] = X["loan_amount"] / income
        X["payment_to_income"] = X["monthly_payment"] * 12 / income
        X["total_dti"] = X["dti"] + 100 * X["payment_to_income"]             # DTI incl. new loan
        X["has_delinquency"] = (X["num_delinquencies_2y"] > 0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        base = self.feature_names_in_ if input_features is None else input_features
        return np.array(list(base) + DERIVED, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips numeric columns to quantiles learned on TRAINING data only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features,
                          dtype=object)


def build_preprocessor(scale: bool) -> Pipeline:
    numeric = RAW_NUMERIC + DERIVED
    # clip first (NaN-safe), then impute + add "was missing" flags (missingness is predictive)
    num_steps = [("clip", Winsorizer()),
                 ("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    columns = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=0.01), RAW_CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("ratios", FinancialRatios()), ("columns", columns)])
