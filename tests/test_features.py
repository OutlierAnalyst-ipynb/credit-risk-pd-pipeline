import numpy as np
import pandas as pd

from credit_risk.features.transformers import (MODEL_INPUTS, FinancialRatios, Winsorizer,
                                               build_preprocessor)


def test_monthly_payment_matches_amortisation_formula():
    X = pd.DataFrame({"annual_income": [60000], "loan_amount": [10000], "interest_rate": [0.12],
                      "term_months": [36], "dti": [10.0], "num_delinquencies_2y": [0]})
    out = FinancialRatios().fit(X).transform(X)
    assert abs(out["monthly_payment"].iloc[0] - 332.14) < 0.01


def test_winsorizer_uses_train_quantiles_only():
    w = Winsorizer(0.0, 0.9).fit(np.arange(100).reshape(-1, 1))
    assert w.transform(np.array([[1000.0]]))[0, 0] < 100


def test_preprocessor_handles_missing_and_unseen_categories(apps):
    pre = build_preprocessor(scale=True).fit(apps[MODEL_INPUTS])
    new = apps[MODEL_INPUTS].head(5).copy()
    new["loan_purpose"] = "never_seen_before"
    new["employment_length_years"] = np.nan
    out = pre.transform(new)
    assert not np.isnan(np.asarray(out, dtype=float)).any()
