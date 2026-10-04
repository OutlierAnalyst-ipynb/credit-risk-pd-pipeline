# Consumer Loan Probability-of-Default (PD) Pipeline

An end-to-end credit-risk modelling system in the style of a bank's consumer lending
model team (JPMC / Wells Fargo): data contracts → sklearn feature pipeline with custom
transformers → hyperparameter search → probability calibration → model-validation pack
(Gini, KS, deciles, OOT) → fair-lending test → governance gates → registry →
real-time **FastAPI** decisioning service with adverse-action reason codes → PSI monitoring.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all          # generate -> train -> monitor  (~1 min)
make test
make serve        # http://localhost:8000/docs
curl -X POST localhost:8000/v1/score -H "Content-Type: application/json" -d @sample_request.json
```

After training, read `reports/model_card.md`; that's the document a Model Risk
Management (MRM) team would review before the model goes live.

Docker:
```bash
docker build -t credit-pd .
docker run -p 8000:8000 -v $(pwd)/model_registry:/app/model_registry credit-pd
```

## Architecture

```
applications.csv
  │ data/validate.py         schema, ranges, allowed values, missingness spikes
  ▼
features/transformers.py     ONE sklearn Pipeline used in training AND serving:
  │                          FinancialRatios (payment, PTI, total DTI) -> Winsorizer
  │                          -> SimpleImputer(+missing flags) -> Scaler / OneHotEncoder
  ▼
models/train.py              out-of-time split (dev < 2025-01-01 <= OOT)
  │                          RandomizedSearchCV: LogisticRegression vs
  │                          HistGradientBoosting w/ monotonic constraints
  │                          CalibratedClassifierCV (isotonic) -> real probabilities
  ▼
validation pack              AUC, Gini, KS, Brier, calibration ratio, decile table,
  │                          permutation importance, adverse impact ratio by age band
  ▼
governance gates             OOT AUC, calibration, dev→OOT stability, fairness
  │                          -> only promoted if ALL pass and it beats the champion
  ▼
models/registry.py  ──►  serving/api.py   POST /v1/score -> PD, score points,
  │                                        APPROVE / REVIEW / DECLINE, reason codes
  ▼
monitoring/drift.py          PSI on every input + the PD itself
```

## Concepts this project teaches

1. **Custom sklearn transformers** (`FinancialRatios`, `Winsorizer`) that plug into
   `Pipeline`/`ColumnTransformer` and keep feature names (`set_output(transform="pandas")`).
2. **Out-of-time validation.** The synthetic economy deteriorates over time, so OOT default
   rate is higher than dev, exactly the situation that bites real credit models.
3. **Calibration.** Ranking (AUC) isn't enough; PDs feed pricing, limits and CECL/IFRS 9
   loss reserves, so they must be calibrated. Watch `calibration_ratio` in the report.
4. **Monotonic constraints.** The GBM is forced to treat higher credit score as lower risk.
   Regulators reject models whose behaviour can't be explained.
5. **Scorecard scaling.** PD → points with base score / odds / PDO, then policy cut-offs.
6. **Fair lending.** Age is *never* a model input, but approvals are tested by age band with
   the four-fifths rule. A failing gate blocks promotion.
7. **Adverse-action reason codes.** Declined applicants get the top reasons (ECOA / Reg B).
8. **Serving.** Pydantic validates requests (bad credit score → HTTP 422), the model loads
   once at startup, and the response carries the model version for audit.

## Practice exercises

- Implement **WoE/IV binning** and a classic logistic scorecard; compare to the current model.
- Swap permutation-based reason codes for **SHAP** and compare the reasons.
- Add a `/v1/score/batch` endpoint and a latency test.
- Add a **reject-inference** step (declined applicants have no outcome labels; how do you fix
  that bias?).
- Add **MLflow** tracking for every RandomizedSearchCV trial.
- Add an **LGD** or **EAD** model and combine into Expected Loss = PD × LGD × EAD.
- Schedule `monitor` monthly and auto-open a ticket when PSI > 0.25.
- Test for proxy discrimination: does `state` or `home_ownership` correlate with age band?
