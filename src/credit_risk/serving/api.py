"""Real-time decisioning API.

    uvicorn credit_risk.serving.api:app --port 8000
    curl -X POST localhost:8000/v1/score -H 'Content-Type: application/json' -d @sample_request.json
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.config import load_config
from credit_risk.features.transformers import MODEL_INPUTS
from credit_risk.models.explain import reason_codes
from credit_risk.models.registry import ModelRegistry
from credit_risk.models.scorecard import decide, pd_to_score

STATE: dict = {}


class LoanApplication(BaseModel):
    application_id: str
    annual_income: float = Field(gt=0)
    employment_length_years: float | None = Field(default=None, ge=0, le=60)
    home_ownership: Literal["RENT", "MORTGAGE", "OWN"]
    loan_purpose: str
    loan_amount: float = Field(gt=0, le=1_000_000)
    term_months: Literal[36, 60]
    interest_rate: float = Field(gt=0, lt=1)
    credit_score: int = Field(ge=300, le=850)
    dti: float = Field(ge=0, le=100)
    num_open_accounts: int = Field(ge=0)
    num_delinquencies_2y: int = Field(ge=0)
    inquiries_6m: int = Field(ge=0)
    revolving_utilization: float | None = Field(default=None, ge=0, le=2)


class Reason(BaseModel):
    feature: str
    reason: str
    pd_impact: float


class Decision(BaseModel):
    application_id: str
    probability_of_default: float
    credit_score_points: int
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    reason_codes: list[Reason]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(os.getenv("CREDIT_RISK_CONFIG"))
    model, meta, _ = ModelRegistry(cfg["paths"]["registry"]).load()
    STATE.update(model=model, meta=meta)
    yield
    STATE.clear()


app = FastAPI(title="Consumer Loan PD Service", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE.get("meta", {}).get("version")}


@app.post("/v1/score", response_model=Decision)
def score(application: LoanApplication) -> Decision:
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    model, meta = STATE["model"], STATE["meta"]
    X = pd.DataFrame([application.model_dump()])[MODEL_INPUTS]
    X = X.astype({"employment_length_years": float, "revolving_utilization": float})
    pd_ = float(model.predict_proba(X)[:, 1][0])
    sc, dc = meta["scorecard"], meta["decision"]
    points = int(pd_to_score([pd_], sc["base_score"], sc["base_odds"], sc["pdo"])[0])
    decision = str(decide([points], dc["approve_min_score"], dc["decline_max_score"])[0])
    reasons = (reason_codes(model, X, meta["reason_code_reference"])
               if decision != "APPROVE" else [])
    return Decision(application_id=application.application_id,
                    probability_of_default=round(pd_, 5), credit_score_points=points,
                    decision=decision, reason_codes=reasons, model_version=meta["version"])
