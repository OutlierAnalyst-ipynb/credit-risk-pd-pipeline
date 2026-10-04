"""Train on small data, then hit the real FastAPI app with the registered model."""
import json

import pytest

from credit_risk.models.train import run_training

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def trained(apps, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    cfg = {
        "seed": 0,
        "paths": {"registry": str(root / "reg"), "reports": str(root / "rep")},
        "split": {"oot_start": "2024-07-01"},
        "model": {"candidates": ["logreg"], "primary_metric": "roc_auc", "cv_folds": 3,
                  "n_iter_search": 2, "calibration": "sigmoid"},
        "scorecard": {"base_score": 600, "base_odds": 20, "pdo": 20},
        "decision": {"approve_min_score": 590, "decline_max_score": 560},
        "fairness": {"protected_attribute": "age_band", "min_adverse_impact_ratio": 0.0},
        "monitoring": {"psi_warn": 0.1, "psi_alert": 0.25},
    }
    result = run_training(apps, cfg)
    return cfg, result, root


def test_training_produces_reasonable_model(trained):
    _, result, root = trained
    assert result["oot_metrics"]["roc_auc"] > 0.65
    assert (root / "rep" / "model_card.md").exists()


def test_api_scores_application(trained, monkeypatch, tmp_path):
    cfg, _, root = trained
    cfg_file = tmp_path / "cfg.yaml"
    import yaml
    # paths are resolved relative to the project root, so give absolute ones
    cfg_file.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CREDIT_RISK_CONFIG", str(cfg_file))
    from credit_risk.serving.api import app

    payload = json.load(open("sample_request.json"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/v1/score", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert 0 < body["probability_of_default"] < 1
        assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
        if body["decision"] != "APPROVE":
            assert len(body["reason_codes"]) >= 1
        assert client.post("/v1/score", json={**payload, "credit_score": 999}).status_code == 422
