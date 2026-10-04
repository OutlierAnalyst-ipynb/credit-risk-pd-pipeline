"""Pipeline entrypoint.

    python -m credit_risk.cli generate   # synthetic applications (+ a drifted "new month")
    python -m credit_risk.cli train      # validate -> tune -> calibrate -> validate -> register
    python -m credit_risk.cli monitor    # PSI on inputs + PD vs. training reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from credit_risk.config import get_logger, load_config
from credit_risk.data.generate import generate_applications
from credit_risk.data.validate import load_applications
from credit_risk.features.transformers import MODEL_INPUTS
from credit_risk.models.registry import ModelRegistry
from credit_risk.models.train import run_training
from credit_risk.monitoring.drift import drift_report

log = get_logger("credit_risk.cli")


def cmd_generate(cfg):
    d = cfg["data"]
    apps = generate_applications(d["n_applications"], d["start_date"], d["n_months"], cfg["seed"])
    Path(cfg["paths"]["raw_applications"]).parent.mkdir(parents=True, exist_ok=True)
    apps.to_csv(cfg["paths"]["raw_applications"], index=False)
    log.info("wrote %d applications, default rate %.4f", len(apps), apps["default_12m"].mean())

    # Next month's applications: a marketing campaign pulled in riskier, thinner-file borrowers
    start_new = pd.Timestamp(d["start_date"]) + pd.DateOffset(months=d["n_months"])
    new = generate_applications(3000, str(start_new.date()), 1, cfg["seed"] + 1)
    new["credit_score"] = (new["credit_score"] - 30).clip(300, 850)
    new["inquiries_6m"] = new["inquiries_6m"] + 1
    new.drop(columns="default_12m").to_csv(cfg["paths"]["new_applications"], index=False)
    log.info("wrote %d new unlabeled applications for monitoring", len(new))


def cmd_train(cfg):
    apps = load_applications(cfg["paths"]["raw_applications"])
    log.info("validated %d applications", len(apps))
    run_training(apps, cfg)


def cmd_monitor(cfg):
    model, meta, reference = ModelRegistry(cfg["paths"]["registry"]).load()
    new = load_applications(cfg["paths"]["new_applications"], require_label=False)
    new["pd"] = model.predict_proba(new[MODEL_INPUTS])[:, 1]
    rep = drift_report(reference, new, MODEL_INPUTS + ["pd"],
                       cfg["monitoring"]["psi_warn"], cfg["monitoring"]["psi_alert"])
    print(rep.to_string(index=False))
    log.info("avg PD: training reference %.4f -> new %.4f", reference["pd"].mean(), new["pd"].mean())
    out = Path(cfg["paths"]["reports"]) / "drift_report.json"
    out.write_text(json.dumps({"model_version": meta["version"],
                               "avg_pd_reference": float(reference["pd"].mean()),
                               "avg_pd_current": float(new["pd"].mean()),
                               "features": rep.to_dict(orient="records")}, indent=2))
    if (rep["status"] == "ALERT").any():
        log.warning("ALERT-level drift detected -> notify model owner / consider recalibration")


def main():
    p = argparse.ArgumentParser(description="Consumer loan PD model pipeline")
    p.add_argument("command", choices=["generate", "train", "monitor", "all"])
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    steps = {"generate": cmd_generate, "train": cmd_train, "monitor": cmd_monitor}
    for name in (steps if args.command == "all" else [args.command]):
        log.info("===== %s =====", name.upper())
        steps[name](cfg)


if __name__ == "__main__":
    main()
