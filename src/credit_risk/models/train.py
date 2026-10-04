"""Training: OOT split -> tuned candidates -> calibration -> validation pack -> gates -> registry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk.config import get_logger
from credit_risk.features.transformers import MODEL_INPUTS, TARGET, build_preprocessor
from credit_risk.models.explain import build_reference
from credit_risk.models.fairness import adverse_impact, age_band
from credit_risk.models.metrics import classification_report, decile_table
from credit_risk.models.registry import ModelRegistry
from credit_risk.models.scorecard import decide, pd_to_score

log = get_logger(__name__)

# Business sense constraints: risk must never go DOWN as delinquencies go UP, etc.
# Regulators/MRM push back hard on models that violate these.
MONOTONIC = {"credit_score": -1, "num_delinquencies_2y": 1, "dti": 1, "inquiries_6m": 1}


def build_candidates(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    logreg = Pipeline([("pre", build_preprocessor(scale=True)),
                       ("clf", LogisticRegression(max_iter=3000))])
    hgb = Pipeline([("pre", build_preprocessor(scale=False)),
                    ("clf", HistGradientBoostingClassifier(monotonic_cst=MONOTONIC,
                                                           random_state=seed))])
    for p in (logreg, hgb):
        p.named_steps["pre"].set_output(transform="pandas")   # keep feature names
    return {
        "logreg": (logreg, {"clf__C": loguniform(1e-3, 10)}),
        "hgb": (hgb, {"clf__learning_rate": [0.03, 0.05, 0.1],
                      "clf__max_leaf_nodes": [7, 15, 31],
                      "clf__min_samples_leaf": [50, 100, 200],
                      "clf__l2_regularization": [0.0, 1.0, 5.0],
                      "clf__max_iter": [150, 300]}),
    }


def run_training(apps: pd.DataFrame, cfg: dict) -> dict:
    mcfg, sc, dc = cfg["model"], cfg["scorecard"], cfg["decision"]
    dev = apps[apps["application_date"] < cfg["split"]["oot_start"]]
    oot = apps[apps["application_date"] >= cfg["split"]["oot_start"]]
    log.info("dev rows=%d DR=%.4f | oot rows=%d DR=%.4f",
             len(dev), dev[TARGET].mean(), len(oot), oot[TARGET].mean())

    cv = StratifiedKFold(mcfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    # 1) Hyperparameter search per candidate (CV on development sample only)
    searches = {}
    for name, (pipe, grid) in build_candidates(cfg["seed"]).items():
        if name not in mcfg["candidates"]:
            continue
        search = RandomizedSearchCV(pipe, grid, n_iter=mcfg["n_iter_search"], scoring="roc_auc",
                                    cv=cv, n_jobs=-1, random_state=cfg["seed"], refit=False)
        search.fit(dev[MODEL_INPUTS], dev[TARGET])
        searches[name] = search
        log.info("candidate=%-6s cv_auc=%.4f params=%s", name, search.best_score_,
                 search.best_params_)

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best_pipe = clone(build_candidates(cfg["seed"])[best_name][0]).set_params(
        **searches[best_name].best_params_)
    log.info("selected: %s", best_name)

    # 2) Calibrate (PDs feed pricing, limits, loss forecasting -> must be probabilities)
    model = CalibratedClassifierCV(best_pipe, method=mcfg["calibration"], cv=cv)
    model.fit(dev[MODEL_INPUTS], dev[TARGET])

    # 3) Validation pack on out-of-time sample
    p_oot = model.predict_proba(oot[MODEL_INPUTS])[:, 1]
    oot_m = classification_report(oot[TARGET], p_oot)
    dev_m = classification_report(dev[TARGET], model.predict_proba(dev[MODEL_INPUTS])[:, 1])
    deciles = decile_table(oot[TARGET], p_oot)
    log.info("OOT auc=%.4f gini=%.4f ks=%.4f brier=%.4f calib_ratio=%.3f",
             oot_m["roc_auc"], oot_m["gini"], oot_m["ks"], oot_m["brier"],
             oot_m["calibration_ratio"])

    scores = pd_to_score(p_oot, sc["base_score"], sc["base_odds"], sc["pdo"])
    decisions = decide(scores, dc["approve_min_score"], dc["decline_max_score"])
    decision_mix = pd.Series(decisions).value_counts(normalize=True).round(4).to_dict()
    approved = pd.Series(decisions == "APPROVE", index=oot.index)
    bad_rate_approved = float(oot.loc[approved, TARGET].mean())

    fair = adverse_impact(age_band(oot["age"]), approved)
    min_air = float(fair["air"].min())

    sample = oot.sample(min(4000, len(oot)), random_state=cfg["seed"])
    pi = permutation_importance(model, sample[MODEL_INPUTS], sample[TARGET], scoring="roc_auc",
                                n_repeats=3, random_state=cfg["seed"], n_jobs=-1)
    importance = (pd.Series(pi.importances_mean, index=MODEL_INPUTS)
                    .sort_values(ascending=False).round(4))

    # 4) Governance gates -> promotion
    gates = {
        "oot_auc>=0.70": oot_m["roc_auc"] >= 0.70,
        "calibration_ratio_in_[0.8,1.25]": 0.8 <= oot_m["calibration_ratio"] <= 1.25,
        "auc_drop_dev_to_oot<0.05": dev_m["roc_auc"] - oot_m["roc_auc"] < 0.05,
        f"min_AIR>={cfg['fairness']['min_adverse_impact_ratio']}":
            min_air >= cfg["fairness"]["min_adverse_impact_ratio"],
    }
    for g, ok in gates.items():
        log.info("gate %-35s %s", g, "PASS" if ok else "FAIL")

    reference = dev[MODEL_INPUTS].sample(min(20000, len(dev)), random_state=cfg["seed"]).copy()
    reference["pd"] = model.predict_proba(reference[MODEL_INPUTS])[:, 1]
    metadata = {
        "model_type": best_name, "best_params": searches[best_name].best_params_,
        "cv_auc": {n: float(s.best_score_) for n, s in searches.items()},
        "dev_metrics": dev_m, "oot_metrics": oot_m, "gates": gates,
        "decision_mix_oot": decision_mix, "bad_rate_approved_oot": bad_rate_approved,
        "fairness_age_band": fair.reset_index(names="age_band").to_dict(orient="records"),
        "permutation_importance": importance.to_dict(),
        "reason_code_reference": build_reference(dev, TARGET),
        "scorecard": sc, "decision": dc, "inputs": MODEL_INPUTS,
    }
    registry = ModelRegistry(cfg["paths"]["registry"])
    version = registry.register(model, metadata, reference)
    promoted = registry.maybe_promote(version, mcfg["primary_metric"], gates)
    log.info("registered %s | promoted=%s", version, promoted)

    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_report.json").write_text(json.dumps(
        {"version": version, "promoted": promoted, **metadata}, indent=2, default=str))
    write_model_card(reports / "model_card.md", version, promoted, metadata, deciles, fair,
                     importance)
    return {"version": version, "promoted": promoted, "oot_metrics": oot_m, "gates": gates}


def write_model_card(path: Path, version, promoted, meta, deciles, fair, importance) -> None:
    m = meta["oot_metrics"]
    lines = [
        f"# Model Card: Consumer Loan PD Model `{version}`", "",
        f"**Status:** {'PRODUCTION' if promoted else 'NOT PROMOTED'}  ",
        f"**Algorithm:** {meta['model_type']} + isotonic calibration  ",
        "**Target:** default within 12 months of origination  ",
        "**Intended use:** application decisioning (approve / review / decline) for unsecured "
        "personal loans. Not for use on other products without revalidation.", "",
        "## Out-of-time performance",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in m.items()],
        "", "## Governance gates",
        *[f"- {'✅' if ok else '❌'} {g}" for g, ok in meta["gates"].items()],
        "", "## Rank ordering (OOT deciles, 10 = riskiest)", deciles.to_markdown()
        if _has_tabulate() else "```\n" + deciles.to_string() + "\n```",
        "", "## Fair lending: adverse impact ratio by age band",
        "```\n" + fair.to_string() + "\n```",
        "", "## Permutation importance (drop in OOT AUC)",
        "```\n" + importance.to_string() + "\n```",
        "", f"Decision mix on OOT: {meta['decision_mix_oot']}  ",
        f"Bad rate among approved: {meta['bad_rate_approved_oot']:.4f}", "",
        "## Limitations",
        "- Trained on synthetic data; real deployment needs bureau data and a full MRM review.",
        "- Age/state are excluded as inputs; proxy discrimination should still be tested.",
        "- Macro conditions shift default rates; monitor PSI and calibration monthly.",
    ]
    path.write_text("\n".join(lines))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
