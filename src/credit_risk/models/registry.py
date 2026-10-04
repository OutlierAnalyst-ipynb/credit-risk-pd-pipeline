"""File-based model registry (stand-in for MLflow Model Registry)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

MODEL_NAME = "credit_pd"


class ModelRegistry:
    def __init__(self, root: str):
        self.root = Path(root) / MODEL_NAME
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model, metadata: dict, reference: pd.DataFrame) -> str:
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S_%f")
        d = self.root / version
        d.mkdir()
        joblib.dump(model, d / "model.joblib")
        reference.to_csv(d / "reference.csv", index=False)
        meta = {**metadata, "version": version,
                "registered_at": datetime.now(timezone.utc).isoformat()}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return version

    def production_version(self) -> str | None:
        p = self.root / "PRODUCTION"
        return p.read_text().strip() if p.exists() else None

    def promote(self, version: str) -> None:
        (self.root / "PRODUCTION").write_text(version)

    def load(self, version: str | None = None):
        version = version or self.production_version()
        if version is None:
            raise FileNotFoundError("No PRODUCTION model registered. Run `make train`.")
        d = self.root / version
        return (joblib.load(d / "model.joblib"),
                json.loads((d / "metadata.json").read_text()),
                pd.read_csv(d / "reference.csv"))

    def maybe_promote(self, version: str, metric: str, gates: dict[str, bool]) -> bool:
        """Promote only if all governance gates pass AND it beats the champion on OOT."""
        if not all(gates.values()):
            return False
        champ = self.production_version()
        if champ is not None:
            _, cm, _ = self.load(champ)
            _, nm, _ = self.load(version)
            if nm["oot_metrics"][metric] < cm["oot_metrics"][metric]:
                return False
        self.promote(version)
        return True
