#!/usr/bin/env python3
"""Log baseline vs Thompson Sampling metrics to a local MLflow tracking store."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "bandit_metrics.json"
MLRUNS_DIR = ROOT / "mlruns"


def _sanitize(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    with open(METRICS_PATH, encoding="utf-8") as f:
        payload = json.load(f)

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("datathon-bandit")

    with mlflow.start_run(run_name="thompson_vs_baseline") as run:
        mlflow.log_param("algorithm", "thompson_sampling")
        mlflow.log_param("arms", ",".join(payload["arms"]))
        mlflow.log_param("random_seed", payload["random_seed"])
        mlflow.log_param("n_rounds", payload["n_rounds"])
        mlflow.log_param("best_historical_arm", payload["best_historical_arm"])
        mlflow.log_param("prior_alpha", 1.0)
        mlflow.log_param("prior_beta", 1.0)

        for row in payload["metrics"]:
            policy = row["policy"]
            conversion = _sanitize(row.get("conversion"))
            exploration = _sanitize(row.get("exploration_rate"))
            if conversion is not None:
                mlflow.log_metric(f"conversion__{policy}", conversion)
            if exploration is not None:
                mlflow.log_metric(f"exploration_rate__{policy}", exploration)

        mlflow.log_metric("lift_ts_vs_legacy", payload["lift_ts_vs_legacy"])
        mlflow.log_metric("lift_ts_vs_random", payload["lift_ts_vs_random"])

        policy = payload["thompson_policy"]
        for arm, alpha in policy["alpha"].items():
            mlflow.log_metric(f"posterior_alpha__{arm}", alpha)
        for arm, beta in policy["beta"].items():
            mlflow.log_metric(f"posterior_beta__{arm}", beta)
        for arm, mean in policy["mean_estimates"].items():
            mlflow.log_metric(f"posterior_mean__{arm}", mean)

        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(ROOT / "artifacts" / "thompson_policy.json"))

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Tracking URI: {MLRUNS_DIR.resolve().as_uri()}")
        print("UI: mlflow ui --backend-store-uri ./mlruns --port 5000")


if __name__ == "__main__":
    main()
