"""Recommendation helpers for the FastAPI service and Golden Set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from src.bandit import (
    CONTEXT_CAT,
    CONTEXT_NUM,
    DEFAULT_ARMS,
    ThompsonSamplingPolicy,
    predict_arm_probabilities,
)

Mode = Literal["exploit", "explore"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_policy(path: Optional[Path] = None) -> ThompsonSamplingPolicy:
    policy_path = path or (project_root() / "artifacts" / "thompson_policy.json")
    with open(policy_path, encoding="utf-8") as f:
        payload = json.load(f)
    return ThompsonSamplingPolicy.from_dict(payload)


def load_reward_model(path: Optional[Path] = None):
    model_path = path or (project_root() / "artifacts" / "reward_model.joblib")
    return joblib.load(model_path)


def customer_to_frame(customer: Dict[str, Any]) -> pd.DataFrame:
    row = {col: customer.get(col) for col in CONTEXT_NUM + CONTEXT_CAT}
    return pd.DataFrame([row])


def recommend(
    customer: Dict[str, Any],
    policy: Optional[ThompsonSamplingPolicy] = None,
    reward_model=None,
    mode: Mode = "exploit",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return recommended contact channel and conversion probabilities."""
    policy = policy or load_policy()
    reward_model = reward_model or load_reward_model()

    context = customer_to_frame(customer)
    arm_probs = predict_arm_probabilities(reward_model, context, arms=DEFAULT_ARMS)
    conversion_by_arm = {arm: float(probs[0]) for arm, probs in arm_probs.items()}

    if mode == "explore":
        rng = np.random.default_rng(seed)
        offered_channel = policy.select_arm(rng)
    else:
        offered_channel = policy.select_greedy()

    mean_estimates = policy.mean_estimates()
    return {
        "recommended_offer": offered_channel,
        "conversion_probability": conversion_by_arm[offered_channel],
        "conversion_by_arm": conversion_by_arm,
        "policy_mean_estimates": mean_estimates,
        "algorithm": "thompson_sampling",
        "mode": mode,
        "rationale": (
            f"Canal '{offered_channel}' selecionado via Thompson Sampling "
            f"(mode={mode}); estimativa média da política="
            f"{mean_estimates[offered_channel]:.4f}."
        ),
    }
